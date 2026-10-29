"""
SRRT -- Semantic Rectified Restoration Transformer (full integration).

Pipeline
--------
        y (degraded)
         |
         +--------------------+
         |                    |
      SeeCoder             DiffAE            (F_v = E_v(y),  F_d = E_d(y))
         |                    |
         +---------+----------+
                   |
                  LCR                        (z_0 = L(F_v, F_d))
                   |
            Rectified Flow                   (z_0 --ODE--> z_1_hat)
                   |
        Restormer backbone                   (latent injected at bottleneck)
                   |
                  SRA                        (F_rect = F_R + gamma * A_s (*) R(F_R))
                   |
              x_hat (restored)

Integration strategy (documented design decision)
-------------------------------------------------
The Restormer backbone is NOT modified. SRRT holds a Restormer instance and
calls its encoder / bottleneck / decoder / refinement / output submodules in
order, splicing the SRRT components in between. This keeps the baseline
bit-identical and therefore keeps the ablation honest -- with every module
disabled, SRRT(y) is numerically identical to Restormer(y) (asserted in
tests/test_srrt.py).

The SRRT latent path meets the backbone at the BOTTLENECK, which is the natural
junction: it is the lowest-resolution, most semantic tensor (dim*8, H/8, W/8),
so latent transport is cheap there, and the decoder can still propagate the
modification back to full resolution through the skip connections.

Latent injection uses a ZERO-INITIALIZED gate:

    latent' = latent + gamma_latent * W_inject(z)

so at initialization the injected latent contributes nothing and the model
starts exactly as the trained-from-scratch backbone would. Without this, an
untrained LCR/flow latent would inject noise into the bottleneck and the network
would spend early training merely recovering from it.

Shape bookkeeping (dim=48 -> latent 384; input (B,3,H,W), H,W divisible by 8):
    SeeCoder context : (B, 256, H/8, W/8)
    DiffAE spatial   : (B, 256, H/8, W/8)
    LCR z_0          : (B, 384, H/8, W/8)
    backbone latent  : (B, 384, H/8, W/8)
    refined features : (B,  96, H,   W)      (dim*2, see Decoder docstring)
    output           : (B,   3, H,   W)

STATUS: implemented and unit-tested end to end. NOT trained. NO experimental
claim is made about any module improving restoration quality.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone.restormer import Restormer
from models.modules.diffae import DiffAE
from models.modules.lcr import LCR
from models.modules.rectified_flow import RectifiedFlow
from models.modules.sra import SRA
from models.modules.seecoder import SeeCoder


class SRRT(nn.Module):
    """
    Args:
        cfg: full loaded YAML config. Module on/off comes from
            `model.modules.<name>.enabled`, which drives the ablation study.

    Dependency rules (validated at construction):
        * LCR requires BOTH SeeCoder and DiffAE (it fuses their two latents).
        * Rectified Flow requires LCR (it transports the latent LCR produces).
        * SeeCoder, DiffAE and SRA are independently switchable.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()

        self.cfg = cfg
        mods = cfg["model"]["modules"]

        self.use_seecoder = bool(mods["seecoder"]["enabled"])
        self.use_diffae = bool(mods["diffae"]["enabled"])
        self.use_lcr = bool(mods["lcr"]["enabled"])
        self.use_flow = bool(mods["rectified_flow"]["enabled"])
        self.use_sra = bool(mods["sra"]["enabled"])

        if self.use_lcr and not (self.use_seecoder and self.use_diffae):
            raise ValueError(
                "model.modules.lcr.enabled requires BOTH seecoder and diffae to be enabled "
                "(LCR fuses their latents)."
            )
        if self.use_flow and not self.use_lcr:
            raise ValueError(
                "model.modules.rectified_flow.enabled requires lcr to be enabled "
                "(the flow transports the latent produced by LCR)."
            )

        self.backbone = Restormer.from_config(cfg)
        self.latent_dim = self.backbone.latent_dim
        self.refined_dim = self.backbone.dim * 2

        self.seecoder = SeeCoder.from_config(cfg) if self.use_seecoder else None
        self.diffae = DiffAE.from_config(cfg) if self.use_diffae else None

        self.lcr = None
        self.inject = None
        self.gamma_latent = None

        if self.use_lcr:
            self.lcr = LCR.from_config(
                cfg,
                visual_dim=self.seecoder.embed_dim,
                damage_dim=self.diffae.latent_dim,
            )

        # Width of whatever latent reaches the bottleneck. LCR fuses SeeCoder and
        # DiffAE into one code; without it each enabled module contributes its own
        # latent directly, concatenated when both are on. Building the injection
        # for those cases too is what makes the intermediate ablations measure
        # anything -- with the path gated on LCR alone, enabling SeeCoder by
        # itself left it with no route to the output and no gradient.
        if self.use_lcr:
            inject_dim = self.lcr.latent_dim
        else:
            inject_dim = ((self.seecoder.embed_dim if self.use_seecoder else 0)
                          + (self.diffae.latent_dim if self.use_diffae else 0))

        if inject_dim:
            self.inject = nn.Conv2d(inject_dim, self.latent_dim, kernel_size=1, bias=False)
            # Zero-init gate: the injected latent is a no-op at initialization.
            self.gamma_latent = nn.Parameter(torch.zeros(1))

        self.flow = RectifiedFlow.from_config(cfg) if self.use_flow else None
        self.sra = SRA.from_config(cfg, dim=self.refined_dim) if self.use_sra else None

    # ------------------------------------------------------------------
    # latent helpers
    # ------------------------------------------------------------------

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Backbone latent for an image: (B, latent_dim, H/8, W/8).

        Applied to the CLEAN target during training this yields the reference
        latent z_1 used by both the flow loss and the LCR consistency loss.
        """
        *_, x4 = self.backbone.encoder(x)
        return self.backbone.bottleneck(x4)

    def _srrt_latent(self, y: torch.Tensor, grid: tuple[int, int]) -> dict[str, torch.Tensor]:
        """Run SeeCoder / DiffAE / LCR and return the intermediate latents."""
        out: dict[str, torch.Tensor] = {}

        visual = None
        if self.seecoder is not None:
            see = self.seecoder(y)
            visual = see["context"]
            out["visual_features"] = visual

        damage = None
        if self.diffae is not None:
            dif = self.diffae(y)
            damage = dif["spatial"]
            out["damage_features"] = damage
            out["damage_semantic"] = dif["semantic"]

        if self.lcr is not None:
            if visual.shape[-2:] != grid:
                visual = F.interpolate(visual, size=grid, mode="bilinear", align_corners=False)
            if damage.shape[-2:] != grid:
                damage = F.interpolate(damage, size=grid, mode="bilinear", align_corners=False)
            out["latent_initial"] = self.lcr(visual, damage)["z0"]
        else:
            # No LCR to fuse them: send the enabled modules' latents to the
            # bottleneck directly, concatenated on the channel axis when both
            # are present. Same grid, same zero-init gate as the LCR path.
            parts = []
            for t in (visual, damage):
                if t is None:
                    continue
                if t.shape[-2:] != grid:
                    t = F.interpolate(t, size=grid, mode="bilinear", align_corners=False)
                parts.append(t)
            if parts:
                out["latent_direct"] = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        return out

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        y: torch.Tensor,
        target: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, Any]:
        """
        Args:
            y: degraded input (B, 3, H, W).
            target: clean image. Training only -- supplying it enables the flow
                and LCR losses, which need the reference latent z_1. Never pass
                it at inference.
            return_features: also return the heavy intermediate tensors
                (SeeCoder/DiffAE features, SRA maps) for visualization.

        Returns a dict always containing "output"; loss terms ("flow_loss",
        "lcr_loss") appear only when `target` is given and the corresponding
        module is enabled.
        """
        out: dict[str, Any] = {}

        skip1, skip2, skip3, x4 = self.backbone.encoder(y)
        latent = self.backbone.bottleneck(x4)
        grid = latent.shape[-2:]

        if self.use_seecoder or self.use_diffae:
            srrt_latents = self._srrt_latent(y, grid)
            out.update(srrt_latents)

            # LCR's fused code when it is enabled, otherwise the direct latent.
            z0 = srrt_latents.get("latent_initial")
            if z0 is None:
                z0 = srrt_latents.get("latent_direct")

            if z0 is not None:
                z_used = z0

                if self.flow is not None:
                    if target is not None:
                        # Training: match velocity against the true transport
                        # z0 -> z1, where z1 is the clean-image latent.
                        #
                        # DETACHED. z1 is a regression TARGET, not a quantity to
                        # be optimized: it is produced by the shared restoration
                        # encoder, so leaving it in the graph lets flow_loss be
                        # minimized by dragging z1 toward z0 -- i.e. by degrading
                        # the encoder the restoration path depends on. Measured
                        # on the un-detached version, flow_loss sent MORE
                        # gradient into backbone.encoder (4.23e-02) than into
                        # the velocity network it is meant to train (1.97e-02),
                        # and runs E/F collapsed ~10 dB at epoch 6 while
                        # flow_loss fell 31x. `lcr.consistency_loss` already
                        # detaches its reference for the same reason.
                        z1 = self.encode_latent(target).detach()
                        flow_loss, info = self.flow.flow_loss(z0, z1)
                        out["flow_loss"] = flow_loss
                        out["flow_t"] = info["t"]
                        # z1 stays the flow regression target AND the LCR
                        # reference, so `latent_flow` still carries it.
                        out["latent_flow"] = z1
                        # But the INJECTED tensor is z0, not z1. Injecting z1
                        # feeds the backbone an encoding of the ground truth,
                        # which does not exist at inference: the model learns to
                        # lean on it (gamma_latent grew 318x in 10 epochs) and
                        # then meets `integrate(z0)` at test time instead.
                        # Measured on that checkpoint, same weights and images:
                        # 24.819 dB with z1 injected vs 18.946 dB with the ODE
                        # latent -- a 5.874 dB train/inference gap, with train
                        # loss falling while val loss rose.
                        z_used = z0
                    else:
                        # Inference: solve the ODE from z0 to recover z1_hat.
                        z_used = self.flow.integrate(z0).to(z0.dtype)
                        out["latent_flow"] = z_used

                if target is not None and self.lcr is not None:
                    z_ref = out.get("latent_flow")
                    if z_ref is None:
                        z_ref = self.encode_latent(target)
                    out["lcr_loss"] = self.lcr.consistency_loss(z0, z_ref)

                latent = latent + self.gamma_latent * self.inject(z_used)

        out["latent"] = latent

        decoded = self.backbone.decoder(latent, skip1, skip2, skip3)
        refined = self.backbone.refinement(decoded)

        if self.sra is not None:
            refined, maps = self.sra(refined, return_maps=True)
            if return_features:
                out["sra_attention"] = maps["attention"]
                out["sra_residual"] = maps["residual"]

        residual = self.backbone.output(refined)
        out["residual"] = residual
        out["output"] = residual + y

        if not return_features:
            for key in ("visual_features", "damage_features", "damage_semantic"):
                out.pop(key, None)

        return out

    def restore(self, y: torch.Tensor) -> torch.Tensor:
        """Convenience inference call: restored = model.restore(degraded)."""
        return self.forward(y)["output"]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "SRRT":
        return cls(cfg)

    def module_summary(self) -> dict[str, Any]:
        """Which modules are active and how many parameters each contributes."""
        def count(module: nn.Module | None) -> int:
            return sum(p.numel() for p in module.parameters()) if module is not None else 0

        return {
            "seecoder": {"enabled": self.use_seecoder, "params": count(self.seecoder)},
            "diffae": {"enabled": self.use_diffae, "params": count(self.diffae)},
            "lcr": {"enabled": self.use_lcr, "params": count(self.lcr)},
            "rectified_flow": {"enabled": self.use_flow, "params": count(self.flow)},
            "sra": {"enabled": self.use_sra, "params": count(self.sra)},
            "backbone": {"enabled": True, "params": count(self.backbone)},
            "total": sum(p.numel() for p in self.parameters()),
        }
