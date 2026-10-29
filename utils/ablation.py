"""
Ablation configuration presets (spec section 19 / Phase 8).

Each preset maps an experiment name to the set of modules enabled. Presets are
applied on top of configs/srrt_base.yaml with `apply_ablation`, so a single base
config drives every run and the experiments differ ONLY by these flags.

    A baseline_restormer          Restormer only
    B seecoder                    + SeeCoder
    C seecoder_diffae             + DiffAE
    D seecoder_diffae_lcr         + LCR                (the novelty under test)
    E seecoder_diffae_lcr_flow    + Rectified Flow
    F full_srrt                   + SRA
"""

import copy
from typing import Any

ABLATIONS: dict[str, dict[str, bool]] = {
    "baseline_restormer": {
        "seecoder": False, "diffae": False, "lcr": False,
        "rectified_flow": False, "sra": False,
    },
    "seecoder": {
        "seecoder": True, "diffae": False, "lcr": False,
        "rectified_flow": False, "sra": False,
    },
    "seecoder_diffae": {
        "seecoder": True, "diffae": True, "lcr": False,
        "rectified_flow": False, "sra": False,
    },
    "seecoder_diffae_lcr": {
        "seecoder": True, "diffae": True, "lcr": True,
        "rectified_flow": False, "sra": False,
    },
    "seecoder_diffae_lcr_flow": {
        "seecoder": True, "diffae": True, "lcr": True,
        "rectified_flow": True, "sra": False,
    },
    "full_srrt": {
        "seecoder": True, "diffae": True, "lcr": True,
        "rectified_flow": True, "sra": True,
    },
}


def apply_ablation(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a deep copy of cfg with the named ablation preset applied."""
    if name not in ABLATIONS:
        raise KeyError("Unknown ablation " + repr(name) + ". Available: " + str(sorted(ABLATIONS)))

    out = copy.deepcopy(cfg)
    for module, enabled in ABLATIONS[name].items():
        out["model"]["modules"][module]["enabled"] = enabled

    out["experiment"]["name"] = name
    return out
