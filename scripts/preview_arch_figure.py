"""
Render a PREVIEW of the paper's Figure 1 (the architecture diagram).

IMPORTANT: this is NOT the figure the paper uses. Figure 1 is drawn by TikZ
code inside paper/srrt_ieee.tex (look for \begin{tikzpicture}), so it has no
image file and cannot be seen without compiling the paper.

This script replicates that TikZ picture's EXACT coordinates in matplotlib so
the layout can be checked -- box positions, connector routing, label collisions
-- on a machine with no LaTeX toolchain. If you move a box in the .tex, move it
here too or this preview stops telling the truth.

Output: paper/figures/fig_arch_preview.png (not referenced by the paper, and
not included by scripts/bundle_paper.py).

    python scripts/preview_arch_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, Circle

fig, ax = plt.subplots(figsize=(7.6, 4.3))

def panel(x0, y0, x1, y1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1-x0, y1-y0,
        boxstyle="round,pad=0,rounding_size=0.1",
        fc="#f2f2f2", ec="#d8d8d8", lw=0.8, zorder=0))

def box(x0, y0, x1, y1, fc, ec):
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fc=fc, ec=ec, lw=0.9, zorder=2))

def arrow(pts, style="-", color="#4d4d4d", dashed=False):
    for i in range(len(pts)-1):
        ax.annotate("", xy=pts[i+1], xytext=pts[i],
            arrowprops=dict(arrowstyle="->" if i == len(pts)-2 else "-",
                            color=color, lw=0.9,
                            linestyle=(0, (1.5, 1.5)) if dashed else "solid",
                            shrinkA=0, shrinkB=0), zorder=3)

SEM, SEME = "#fde3c4", "#e8923a"
BB,  BBE  = "#cfdcf5", "#4a74c9"
IO,  IOE  = "#e8e8e8", "#8f8f8f"

panel(0.62, 2.06, 5.34, 3.84)
panel(0.62, 0.02, 7.02, 1.20)
ax.text(0.72, 3.70, "I. semantic pathway (proposed)", fontsize=6, color="#8a8a8a", va="center")
ax.text(0.72, 1.07, "II. Restormer backbone (unmodified)", fontsize=6, color="#8a8a8a", va="center")

box(0.06, 0.12, 0.50, 0.88, IO, IOE); ax.text(0.28, 0.50, "$y$", ha="center", va="center", fontsize=8, zorder=4)
ax.add_patch(Polygon([(0.80,0.14),(0.80,0.86),(2.00,0.70),(2.00,0.30)], fc=BB, ec=BBE, lw=0.9, zorder=2))
ax.text(1.40, 0.50, "encoder", ha="center", va="center", fontsize=7, zorder=4)
box(2.55, 0.12, 3.95, 0.88, BB, BBE); ax.text(3.25, 0.50, "bottleneck\n$\\ell\\to\\ell'$", ha="center", va="center", fontsize=7, zorder=4)
ax.add_patch(Polygon([(4.50,0.30),(4.50,0.70),(5.70,0.86),(5.70,0.14)], fc=BB, ec=BBE, lw=0.9, zorder=2))
ax.text(5.10, 0.50, "decoder", ha="center", va="center", fontsize=7, zorder=4)
box(6.00, 0.12, 6.88, 0.88, SEM, SEME); ax.text(6.44, 0.50, "SRA", ha="center", va="center", fontsize=7, zorder=4)
box(7.08, 0.12, 7.52, 0.88, IO, IOE); ax.text(7.30, 0.50, "$\\hat{x}$", ha="center", va="center", fontsize=8, zorder=4)

for a, b in [(0.50,0.80),(2.00,2.55),(3.95,4.50),(5.70,6.00),(6.88,7.08)]:
    arrow([(a,0.50),(b,0.50)])

box(0.80, 3.07, 2.04, 3.53, SEM, SEME); ax.text(1.42, 3.30, "SeeCoder", ha="center", va="center", fontsize=7, zorder=4)
box(0.80, 2.29, 2.04, 2.75, SEM, SEME); ax.text(1.42, 2.52, "DiffAE", ha="center", va="center", fontsize=7, zorder=4)
box(2.38, 2.68, 3.46, 3.14, SEM, SEME); ax.text(2.92, 2.91, "LCR", ha="center", va="center", fontsize=7, zorder=4)
box(3.80, 2.68, 5.12, 3.14, SEM, SEME); ax.text(4.46, 2.91, "rectified flow", ha="center", va="center", fontsize=7, zorder=4)

arrow([(0.28,0.88),(0.28,3.30),(0.80,3.30)])
arrow([(0.28,2.52),(0.80,2.52)])
ax.plot(0.28, 2.52, "o", ms=2, color="#4d4d4d", zorder=4)
arrow([(2.04,3.30),(2.21,3.30),(2.21,2.91)])
arrow([(2.04,2.52),(2.21,2.52),(2.21,2.91)])
ax.plot(2.21, 2.91, "o", ms=2, color="#4d4d4d", zorder=4)
arrow([(2.21,2.91),(2.38,2.91)])
arrow([(3.46,2.91),(3.80,2.91)])
ax.text(3.63, 3.06, "$z_0$", ha="center", fontsize=6, zorder=4)

arrow([(4.46,2.68),(4.46,1.95),(3.25,1.95),(3.25,1.79)])
ax.text(4.56, 2.32, "$\\hat{z}_1$", ha="left", va="center", fontsize=6, zorder=4)
ax.add_patch(Circle((3.25,1.62), 0.17, fc="#f8ccc9", ec="#d0453c", lw=0.9, zorder=3))
ax.text(3.25, 1.62, "$\\gamma$", ha="center", va="center", fontsize=7, zorder=4)
ax.text(3.50, 1.62, "$\\gamma=0$ at init", ha="left", va="center", fontsize=6, color="#a33", zorder=4)
arrow([(3.25,1.45),(3.25,0.88)])

arrow([(0.28,0.12),(0.28,-0.30),(7.30,-0.30),(7.30,0.12)], dashed=True, color="#999")
ax.text(3.80, -0.26, "global residual  $\\hat{x}=f(y)+y$", ha="center", va="bottom", fontsize=6, color="#8a8a8a")

ax.set_xlim(-0.15, 7.75); ax.set_ylim(-0.55, 4.00)
ax.set_aspect("equal"); ax.axis("off")
import os
os.makedirs(os.path.join("paper", "figures"), exist_ok=True)
out = os.path.join("paper", "figures", "fig_arch_preview.png")
fig.savefig(out, dpi=220, bbox_inches="tight")
print("wrote", out)
print("NOTE: preview only -- the paper draws Fig. 1 from TikZ, not from this file.")
