"""
Bundle the paper and every figure it references into one Overleaf-ready zip.

Exists because uploading srrt_ieee.tex on its own produces a PDF with no
images -- LaTeX reports a missing-file warning and keeps going, so the build
"succeeds" and the figures are simply absent.

Each figure is written TWICE inside the zip: flat at the root and under
figures/. The paper's \\graphicspath searches both, so the build works whether
Overleaf flattens the upload or preserves the folder.

    python scripts/bundle_paper.py
"""

import os
import re
import sys
import zipfile

TEX = os.path.join("paper", "srrt_ieee.tex")
FIGDIR = os.path.join("paper", "figures")
OUT = os.path.join("paper", "srrt_ieee_overleaf.zip")

INCLUDE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def main() -> int:
    if not os.path.exists(TEX):
        print("no %s" % TEX)
        return 1

    src = open(TEX, encoding="utf-8").read()
    figs = sorted(set(INCLUDE.findall(src)))

    missing = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        # Named main.tex, not srrt_ieee.tex: Overleaf selects main.tex as the
        # main document automatically. Uploaded under any other name it keeps
        # compiling its own default main.tex and the paper never builds.
        z.write(TEX, "main.tex")
        for g in figs:
            p = os.path.join(FIGDIR, g)
            if os.path.exists(p):
                z.write(p, g)
                z.write(p, "figures/" + g)
            else:
                missing.append(g)

    print("wrote %s (%.1f KB)" % (OUT, os.path.getsize(OUT) / 1024.0))
    print("figures referenced: %d" % len(figs))
    for g in figs:
        print("   %-28s %s" % (g, "MISSING" if g in missing else "bundled"))

    if missing:
        print()
        print("INCOMPLETE: %d figure(s) missing. Run:" % len(missing))
        print("   python scripts/make_paper_figures.py")
        return 1

    print()
    print("Upload this zip to Overleaf (New Project -> Upload Project).")
    print("The .tex is named main.tex so Overleaf selects it automatically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
