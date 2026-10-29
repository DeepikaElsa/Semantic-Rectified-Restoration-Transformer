"""
Structural validation for paper/srrt_ieee.tex.

No LaTeX toolchain is installed in this environment, so the paper cannot be
compiled here. This checks everything that can be checked without compiling --
balanced environments and braces, resolvable \\ref, consistent \\cite/\\bibitem,
present figure files -- and estimates the page count.

The page estimate is an ESTIMATE. Only a real pdflatex run settles the 6-page
limit; compile on Overleaf to confirm.

    python scripts/check_paper.py
"""

import os
import re
import sys

TEX = os.path.join("paper", "srrt_ieee.tex")
FIGDIR = os.path.join("paper", "figures")

ok = True


def fail(msg: str) -> None:
    global ok
    ok = False
    print("  FAIL:", msg)


def main() -> int:
    src = open(TEX, encoding="utf-8").read()
    body = src.split("\\begin{document}", 1)[1]

    # --- environments -----------------------------------------------------
    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", src):
        if kind == "begin":
            stack.append(name)
        elif not stack:
            fail(r"\end{%s} with no matching \begin" % name)
        elif stack[-1] != name:
            fail(r"\end{%s} closes \begin{%s}" % (name, stack[-1]))
        else:
            stack.pop()
    if stack:
        fail("unclosed environments: %s" % stack)
    print("environments      :", "balanced" if not stack else "UNBALANCED")

    # --- braces (strip comments and escaped braces first) -----------------
    t = re.sub(r"(?<!\\)%.*", "", src)
    t = t.replace(r"\{", "").replace(r"\}", "")
    delta = t.count("{") - t.count("}")
    if delta:
        fail("brace imbalance %+d" % delta)
    print("braces            :", "balanced" if delta == 0 else "%+d" % delta)

    # --- refs vs labels ---------------------------------------------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", src))
    # \eqref counts: an equation referred to only by \eqref is referenced.
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", src))
    missing = refs - labels
    if missing:
        fail("undefined \\ref: %s" % sorted(missing))
    print("refs              : %d used, %d labels, %s"
          % (len(refs), len(labels), "all resolve" if not missing else "MISSING"))
    unused = labels - refs
    if unused:
        print("   unreferenced labels (harmless):", sorted(unused))

    # --- citations --------------------------------------------------------
    bib = set(re.findall(r"\\bibitem\{([^}]+)\}", src))
    cited = set()
    for group in re.findall(r"\\cite\{([^}]+)\}", src):
        cited |= {x.strip() for x in group.split(",")}
    if cited - bib:
        fail("cited with no bibitem: %s" % sorted(cited - bib))
    if bib - cited:
        fail("bibitem never cited: %s" % sorted(bib - cited))
    print("citations         : %d cited, %d bibitems, %s"
          % (len(cited), len(bib), "consistent" if cited == bib else "MISMATCH"))

    # --- graphics ---------------------------------------------------------
    for g in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
        p = os.path.join(FIGDIR, g)
        present = os.path.exists(p)
        print("graphic %-26s: %s" % (g, "OK" if present else "MISSING"))
        if not present:
            fail("missing figure file %s" % p)

    # --- length estimate --------------------------------------------------
    # Split the bibliography out: it sets at \footnotesize and packs far
    # denser than body prose, so counting it at body rate understates it.
    main, _, bibpart = body.partition(r"\begin{thebibliography}")
    prose = re.sub(r"\\[a-zA-Z]+\*?", " ", main)
    words = len(re.findall(r"\b[A-Za-z][A-Za-z\-]+\b", prose))
    nbib = len(re.findall(r"\\bibitem", bibpart))

    fig_all = len(re.findall(r"\\begin\{figure", src))
    fig_star = len(re.findall(r"\\begin\{figure\*\}", src))
    tables = len(re.findall(r"\\begin\{table", src))
    # Display equations cost vertical space that word counts do not see.
    eqs = (len(re.findall(r"\\begin\{equation", src))
           + len(re.findall(r"\\begin\{align", src)))

    # IEEEtran conference, 10pt, two columns: ~575 words per column,
    # ~1150 per page of solid prose. Measured against compiled drafts.
    text_pages = words / 1150.0
    float_pages = fig_star * 0.42 + (fig_all - fig_star) * 0.20 + tables * 0.17
    eq_pages = eqs * 0.022
    bib_pages = nbib * 0.030
    head_pages = 0.34                      # title block + abstract furniture
    total = text_pages + float_pages + eq_pages + bib_pages + head_pages

    print()
    print("body words        : ~%d (excl. bibliography)" % words)
    print("floats            : %d figures (%d full-width), %d tables, %d equations"
          % (fig_all, fig_star, tables, eqs))
    print("estimated pages   : ~%.1f  (text %.1f + floats %.1f + eqs %.1f"
          " + bib %.1f + head %.1f)"
          % (total, text_pages, float_pages, eq_pages, bib_pages, head_pages))
    if total > 6.3:
        print("  -> likely OVER 6 pages; trim before submission")
    elif total > 5.7:
        print("  -> close to the limit; verify by compiling")

    print()
    print("STRUCTURAL CHECK:", "PASS" if ok else "FAIL")
    print("NOTE: not compiled -- no LaTeX toolchain here. Confirm on Overleaf.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
