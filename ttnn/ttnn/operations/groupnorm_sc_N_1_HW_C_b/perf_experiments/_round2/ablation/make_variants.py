#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""(Superseded: bench.py injects the GN_ABL_* defines per variant, `ablation/src@p1_p2`.) Materialize one kernel dir per ablation variant: k_<tag>/ = src/ with `#define GN_ABL_<X> 1` lines prepended.
usage: make_variants.py full p2 p2_p1 ...   (tag = '_'-joined subset of p1,p2,tz,cmb,peer,dm; 'full' = no stub)"""
import pathlib, sys

HERE = pathlib.Path(__file__).parent
for tag in sys.argv[1:]:
    flags = [] if tag == "full" else tag.split("_")
    d = HERE / f"k_{tag}"
    d.mkdir(exist_ok=True)
    head = "".join(f"#define GN_ABL_{f.upper()} 1\n" for f in flags)
    for src in (HERE / "src").glob("*.cpp"):
        txt = src.read_text()
        # keep the SPDX header first; defines right after it
        i = txt.index("\n\n") + 2
        (d / src.name).write_text(txt[:i] + head + txt[i:])
    print(d)
