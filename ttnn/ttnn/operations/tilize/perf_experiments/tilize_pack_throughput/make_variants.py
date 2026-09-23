"""Regenerate the (git-ignored) kernels_<name>/ variant dirs of this experiment.

Each dir = a copy of the op's kernels/ with tilize_compute.cpp replaced by the hand-authored
tilize_compute_pt.cpp and a generated pt_knobs.hpp. usage: python3 make_variants.py
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OP_KERNELS = HERE.parents[1] / "kernels"

# name: (PT_MODE, PT_RECONFIG, PT_UNINIT, PT_SECTION, PT_NOZERO, PT_AB, PT_NOINIT, PT_BPS, PT_GROUP_CB,
#        PT_STD_NO_TO_DEST)
VARIANTS = {
    "ctl": (0, 1, 1, 8, 0),  # A/A: identical to head through the variant source
    "norc": (0, 0, 1, 8, 0),
    "nou": (0, 1, 0, 8, 0),
    "norc_nou": (0, 0, 0, 8, 0),
    "s8": (1, 1, 1, 8, 0),
    "s6": (1, 1, 1, 6, 0),
    "s4": (1, 1, 1, 4, 0),
    "s2": (1, 1, 1, 2, 0),
    "s4nz": (1, 1, 1, 4, 1),
    "s8nz": (1, 1, 1, 8, 1),
    "s4lean": (1, 0, 0, 4, 0),
    "s2lean": (1, 0, 0, 2, 0),
    "std8": (2, 1, 1, 8, 0),
    "std4": (2, 1, 1, 4, 0),
    # ablations (payload stubbed, synchronization kept) -- NOT correct, run with TPT_CHECK=0
    "ab_f_nopack": (1, 1, 1, 8, 0, 2),
    "ab_f_packonly": (1, 1, 1, 8, 0, 1),
    "ab_f_none": (1, 1, 1, 8, 0, 3),
    "ab_s_nopack": (2, 1, 1, 8, 0, 2),
    "ab_s_packonly": (2, 1, 1, 8, 0, 1),
    "ab_f_none_lean": (1, 0, 0, 8, 0, 3),
    "ab_f_none_noinit": (1, 0, 0, 8, 0, 3, 1),
    "ab_s_none": (2, 1, 1, 8, 0, 3),
    # lean = no redundant reconfig (hw_startup already configured these CBs) + no kernel-end uninit
    "s8lean": (1, 0, 0, 8, 0),
    "s6lean": (1, 0, 0, 6, 0),
    "s8nou": (1, 1, 0, 8, 0),
    "s8norc": (1, 0, 1, 8, 0),
    # mode 3: DEST sections spanning tile-row blocks (bps = blocks per section, g = grouped CB ops)
    "x8": (3, 1, 1, 8, 0, 0, 0, 8, 0),
    "x8g": (3, 1, 1, 8, 0, 0, 0, 8, 1),
    "x8lean": (3, 0, 0, 8, 0, 0, 0, 8, 0),
    "x8glean": (3, 0, 0, 8, 0, 0, 0, 8, 1),
    "x2glean": (3, 0, 0, 8, 0, 0, 0, 2, 1),
    # mode 5: helper fast path (NoReconfigure) + DEST-batched slow path wherever the helper
    # would fall back to per-tile tilize_block (32-bit / uint8 / uint16 / fp32 out / full sync)
    "b16": (5, 0, 1, 16, 0),
    "b4": (5, 0, 1, 4, 0),
    "b2": (5, 0, 1, 2, 0),
    "b2nd": (5, 0, 1, 2, 0, 0, 0, 8, 0, 1),
    "b4nd": (5, 0, 1, 4, 0, 0, 0, 8, 0, 1),
}


def main():
    for name, knobs in VARIANTS.items():
        defaults = (0, 1, 1, 8, 0, 0, 0, 8, 0, 0)
        mode, rc, un, sec, nz, ab, noinit, bps, grp, ntd = tuple(knobs) + defaults[len(knobs) :]
        d = HERE / f"kernels_{name}"
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(OP_KERNELS, d)
        shutil.copy(HERE / "tilize_compute_pt.cpp", d / "tilize_compute.cpp")
        (d / "pt_knobs.hpp").write_text(
            "#pragma once\n"
            f"#define PT_MODE {mode}\n#define PT_RECONFIG {rc}\n#define PT_UNINIT {un}\n"
            f"#define PT_SECTION {sec}\n#define PT_NOZERO {nz}\n#define PT_AB {ab}\n#define PT_NOINIT {noinit}\n"
            f"#define PT_BPS {bps}\n#define PT_GROUP_CB {grp}\n#define PT_STD_NO_TO_DEST {ntd}\n"
        )
    # the graduation candidate: the op's kernels/ with graduate/tilize_compute.cpp (hand-authored)
    d = HERE / "kernels_graduate"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(OP_KERNELS, d)
    shutil.copy(HERE / "graduate" / "tilize_compute.cpp", d / "tilize_compute.cpp")
    print("generated:", ", ".join(VARIANTS), "+ graduate")


if __name__ == "__main__":
    main()
