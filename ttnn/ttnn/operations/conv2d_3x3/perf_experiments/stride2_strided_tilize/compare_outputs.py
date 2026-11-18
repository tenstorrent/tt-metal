# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-compare saved variant outputs against the baseline with torch.equal (host only, torch only).

    python3 -I compare_outputs.py <variant> [case ...]      (default: every case with both files)
"""

import os
import sys
from pathlib import Path

OUT_DIR = Path(os.environ.get("S2W_OUT_DIR", "/tmp/s2w_outputs"))


def main():
    import torch

    variant = sys.argv[1]
    cases = sys.argv[2:] or sorted({p.name.split("__")[0] for p in OUT_DIR.glob("*__v0.pt")})
    for case in cases:
        a_path, b_path = OUT_DIR / f"{case}__v0.pt", OUT_DIR / f"{case}__v{variant}.pt"
        if not (a_path.exists() and b_path.exists()):
            print(f"{case}: missing {'baseline' if not a_path.exists() else 'variant'} output")
            continue
        a, b = torch.load(a_path), torch.load(b_path)
        diff = (a.float() - b.float()).abs().nan_to_num(0)
        print(
            f"{case:<26} v0 vs v{variant}: torch.equal={torch.equal(a, b)}  mismatches {int((diff > 0).sum())}/"
            f"{diff.numel()}  max abs diff {float(diff.max()):.6g}"
        )


if __name__ == "__main__":
    main()
