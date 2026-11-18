# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-compare saved variant outputs (host only, torch only).

    python3 -I compare_outputs.py <case> <variant_a> <variant_b> [<variant_a> <variant_b> ...]
"""

import os
import sys
from pathlib import Path


OUT_DIR = Path(os.environ.get("TAB_OUT_DIR", "/tmp/tap_aligned_blocking_outputs"))


def main():
    import torch

    case, pairs = sys.argv[1], sys.argv[2:]
    for a_name, b_name in zip(pairs[::2], pairs[1::2]):
        a = torch.load(OUT_DIR / f"{case}__{a_name}.pt").float()
        b = torch.load(OUT_DIR / f"{case}__{b_name}.pt").float()
        diff = (a - b).abs()
        n = diff.numel()
        mism = int((diff > 0).sum())
        print(
            f"{case}: {a_name} vs {b_name}: bit-identical={mism == 0}  mismatches {mism}/{n} ({100 * mism / n:.4f}%)  "
            f"max abs diff {float(diff.max()):.6g}  (output max |x| {float(a.abs().max()):.4g})"
        )


if __name__ == "__main__":
    main()
