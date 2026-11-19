# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact comparison of test_sd_unit_shapes dumps: every variant against `baseline` of the same case."""
import glob, sys


def main():
    import torch

    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sd_unit_dumps"
    for f in sorted(glob.glob(f"{d}/*_baseline.pt")):
        base = torch.load(f)
        for v in ("sd", "sd_nofuse"):
            g = f.replace("_baseline.pt", f"_{v}.pt")
            try:
                o = torch.load(g)
            except FileNotFoundError:
                continue
            diff = (o["out"] - base["out"]).abs()
            rel = diff / base["out"].abs().clamp_min(1e-3)
            print(
                f"{f.split('/')[-1][:-12]:<16} {v:<9} identical={bool(torch.equal(o['out'], base['out']))!s:<5} "
                f"max|d| {diff.max():.5f} n_diff {int((diff > 0).sum()):>6}/{diff.numel()} max_rel {rel.max():.5f} | "
                f"max|err| vs ref: base {(base['out'] - base['ref']).abs().max():.4f} {v} {(o['out'] - o['ref']).abs().max():.4f}"
            )


if __name__ == "__main__":
    main()
