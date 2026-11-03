"""compare.py <prefix_a> <prefix_b> case... -> bit-exactness of saved outputs on the real rows (padding rows are
never written by the op). Run from the repo root with python_env active."""
import math
import sys

sys.path.insert(0, ".")
from tests.ttnn.unit_tests.operations.moe_down_proj.probes.test_perf_bench import CASES


def main():
    import torch

    CASES["t200_rect"] = (200, (90, 10, 60, 70, 5, 55, 40, 70))
    a, b, cases = sys.argv[1], sys.argv[2], sys.argv[3:]
    for case in cases:
        x, y = torch.load(f"{a}_{case}.pt"), torch.load(f"{b}_{case}.pt")
        rows = x.shape[-2]
        real = torch.zeros(rows, dtype=torch.bool)
        start = 0
        for count in CASES[case][1]:
            real[start : start + count] = True
            start += 32 * math.ceil(count / 32)
        xr, yr = x.reshape(rows, -1)[real], y.reshape(rows, -1)[real]
        same = torch.equal(xr, yr)
        print(case, "equal" if same else f"DIFF max_abs={(xr.float() - yr.float()).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
