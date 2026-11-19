# usage: compare_unit_dumps.py : compare /tmp/wr_unit_dumps/<case>_<variant>.pt (test_wr.py::test_wr_unit_shapes,
# WR_UNIT_SHAPES=1) across variants: differing output elements against `base`.
import collections
import glob
import os


def main():
    import torch

    by_case = collections.defaultdict(dict)
    for f in sorted(glob.glob("/tmp/wr_unit_dumps/*.pt")):
        stem = os.path.basename(f)[:-3]
        if stem.endswith("_cand_s1"):
            case, variant = stem[: -len("_cand_s1")], "cand_s1"
        else:
            case, variant = stem.rsplit("_", 1)
        by_case[case][variant] = torch.load(f)["out"]
    for case, vs in by_case.items():
        b = vs.get("base")
        cells = [f"{v}: {int((out != b).sum())} diff" for v, out in vs.items() if v != "base" and b is not None]
        print(f"{case:24} {', '.join(cells)}")


if __name__ == "__main__":
    main()
