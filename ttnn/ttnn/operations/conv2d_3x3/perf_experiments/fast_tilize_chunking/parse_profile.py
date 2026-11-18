# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: label the GenericOp rows of a --profile ops_perf_results CSV with the node ids, in run order, and append
them to results_op.jsonl. No ttnn import.

    python3 parse_profile.py <csv> <variant> <run_tag> <id1> <id2> ...
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    path, variant, tag, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    rows = [r for r in csv.DictReader(open(path)) if r["OP CODE"] == "GenericOpDeviceOperation"]
    assert len(rows) == len(ids), (len(rows), len(ids))
    with (HERE / "results_op.jsonl").open("a") as f:
        for r, case in zip(rows, ids):
            ns = float(r["DEVICE KERNEL DURATION [ns]"])
            print(f"{variant:<6} {tag:<4} {case:<24} {ns / 1000:8.2f} us  cores {r['CORE COUNT']}")
            f.write(json.dumps(dict(case=case, variant=variant, tag=tag, ns=ns, csv=path)) + "\n")


if __name__ == "__main__":
    main()
