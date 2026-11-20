# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-identity: every output hash of BRF_BANK_RUN=1 equals the BRF_BANK_RUN=0 one (runs/hashes.jsonl), per shape."""
import collections
import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "runs" / "hashes.jsonl")
by = collections.defaultdict(lambda: collections.defaultdict(set))
for line in open(path):
    r = json.loads(line)
    by[(tuple(r["shape"]), tuple(r["args"]))][r["bank_run"]].add(r["sha1"])
ok = bad = 0
for key, v in by.items():
    same = bool(v.get("0")) and v.get("0") == v.get("1")  # equal hash sets (two cases may share an output shape)
    ok += bool(same)
    bad += not same
    print("IDENTICAL" if same else "DIFF/INCOMPLETE", key, {k: sorted(s) for k, s in v.items()})
print(f"{ok} identical, {bad} not")
