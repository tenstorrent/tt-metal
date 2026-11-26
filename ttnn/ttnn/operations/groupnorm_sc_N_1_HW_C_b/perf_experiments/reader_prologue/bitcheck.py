#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit identity across variants per case, from the output hashes bench.py records.
usage: bitcheck.py results_<tag>.json [more.json ...]"""
import json
import sys
from collections import defaultdict

by = defaultdict(lambda: defaultdict(set))
for p in sys.argv[1:]:
    for r in json.load(open(p)):
        if "hash" in r:
            by[r["case"]][r["hash"]].add(r["variant"])
bad = 0
for case, hs in by.items():
    ok = len(hs) == 1
    bad += not ok
    print(
        f"{case:30s} {'BIT-IDENTICAL' if ok else 'DIFFERS'}  " + "; ".join(f"{h}: {sorted(v)}" for h, v in hs.items())
    )
sys.exit(1 if bad else 0)
