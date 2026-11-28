# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""PL_ITEMS="CASE|mode;CASE|mode;..."  mode = placement mode (placement.py) or "stock" (ttnn.typecast black box).
Each item: the golden suite's own runner + check on launch 1, then 11 more launches on the same input.
Item order -> PL_ORDER file so parse.py maps CSV rows (12 per item) back to items."""
import os
import random
import sys
import zlib

import pytest
import importlib as _importlib

torch = _importlib.import_module("torch")  # experiment bench, not ttnn package code
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
import placement  # noqa: E402
from eval.golden_tests.g6_typecast import feature_spec as SPEC  # noqa: E402
from eval.golden_tests.g6_typecast import helpers as H  # noqa: E402

LAUNCHES = 12


def test_items(device, monkeypatch):
    placement.install()
    items = [i.split("|") for i in os.environ["PL_ITEMS"].split(";") if i]
    order = open(os.environ.get("PL_ORDER", "/tmp/pl_order.txt"), "w")
    real = H.g6_typecast
    for case, mode in items:
        seed = zlib.crc32(case.encode()) & 0x7FFFFFFF
        torch.manual_seed(seed)
        random.seed(seed)
        scen = SPEC._BY_CASE[case][0]
        op = (lambda t, d=None, **kw: ttnn.typecast(t, kw.get("dtype", d))) if mode == "stock" else real
        if mode != "stock":
            placement.set_mode(mode)

        def repeated(*args, **kwargs):
            first = op(*args, **kwargs)
            for _ in range(LAUNCHES - 1):
                ttnn.synchronize_device(device)
                ttnn.deallocate(op(*args, **kwargs))
            ttnn.synchronize_device(device)
            return first

        monkeypatch.setattr(H, "g6_typecast", repeated)
        try:
            H.run_g6_typecast(
                (scen,),
                dtype=scen["dtype"],
                output_dtype=scen["output_dtype"],
                layout=scen["layout"],
                buffer=scen["buffer"],
                device=device,
            )
            ok = True
        except AssertionError as e:
            ok = False
            print(f"PL_CHECK_FAIL {case} {mode}: {str(e)[:200]}", flush=True)
        placed = placement.LAST.get("placed") if mode not in ("stock", "default") else None
        print(f"PL_CHECK {case} {mode} ok={ok} placed={placed}", flush=True)
        placement.LAST.clear()
        order.write(f"{case} {mode}\n")
        order.flush()
        assert ok or mode == "stock", f"{case} {mode} failed the suite's check"
