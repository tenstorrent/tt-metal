# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""SS_ITEMS="CASE|mode;CASE|mode;..."  mode = placement mode (placement.py) or "stock" (ttnn.typecast black box).
Each item: the golden suite's own runner + check on launch 1, then 11 more launches on the same input.
Item order -> SS_ORDER file so parse.py maps CSV rows (12 per item) back to items."""
import os
import random
import sys
import zlib

import pytest
import importlib as _importlib

torch = _importlib.import_module("torch")  # experiment bench, not ttnn package code
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
import subsplit as placement  # noqa: E402
from eval.golden_tests.g6_typecast import feature_spec as SPEC  # noqa: E402
from eval.golden_tests.g6_typecast import helpers as H  # noqa: E402

LAUNCHES = 12


def test_items(device, monkeypatch):
    placement.install()
    items = [i.split("|") for i in os.environ["SS_ITEMS"].split(";") if i]
    order = open(os.environ.get("SS_ORDER", "/tmp/pl_order.txt"), "w")
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
            print(f"SS_CHECK_FAIL {case} {mode}: {str(e)[:200]}", flush=True)
        placed = placement.LAST.get("placed") if mode not in ("stock", "op") else None
        print(f"SS_CHECK {case} {mode} ok={ok} placed={placed}", flush=True)
        placement.LAST.clear()
        order.write(f"{case} {mode}\n")
        order.flush()
        assert ok or mode == "stock", f"{case} {mode} failed the suite's check"


@pytest.mark.parametrize("mode", ["half", "quarter"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 128), (1, 1, 64, 256), (1, 1, 31, 64)])
def test_exact_int32_bf16(device, mode, shape):
    """Bit-exactness of the split int32 -> bf16 element: double-rounding witnesses at many positions of
    every face (rows 0-1 of each face included), plus full-range random values, 4 seeds."""
    placement.install()
    placement.set_mode(mode)
    wit = torch.tensor(
        [
            33685505,
            -33685505,
            2**24 + 2**16 + 1,
            -(2**31),
            2**31 - 1,
            0,
            1,
            -1,
            257,
            2**24 + 1,
            2**24 + 3,
            2**25 + 2**17 + 2,
            -(2**24 + 2**16 + 1),
            2**31 - 2**7,
            16777217,
            2**30 + 2**22,
        ],
        dtype=torch.int64,
    )
    bad = 0
    for seed in range(4):
        torch.manual_seed(seed)
        n = 1
        for d in shape:
            n *= d
        x = torch.randint(-(2**31), 2**31, (n,), dtype=torch.int64)
        idx = torch.arange(0, n, 3 + seed)
        x[idx] = wit[torch.arange(len(idx)) % len(wit)]
        x = x.to(torch.int32).reshape(shape)
        tin = ttnn.from_torch(
            x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.to_torch(H.g6_typecast(tin, ttnn.bfloat16))
        ref = x.to(torch.float32).to(torch.bfloat16)
        diff = (out.view(torch.int16) != ref.view(torch.int16)).sum().item()
        bad += diff
        print(f"SS_EXACT {mode} {shape} seed={seed} mismatches={diff} nf={placement.LAST.get('nf')}", flush=True)
    assert bad == 0
