# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""PC_ITEMS="case|variant;case|variant;..." -- each item: exact check on launch 1, then 11 more launches.
Order written to PC_ORDER (file) so parse.py can map CSV rows (12 per item) back to items.
case = NAME or SHAPE:layout:buf:dir  (e.g. 3x77x33:rm:dram:u2i).  variant = op | stock | one:M[:noc[:minchunk]] | dual:M | two:M
"""
import os
import sys

import pytest
import importlib as _importlib

torch = _importlib.import_module("torch")  # experiment bench, not ttnn package code
import ttnn

sys.path.insert(0, os.path.dirname(__file__))
from pure_copy import pure_copy  # noqa: E402

LAUNCHES = 12
NAMED = {
    "Q2220": "1x4096:rm:dram:u2i",
    "Q2189": "1x3844:rm:dram:u2i",
    "Q2226": "1x1024:rm:dram:u2i",
    "Q2211": "1x1000:rm:dram:u2i",
    "Q2205": "1x256:rm:dram:u2i",
    "Q2214": "1x129:rm:dram:u2i",
    "Q2234": "1x128:rm:dram:u2i",
    "Q2223": "1x100:rm:dram:u2i",
    "Q2208": "1x77:rm:dram:u2i",
    "Q2196": "1x64:rm:dram:u2i",
    "Q2202": "1x33:rm:dram:u2i",
    "Q2199": "1x31:rm:dram:u2i",
    "F321": "1:rm:dram:i2u",
    "Q2230": "1x1x32x128:tile:dram:u2i",
}


def _make(device, spec):
    shape, lay, buf, d = spec.split(":")
    shape = [int(s) for s in shape.split("x")]
    torch.manual_seed(1)
    bits = torch.randint(-(2**31), 2**31 - 1, shape, dtype=torch.int32)
    if d == "u2i":
        bits = bits.abs()  # uint32 host path: keep values representable; device bits still full 31b
        src_dt, dst_dt = ttnn.uint32, ttnn.int32
    else:
        src_dt, dst_dt = ttnn.int32, ttnn.uint32
    mem = ttnn.DRAM_MEMORY_CONFIG if buf == "dram" else ttnn.L1_MEMORY_CONFIG
    layout = ttnn.TILE_LAYOUT if lay == "tile" else ttnn.ROW_MAJOR_LAYOUT
    t = ttnn.from_torch(bits, dtype=src_dt, layout=layout, device=device, memory_config=mem)
    return bits, t, dst_dt


def _run(t, dst_dt, variant):
    if variant == "op":
        from ttnn.operations.g6_typecast import g6_typecast

        return g6_typecast(t, dst_dt)
    if variant == "stock":
        return ttnn.typecast(t, dst_dt)
    if variant in ("auto", "auto1"):
        return pure_copy(t, dst_dt, variant=variant)
    parts = variant.split(":")
    kw = dict(variant=parts[0], movers=int(parts[1]))
    if len(parts) > 2:
        kw["noc"] = int(parts[2])
    if len(parts) > 3:
        kw["min_chunk"] = int(parts[3])
    if len(parts) > 4:
        kw["first_core"] = int(parts[4])
    if len(parts) > 5:
        kw["buf_cap"] = int(parts[5])
    return pure_copy(t, dst_dt, **kw)


def test_items(device):
    items = [i.split("|") for i in os.environ["PC_ITEMS"].split(";") if i]
    order = open(os.environ.get("PC_ORDER", "/tmp/pc_order.txt"), "w")
    for case, variant in items:
        spec = NAMED.get(case, case)
        bits, t, dst_dt = _make(device, spec)
        first = _run(t, dst_dt, variant)
        got = ttnn.to_torch(first).to(torch.int64) & 0xFFFFFFFF
        ok = torch.equal(got.reshape(bits.shape), bits.to(torch.int64) & 0xFFFFFFFF)
        print(f"PC_CHECK {case} {variant} exact={ok}", flush=True)
        assert ok, f"{case} {variant} not bit-identical"
        assert first.dtype == dst_dt
        for _ in range(LAUNCHES - 1):
            r = _run(t, dst_dt, variant)
            ttnn.deallocate(r)
        ttnn.synchronize_device(device)
        ttnn.deallocate(first)
        ttnn.deallocate(t)
        order.write(f"{case} {variant}\n")
        order.flush()
    order.close()
