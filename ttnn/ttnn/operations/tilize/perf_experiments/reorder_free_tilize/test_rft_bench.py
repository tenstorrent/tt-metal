# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""reorder_free_tilize bench: the op's descriptor copy (pd_rft.py) + kernels from this dir.
RFT=1 -> candidate (no reorder copies; unpacker de-permutes), RFT=0 -> baseline (= the op today).
RFT_KNOBS='{json}' overrides descriptor knobs; RFT_ITERS profiled samples. Gate: torch.equal."""

import json
import os

import pytest

torch = pytest.importorskip("torch")

import ttnn
import importlib

tilize_mod = importlib.import_module("ttnn.operations.tilize.tilize")
from ttnn.operations.tilize.perf_experiments.reorder_free_tilize import pd_rft

SHAPES = [
    (1, 1, 16384, 64),
    (1, 1, 16384, 32),
    (1, 1, 32768, 64),
    (1, 1, 3520, 64),
    (1, 1, 16384, 128),
    (1, 1, 8192, 256),
    (1, 1, 4096, 160),
    (1, 1, 8192, 96),
    (4, 1, 2048, 64),
    (1, 1, 2048, 2048),
]


@pytest.mark.parametrize("out_dtype", ["bf16", "bfp8"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rft_bench(device, monkeypatch, shape, out_dtype):
    monkeypatch.setattr(pd_rft, "REORDER_FREE", os.environ.get("RFT", "1") == "1")
    for name, value in json.loads(os.environ.get("RFT_KNOBS", "{}")).items():
        assert hasattr(pd_rft, name), name
        monkeypatch.setattr(pd_rft, name, value)
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    kw = {} if out_dtype == "bf16" else {"dtype": ttnn.bfloat8_b}
    # Reference: the real op (unpatched descriptor + kernels); bf16 must also equal the input.
    ref = ttnn.to_torch(tilize_mod.tilize(tt_in, **kw))
    if out_dtype == "bf16":
        assert torch.equal(ref, x)
    monkeypatch.setattr(tilize_mod, "create_program_descriptor", pd_rft.create_program_descriptor)
    check = not pd_rft.ABLATE
    for _ in range(int(os.environ.get("RFT_ITERS", "1"))):
        out = tilize_mod.tilize(tt_in, **kw)
        if check:
            got = ttnn.to_torch(out)
            assert torch.equal(got, ref), f"mismatch: {(got != ref).sum().item()} elems"
        out.deallocate()
    print(f"RFT_ACTIVE={pd_rft.LAST_RFT}")
