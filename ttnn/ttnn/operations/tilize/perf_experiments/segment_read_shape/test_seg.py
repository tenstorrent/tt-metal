# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""segment_read_shape bench: the experiment copy of tilize (this dir).

SEG_KNOBS='{"MIN_SEG_BYTES": 512}' overrides descriptor knobs; SEG_ITERS=N runs N profiled samples.
Correctness gate: torch.equal vs the (padded) input.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

import ttnn
from ttnn.operations.tilize.perf_experiments.segment_read_shape import tilize_program_descriptor as pd
from ttnn.operations.tilize.perf_experiments.segment_read_shape.tilize import tilize

SHAPES = [
    (1, 1, 16384, 64),
    (1, 1, 128, 64),
    (1, 1, 32, 2048),
    (1, 1, 2048, 2048),
    (1, 1, 32, 8192),
    (1, 1, 32, 1024),
    (1, 1, 64, 4096),
    (1, 1, 32, 512),
    (1, 1, 256, 256),
    (1, 1, 32, 32768),
    (1, 1, 1024, 4096),
    (1, 1, 512, 1024),
    (1, 1, 96, 1536),
]


def _knobs(monkeypatch):
    for name, value in json.loads(os.environ.get("SEG_KNOBS", "{}")).items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s[2:])))
def test_seg(device, monkeypatch, shape):
    _knobs(monkeypatch)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(int(os.environ.get("SEG_ITERS", "1")) - 1):
        tilize(tt_in).deallocate()
    out = tilize(tt_in)
    assert torch.equal(ttnn.to_torch(out), x)


# Correctness-only: padded / non-aligned / other dtypes on the segment path.
CORR = [
    ((1, 1, 32, 2048), ttnn.bfloat16, None),
    ((1, 1, 20, 2000), ttnn.bfloat16, (1, 1, 32, 2016)),
    ((1, 1, 32, 1000), ttnn.bfloat16, (1, 1, 32, 1024)),
    ((2, 3, 50, 300), ttnn.bfloat16, (2, 3, 64, 320)),
    ((1, 1, 64, 4096), ttnn.float32, None),
    ((1, 1, 32, 8192), ttnn.bfloat16, None),
    ((1, 1, 96, 1536), ttnn.bfloat16, None),
    ((1, 1, 32, 2048), ttnn.uint8, None),
    ((1, 1, 32, 2048), ttnn.int32, None),
    ((1, 1, 128, 64), ttnn.bfloat16, None),
    ((3, 32, 96), ttnn.bfloat16, None),
    ((1, 1, 2048, 2048), ttnn.bfloat16, None),
]


@pytest.mark.parametrize(
    "case", CORR, ids=lambda c: "x".join(map(str, c[0])) + "_" + str(c[1]).split(".")[-1] + ("_pad" if c[2] else "")
)
def test_corr(device, monkeypatch, case):
    _knobs(monkeypatch)
    shape, dtype, padded = case
    if dtype in (ttnn.uint8, ttnn.int32):
        x = torch.randint(0, 100, shape, dtype=torch.int32)
        if dtype == ttnn.uint8:
            x = x.to(torch.uint8)
    else:
        x = torch.randn(shape, dtype=torch.float32 if dtype == ttnn.float32 else torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    if padded is None:
        out = tilize(tt_in)
        assert torch.equal(ttnn.to_torch(out).to(x.dtype), x)
    else:
        out = tilize(tt_in, output_padded_shape=list(padded), pad_value=1.5)
        assert torch.equal(ttnn.to_torch(out), x)
        pads = tuple(j for i in reversed(range(x.dim())) for j in (0, padded[i] - x.shape[i]))
        assert torch.equal(out.cpu().to_torch_with_padded_shape(), torch.nn.functional.pad(x, pads, value=1.5))


@pytest.mark.parametrize("case", ["l1_interleaved", "tile16", "tile4_pad"])
def test_corr_extra(device, monkeypatch, case):
    _knobs(monkeypatch)
    if case == "l1_interleaved":
        x = torch.randn((1, 1, 64, 2048), dtype=torch.bfloat16)
        tt_in = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        out = tilize(tt_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert torch.equal(ttnn.to_torch(out), x)
    elif case == "tile16":
        x = torch.randn((1, 1, 48, 4096), dtype=torch.bfloat16)
        tt_in = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = tilize(tt_in, tile=ttnn.Tile([16, 32]))
        assert torch.equal(ttnn.to_torch(out), x)
    else:
        x = torch.randn((1, 1, 6, 1000), dtype=torch.bfloat16)
        tt_in = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = tilize(tt_in, tile=ttnn.Tile([4, 32]), output_padded_shape=[1, 1, 8, 1024], pad_value=2.0)
        assert torch.equal(ttnn.to_torch(out), x)
        ref = torch.nn.functional.pad(x, (0, 24, 0, 2), value=2.0)
        assert torch.equal(out.cpu().to_torch_with_padded_shape(), ref)
