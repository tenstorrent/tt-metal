# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""geo_read_order bench: experiment copy of tilize (this dir) on the perf shapes.

GEO_KNOBS='{"KNOB": value}' overrides descriptor knobs; GEO_ITERS=N runs N profiled samples.
Correctness gate: torch.equal vs the input (tilize is a pure re-layout).
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")

import ttnn
from ttnn.operations.tilize.perf_experiments.geo_read_order import tilize_program_descriptor as pd
from ttnn.operations.tilize.perf_experiments.geo_read_order.tilize import tilize

SHAPES = [
    (1, 1, 16384, 64),
    (1, 1, 16384, 32),
    (1, 1, 32768, 64),
    (1, 1, 3520, 64),
    (1, 1, 2048, 2048),
    (1, 1, 128, 64),
    (1, 1, 16384, 128),
    (2, 3, 4096, 96),
]


@pytest.mark.parametrize(
    "shape", SHAPES, ids=lambda s: "x".join(map(str, s[2:])) if s[0] == 1 else "x".join(map(str, s))
)
def test_geo(device, monkeypatch, shape):
    for name, value in json.loads(os.environ.get("GEO_KNOBS", "{}")).items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(int(os.environ.get("GEO_ITERS", "1")) - 1):
        tilize(tt_in).deallocate()
    out = tilize(tt_in)
    if not pd.ABLATE:  # ablation drops payloads: output is wrong by design
        assert torch.equal(ttnn.to_torch(out), x)
