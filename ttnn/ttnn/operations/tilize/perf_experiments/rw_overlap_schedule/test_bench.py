# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rw_overlap_schedule bench: the op copy in ./op (baseline = default knobs; candidates = knobs).

RW_KNOBS='{"ROW_SCHEDULE": 1}' overrides ./op/tilize_program_descriptor knobs; RW_ITERS=N runs N
profiled samples. RW_LOW_L1=1 exercises the low_l1 path. Gate: torch.equal.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")  # no global torch import under ttnn/ (pre-commit)

import ttnn
from ttnn.operations.tilize.perf_experiments.rw_overlap_schedule.op import tilize
from ttnn.operations.tilize.perf_experiments.rw_overlap_schedule.op import tilize_program_descriptor as pd


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 16384, 64),
        (1, 1, 16384, 32),
        (1, 1, 32768, 64),
        (1, 1, 128, 64),
        (1, 1, 3520, 64),
        (1, 1, 16384, 128),
        (1, 1, 65536, 64),
        (2, 3, 4096, 96),
    ],
    ids=lambda s: "x".join(map(str, s)),
)
def test_bench(device, monkeypatch, shape):
    for name, value in json.loads(os.environ.get("RW_KNOBS", "{}")).items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)
    if os.environ.get("RW_LOW_L1"):
        orig = pd.create_program_descriptor
        monkeypatch.setattr(pd, "create_program_descriptor", lambda *a, **k: orig(*a, **{**k, "low_l1": True}))
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(int(os.environ.get("RW_ITERS", "1")) - 1):
        tilize(tt_in).deallocate()
    out = tilize(tt_in)
    assert torch.equal(ttnn.to_torch(out), x)
