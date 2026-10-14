# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""dual_noc_reads bench: the experiment copy of tilize (this dir) on the perf-focus shapes.

DUAL_NOC_KNOBS='{"WRITER_READ_SHARE": 0.5}' overrides descriptor knobs; DUAL_NOC_ITERS=N runs N
profiled samples. Correctness gate: torch.equal vs the input.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")  # no global torch import under ttnn/ (pre-commit)

import ttnn
from ttnn.operations.tilize.perf_experiments.dual_noc_reads import tilize_program_descriptor as pd
from ttnn.operations.tilize.perf_experiments.dual_noc_reads.tilize import tilize

SHAPES = [
    (1, 1, 16384, 64),
    (1, 1, 16384, 32),
    (1, 1, 32768, 64),
    (1, 1, 128, 64),
    (1, 1, 3520, 64),
    (1, 1, 65536, 64),
    (1, 1, 16384, 256),
    (2, 3, 4096, 96),
]


@pytest.mark.parametrize(
    "shape", SHAPES, ids=lambda s: "x".join(map(str, s[2:])) if s[0] == 1 else "x".join(map(str, s))
)
def test_dual_noc(device, monkeypatch, shape):
    for name, value in json.loads(os.environ.get("DUAL_NOC_KNOBS", "{}")).items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(int(os.environ.get("DUAL_NOC_ITERS", "1")) - 1):
        tilize(tt_in).deallocate()
    out = tilize(tt_in)
    assert torch.equal(ttnn.to_torch(out), x)
