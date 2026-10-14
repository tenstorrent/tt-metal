# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""reorder_cost bench: the real tilize descriptor, kernels from this dir (variant via RC_VARIANT ->
TILIZE_ABLATE_RC_<V> define). RC_KNOBS='{json}' overrides descriptor knobs. RC_ITERS profiled samples."""

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # no global torch import under ttnn/ (pre-commit)

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd

SHAPES = [
    (1, 1, 16384, 64),
    (1, 1, 16384, 32),
    (1, 1, 32768, 64),
    (1, 1, 128, 64),
    (1, 1, 3520, 64),
    (1, 1, 32, 2048),
    (1, 1, 2048, 2048),
    (1, 1, 1024, 512),
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s[2:])))
def test_reorder_bench(device, monkeypatch, shape):
    monkeypatch.setattr(pd, "KERNEL_DIR", Path(__file__).parent / os.environ.get("RC_KDIR", "kernels"))
    for name, value in json.loads(os.environ.get("RC_KNOBS", "{}")).items():
        assert hasattr(pd, name), name
        monkeypatch.setattr(pd, name, value)
    v = os.environ.get("RC_VARIANT", "")
    if v:
        monkeypatch.setattr(pd, "ABLATE", tuple(pd.ABLATE) + tuple(f"rc_{x}" for x in v.split(",")))
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    check = not pd.ABLATE or all(a.startswith("rc_") for a in pd.ABLATE)  # payload ablation: no gate
    for _ in range(int(os.environ.get("RC_ITERS", "1")) - 1):
        o = tilize(tt_in)
        assert not check or torch.equal(ttnn.to_torch(o), x)
        o.deallocate()
    out = tilize(tt_in)
    assert not check or torch.equal(ttnn.to_torch(out), x)
