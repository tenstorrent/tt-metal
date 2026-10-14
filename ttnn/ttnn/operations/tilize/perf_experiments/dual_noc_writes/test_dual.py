# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""dual_noc_writes bench: the op with this dir's descriptor/kernels. Run under --profile.

DUAL_KNOBS='{"DUAL_NUM": 1, "DUAL_DEN": 2}' sets dual_pd knobs (DUAL_NUM=0 = baseline);
DUAL_ITERS=N runs the op N times (N profiled samples). Gate: torch.equal.
"""

import importlib
import json
import os

import pytest

torch = pytest.importorskip("torch")  # no global torch import under ttnn/ (pre-commit)

import ttnn

from ttnn.operations.tilize.perf_experiments.dual_noc_writes import dual_pd

tilize_mod = importlib.import_module("ttnn.operations.tilize.tilize")


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 16384, 64),
        (1, 1, 16384, 32),
        (1, 1, 32768, 64),
        (1, 1, 128, 64),
        (1, 1, 3520, 64),
        (1, 1, 32, 2048),
        (1, 1, 2048, 2048),
        (1, 1, 32, 8192),
        (1, 1, 100, 64),  # padded (segment path + pad fill)
        (2, 3, 160, 96),  # multi-image
    ],
    ids=lambda s: "x".join(map(str, s[2:])) if s[0] == 1 and s[1] == 1 else "x".join(map(str, s)),
)
def test_dual(device, monkeypatch, shape):
    for name, value in json.loads(os.environ.get("DUAL_KNOBS", "{}")).items():
        assert hasattr(dual_pd, name), name
        monkeypatch.setattr(dual_pd, name, value)
    monkeypatch.setattr(tilize_mod, "create_program_descriptor", dual_pd.create_program_descriptor)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_in = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    for _ in range(int(os.environ.get("DUAL_ITERS", "1")) - 1):
        tilize_mod.tilize(tt_in).deallocate()
    kw = {}
    if shape[-2] % 32:
        kw = dict(output_padded_shape=list(shape[:-2]) + [-(-shape[-2] // 32) * 32, shape[-1]], pad_value=0.0)
    out = tilize_mod.tilize(tt_in, **kw)
    got = ttnn.to_torch(out)
    if not torch.equal(got, x):
        g2, x2 = got.reshape(-1, shape[-1]), x.reshape(-1, shape[-1])
        bad = (
            (g2 != x2)
            .reshape(-1, 32, shape[-1] // 32 if shape[-1] % 32 == 0 else 1, 32 if shape[-1] % 32 == 0 else shape[-1])
            .any(dim=(1, 3))
            if shape[-2] % 32 == 0
            else None
        )
        if bad is not None:
            rows = bad.any(dim=1).nonzero().flatten().tolist()
            print(f"MISMATCH tile-rows ({len(rows)}/{bad.shape[0]}): {rows[:40]}")
            r0 = rows[0]
            for src in range(bad.shape[0]):
                if torch.equal(g2[r0 * 32 : r0 * 32 + 32], x2[src * 32 : src * 32 + 32]):
                    print(f"tile-row {r0} holds input tile-row {src}")
            print("zeros?", bool((g2[r0 * 32 : r0 * 32 + 32] == 0).all()))
    assert torch.equal(got, x)
