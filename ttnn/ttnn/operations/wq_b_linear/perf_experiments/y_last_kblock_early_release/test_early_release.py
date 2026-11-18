# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Whole-op A/B of the last-K-block y early release (kernels/ + early_release_descriptor.py in this dir).

YER_CONFIGS="none:4,column:4,row:4f" = Y_RELEASE:Y_READER_SHARE_TENTHS[f = Y_SPLIT_FRONT] per candidate, run in order after one call of
the op's own descriptor ("ref"). Each candidate call is one profiled op row; it must be bit-identical to ref and
meet PCC >= 0.9998 vs fp64 of the quantized operands.
"grad" runs graduate/ (the column release as a patch-ready copy of the op).
YER_KNOBS="CORE_N_TILES=4,K_REMAINDER_SPLIT=True" overrides descriptor knobs on both the op and the candidate.
YER_SHAPE="M,K,N" (default 160,1536,4096).
"""

import os

import pytest
import ttnn

from ttnn.operations.wq_b_linear.perf_experiments.y_last_kblock_early_release import early_release_descriptor as erd
import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as opd
from ttnn.operations.wq_b_linear.perf_experiments.y_last_kblock_early_release.graduate import (
    wq_b_linear_program_descriptor as grd,
)
from eval.golden_tests.wq_b_linear.helpers import WEIGHT_SCALE
from ttnn.operations.wq_b_linear.wq_b_linear import default_compute_kernel_config

CONFIGS = os.environ.get("YER_CONFIGS", "none:4,column:4,row:4").split(",")
M, K, N = map(int, os.environ.get("YER_SHAPE", "160,1536,4096").split(","))
PCC = 0.9998

for _kv in filter(None, os.environ.get("YER_KNOBS", "").split(",")):
    _k, _v = _kv.split("=")
    for mod in (erd, opd, grd):
        cur = getattr(mod, _k)
        setattr(mod, _k, _v == "True" if isinstance(cur, bool) else type(cur)(_v))


def _run(module, tx, tw):
    device = tx.device()
    y = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, M, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    y = ttnn.generic_op([tx, tw, y], module.create_program_descriptor(tx, tw, y, default_compute_kernel_config()))
    out = ttnn.to_torch(y)
    ttnn.ReadDeviceProfiler(device)
    ttnn.deallocate(y)
    return out


def test_early_release(device):
    import torch

    torch.manual_seed(0)
    x = torch.randn(1, 1, M, K).bfloat16()
    w = (torch.randn(K, N) * WEIGHT_SCALE).bfloat16()
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    expected = torch.matmul(x.to(torch.float64), ttnn.to_torch(tw).to(torch.float64)).flatten()
    ref = _run(opd, tx, tw)
    labels = ["ref"]
    for config in CONFIGS:
        if config == "grad":
            out = _run(grd, tx, tw)
        else:
            mode, share = config.split(":")
            erd.Y_SPLIT_FRONT = share.endswith("f")
            erd.Y_RELEASE, erd.Y_READER_SHARE_TENTHS = mode, int(share.rstrip("f"))
            out = _run(erd, tx, tw)
        pcc = torch.corrcoef(torch.stack([out.flatten().double(), expected]))[0, 1].item()
        same = torch.equal(out, ref)
        print(f"YER config={config} shape={M},{K},{N} pcc={pcc:.6f} bit_identical={same}")
        assert pcc >= PCC, (config, pcc)
        assert same, config
        labels.append(config)
    print("YER labels=" + ",".join(labels))
