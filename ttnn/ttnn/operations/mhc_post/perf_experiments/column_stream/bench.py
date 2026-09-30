# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""column_stream bake-off harness: build inputs, run a variant through ttnn.generic_op, torch reference.

Variants: "baseline" (frozen copy of the op's kernels + descriptor) or a dict of column-stream knobs
(column_stream_descriptor.KNOBS). Every variant runs under the op's default compute config (HiFi4, fp32 DEST,
approx off) — never tuned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ttnn

from ttnn.operations.mhc_post.mhc_post import default_compute_kernel_config

from . import baseline_descriptor
from . import column_stream_descriptor

HERE = Path(__file__).parent
LABELS = HERE / "run_labels.jsonl"  # one line per executed op, in dispatch order (maps profiler CSV rows)


def make_inputs(T, C, n, x_dtype, f_dtype, seed=0):
    import torch  # local: no global torch import under ttnn/ (pre-commit)

    torch.manual_seed(seed)
    f = torch.randn(1, 1, T, C)
    x = torch.randn(1, 1, T, n * C)
    post = torch.rand(1, 1, T, n) * 2
    comb = torch.rand(1, 1, T, n * n)
    return f, x, post, comb


def reference(f, x, post, comb, n, x_dtype, f_dtype):
    import torch

    C = f.shape[-1]
    if f_dtype == ttnn.bfloat16:
        f = f.bfloat16().float()
    if x_dtype == ttnn.bfloat16:
        x = x.bfloat16().float()
    ref = post.reshape(-1, n, 1) * f.reshape(-1, 1, C) + torch.einsum(
        "tij,tic->tjc", comb.reshape(-1, n, n), x.reshape(-1, n, C)
    )
    return ref.reshape(x.shape)


def to_device(device, f, x, post, comb, x_dtype, f_dtype):
    def dev(t, dt):
        return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    return dev(f, f_dtype), dev(x, x_dtype), dev(post, ttnn.float32), dev(comb, ttnn.float32)


def run(device, variant, tensors, label=None):
    f_t, x_t, p_t, m_t = tensors
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(x_t.shape)), x_t.dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = default_compute_kernel_config()
    info = {}
    if variant == "baseline":
        pd = baseline_descriptor.create_program_descriptor(f_t, x_t, p_t, m_t, out, cfg)
    else:
        pd = column_stream_descriptor.create_program_descriptor(f_t, x_t, p_t, m_t, out, cfg, knobs=variant)
        info = dict(column_stream_descriptor.create_program_descriptor.last)
    res = ttnn.generic_op([f_t, x_t, p_t, m_t, out], pd)
    if label is not None and os.environ.get("CS_LABELS"):
        with open(LABELS, "a") as fh:
            fh.write(json.dumps({"label": label, **info}) + "\n")
    return res
