# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Block-knob coverage for mhc_pre.

The default host knobs (BLOCK_TOKEN_TILES_CAP = 1, X_BLOCK_DEPTH_DEFAULT = 2) leave some kernel paths
unexercised by the acceptance suite: >1 token tile-row per block (matmul out-subblock > 1, multi-row
gather slots, multi-tile scatter), a ragged last block (extent < block_token_tiles), and the
x_block_depth = 1 fallback. These tests turn the knobs and check the same reference.
"""

import pytest
import torch
import ttnn

import ttnn.bringup.mhc_pre_ttnn.mhc_pre_program_descriptor as pd
from ttnn.bringup.mhc_pre_ttnn import mhc_pre

import importlib.util
import pathlib

# Reuse the acceptance test's reference and gates (the directory is not a package).
_spec = importlib.util.spec_from_file_location(
    "_mhc_pre_acceptance", pathlib.Path(__file__).with_name("test_mhc_pre.py")
)
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
make_inputs, to_dev, _check = _acc.make_inputs, _acc.to_dev, _acc._check


@pytest.mark.parametrize(
    "x_shape, bt_cap, depth",
    [
        ((1, 1, 640, 4 * 1792), 2, 2),  # 2 rows per block, 1 block per core
        ((1, 1, 1280, 4 * 1024), 3, 2),  # 4 rows per core -> blocks of 3 + ragged 1
        ((1, 2, 64, 4 * 512), 4, 1),  # depth-1 fallback, whole core share in one block
        ((1, 1, 1000, 4 * 256), 8, 2),  # ragged T, uneven token split, bt up to 8
    ],
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v),
)
def test_mhc_pre_block_knobs(device, monkeypatch, x_shape, bt_cap, depth):
    monkeypatch.setattr(pd, "BLOCK_TOKEN_TILES_CAP", bt_cap)
    monkeypatch.setattr(pd, "X_BLOCK_DEPTH_DEFAULT", depth)
    x, w, b, scale = make_inputs(x_shape, seed=5)
    tx, tw, tb = to_dev(x, device, ttnn.float32), to_dev(w, device, ttnn.float32), to_dev(b, device, ttnn.float32)
    plan = pd.make_plan(device, tx, tw, 4)
    assert plan.block_token_tiles == min(bt_cap, max(plan.core_token_tiles))
    assert plan.x_block_depth == depth
    y, post, comb = mhc_pre(tx, tw, tb, scale=scale)
    _check(y, post, comb, x, w, b, scale, ttnn.float32)


@pytest.mark.parametrize(
    "knobs",
    [
        dict(READER_NOC_FLIP_ROWS=0),  # every reader on READER_NOC (no flipped rows)
        dict(W_SHARE_ON_READER=False),  # the writer reads its W column share (Refinement 4 path)
        dict(READER_NOC_FLIP_ROWS=0, W_SHARE_ON_READER=False),
    ],
    ids=["noflip", "wwriter", "noflip_wwriter"],
)
@pytest.mark.parametrize(
    "x_dtype, w_dtype",
    [(ttnn.float32, ttnn.bfloat16), (ttnn.bfloat16, ttnn.float32)],
    ids=["xf32_wbf16", "xbf16_wf32"],
)
def test_mhc_pre_noc_placement_knobs(device, monkeypatch, knobs, x_dtype, w_dtype):
    """Non-default NoC-placement / W-share branches (Refinement 5). They shift when W lands relative to X, which
    exposed a missing resident-W wait on the fp32-X / bf16-W path; two seeds alternate so stale L1 from the
    previous call cannot mask a race."""
    for name, value in knobs.items():
        monkeypatch.setattr(pd, name, value)
    x_shape = (1, 1, 640, 4 * 1792)
    for seed in (7, 8):
        x, w, b, scale = make_inputs(x_shape, seed=seed)
        if w_dtype == ttnn.bfloat16:
            w = w.to(torch.bfloat16).to(torch.float32)  # the reference sees the W the device holds
        tx, tw, tb = to_dev(x, device, x_dtype), to_dev(w, device, w_dtype), to_dev(b, device, ttnn.float32)
        y, post, comb = mhc_pre(tx, tw, tb, scale=scale)
        _check(y, post, comb, x, w, b, scale, x_dtype)
