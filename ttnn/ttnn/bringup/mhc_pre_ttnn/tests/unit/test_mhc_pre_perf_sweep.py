# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf A/B probe for mhc_pre host knobs (Refinement 4): one call per (shape, dtype, knob setting).

Run under run_safe_pytest.sh --profile; the CSV lists one GenericOp per parametrization, in collection
order. Correctness is only smoke-checked here (the acceptance suite owns it).
"""

import pytest
import torch
import ttnn

import ttnn.bringup.mhc_pre_ttnn.mhc_pre_program_descriptor as pd
from ttnn.bringup.mhc_pre_ttnn import mhc_pre

SHAPES = [
    (1, 1, 640, 4 * 1792),
    (1, 1, 640, 4 * 7168),
    (1, 1, 1280, 4 * 4096),
    (1, 1, 4096, 4 * 1792),
    (1, 1, 2048, 4 * 5120),
]


import os

# Knob settings to A/B, each a {module attribute: value} patch on the program descriptor. MHC_PRE_PERF_KNOBS
# (comma-separated names) selects a subset.
ALL_KNOBS = {
    "default": {},
    "fullrow": dict(NARROW_GROUPS=False),
    "nostream": dict(X_STREAM_CHUNKS=1),
    "readernoc1": dict(READER_NOC=ttnn.NOC.NOC_1),
    "inflight_all": dict(X_STREAM_INFLIGHT=15),
    "inflight1": dict(X_STREAM_INFLIGHT=1),
    "chunks2": dict(X_STREAM_CHUNKS=2),
    "chunks6": dict(X_STREAM_CHUNKS=6),
    "chunks8": dict(X_STREAM_CHUNKS=8),
    "chunks8_if3": dict(X_STREAM_CHUNKS=8, X_STREAM_INFLIGHT=3),
    "own0": dict(OWNER_C_DISCOUNT=0),
    "own2": dict(OWNER_C_DISCOUNT=2),
    "own4": dict(OWNER_C_DISCOUNT=4),
    "own6": dict(OWNER_C_DISCOUNT=6),
    "own8": dict(OWNER_C_DISCOUNT=8),
    "own5": dict(OWNER_C_DISCOUNT=5),
    "own7": dict(OWNER_C_DISCOUNT=7),
    "noflip": dict(READER_NOC_FLIP_ROWS=0),
    "noflip_wwriter": dict(READER_NOC_FLIP_ROWS=0, W_SHARE_ON_READER=False),
    "flip2": dict(READER_NOC_FLIP_ROWS=2),
    "flip3": dict(READER_NOC_FLIP_ROWS=3),
    "flip4": dict(READER_NOC_FLIP_ROWS=4),
    "flip5": dict(READER_NOC_FLIP_ROWS=5),
    # Refinement 5: block x depth co-tune
    "bt2": dict(BLOCK_TOKEN_TILES_CAP=2),
    "bt2_d1": dict(BLOCK_TOKEN_TILES_CAP=2, X_BLOCK_DEPTH_DEFAULT=1),
    "bt4": dict(BLOCK_TOKEN_TILES_CAP=4),
    "d3": dict(X_BLOCK_DEPTH_DEFAULT=3),
    "d1": dict(X_BLOCK_DEPTH_DEFAULT=1),
    "ych4": dict(Y_CHUNK_TILES_CAP=4),
    "ych16": dict(Y_CHUNK_TILES_CAP=16),
    "ydepth3": dict(Y_DEPTH=3),
    "ydepth1": dict(Y_DEPTH=1),
    "wwriter": dict(W_SHARE_ON_READER=False),
    "wfirst": dict(W_SHARE_BEFORE_X=True),
    "wfirst_flip4": dict(W_SHARE_BEFORE_X=True, READER_NOC_FLIP_ROWS=4),
    "flip3_d1": dict(READER_NOC_FLIP_ROWS=3, X_BLOCK_DEPTH_DEFAULT=1),
    "flip4_d1": dict(READER_NOC_FLIP_ROWS=4, X_BLOCK_DEPTH_DEFAULT=1),
    "flip4_bt2_d1": dict(READER_NOC_FLIP_ROWS=4, BLOCK_TOKEN_TILES_CAP=2, X_BLOCK_DEPTH_DEFAULT=1),
    "flip4_d3": dict(READER_NOC_FLIP_ROWS=4, X_BLOCK_DEPTH_DEFAULT=3),
    "wwriter_flip4": dict(W_SHARE_ON_READER=False, READER_NOC_FLIP_ROWS=4),
    "inflight3": dict(X_STREAM_INFLIGHT=3),
    "chunks6_if3": dict(X_STREAM_CHUNKS=6, X_STREAM_INFLIGHT=3),
    "ych16_yd3": dict(Y_CHUNK_TILES_CAP=16, Y_DEPTH=3),
}
# Unselected (e.g. the plain unit-test run): a small smoke set covering each non-default placement branch.
DEFAULT_KNOBS = ("default", "noflip", "wwriter", "fullrow", "nostream")
_sel = os.environ.get("MHC_PRE_PERF_KNOBS")
_sel = list(ALL_KNOBS) if _sel == "all" else (_sel.split(",") if _sel else DEFAULT_KNOBS)
KNOBS = {k: ALL_KNOBS[k] for k in _sel}


@pytest.mark.parametrize("knobs", list(KNOBS), ids=list(KNOBS))
@pytest.mark.parametrize("x_dtype", [ttnn.bfloat16, ttnn.float32], ids=["xbf16", "xf32"])
@pytest.mark.parametrize("x_shape", SHAPES, ids=lambda s: "X" + "x".join(map(str, s)))
def test_mhc_pre_perf_sweep(device, monkeypatch, x_shape, x_dtype, knobs):
    for name, value in KNOBS[knobs].items():
        monkeypatch.setattr(pd, name, value)
    torch.manual_seed(0)
    nc = x_shape[-1]
    x = torch.randn(x_shape, dtype=torch.float32)
    w = torch.randn((nc, 24), dtype=torch.float32) / nc**0.5
    b = torch.randn((1, 24), dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=x_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y, post, comb = mhc_pre(tx, tw, tb, scale=(1.0, 1.0, 1.0))
    for t in (y, post, comb):
        assert torch.isfinite(ttnn.to_torch(t)).all()
