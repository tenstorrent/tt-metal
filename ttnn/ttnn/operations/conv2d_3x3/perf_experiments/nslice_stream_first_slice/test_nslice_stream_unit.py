# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness of the streamed N-slice walk on this dir's op copy: the BLOCK unit-test geometries (RM bf16, TILE bf16,
TILE bf8_b; bias+ReLU and plain; short last channel / N-slice; columns that own no output; multi-M-block shards with
one-tile-row blocks; the ragged-row weight multicast). Each test runs the copy's baseline walk and every stream variant
on the same inputs and requires torch.equal against the baseline output (and the baseline's PCC vs torch).

    NSFU_VARIANTS=s1x2,s4x2,rec scripts/run_safe_pytest.sh --run-all <this file> --import-mode=prepend
"""

import importlib
import os

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

_PKG = "ttnn.operations.conv2d_3x3.perf_experiments.nslice_stream_first_slice.op_copy"
op = importlib.import_module(_PKG + ".conv2d_3x3")
pd = importlib.import_module(_PKG + ".conv2d_3x3_program_descriptor")

KNOBS = dict(NSLICE_STREAM_ROWS=0, NSLICE_STREAM_SB_H=None, NSLICE_STREAM_SLICES=1)
VARIANTS = {
    "s1x2": dict(NSLICE_STREAM_ROWS=1, NSLICE_STREAM_SLICES=2),
    "s2x2": dict(NSLICE_STREAM_ROWS=2, NSLICE_STREAM_SLICES=2),
    "s4x2": dict(NSLICE_STREAM_ROWS=4, NSLICE_STREAM_SLICES=2),
    "s4": dict(NSLICE_STREAM_ROWS=4),
    "s2x3": dict(NSLICE_STREAM_ROWS=2, NSLICE_STREAM_SLICES=3),
    "rec": dict(NSLICE_STREAM_ROWS=-1, NSLICE_STREAM_SLICES=2),
}
SEL = [v for v in os.environ.get("NSFU_VARIANTS", "s1x2,s4x2,s2x3,rec").split(",") if v]

RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
LAYOUTS = {"rm_bf16": (ttnn.bfloat16, RM), "tile_bf16": (ttnn.bfloat16, TL), "tile_bf8b": (ttnn.bfloat8_b, TL)}
SHAPES = [
    (((16, 16, 64), 128, 1, 1, 1), {RM: (40, 16), TL: (64, 32)}),
    (((32, 64, 64), 32, 8, 1, 8), {RM: (256, 16), TL: (256, 32)}),
    (((16, 34, 48), 96, 0, 1, 1), {RM: (68, 32), TL: (96, 32)}),
    (((8, 16, 256), 256, 1, 1, 1), {RM: (32, 32), TL: (32, 32)}),
    # stride 2 (split row gather) with a K split
    (((32, 64, 256), 256, 1, 2, 1), {RM: (256, 32), TL: (256, 32)}),
]


def _block_config(num_px, c_in, layout, shard):
    shard_rows, shard_width = shard
    if layout == TL:
        num_px = -(-num_px // 32) * 32
    rows, cols = -(-num_px // shard_rows), -(-c_in // shard_width)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_rows, shard_width], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _one_tile_row_blocks(real_select):
    def select(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        fit = pd._footprint(g, b.core_m_tiles, 1, b.block_k_tiles, b.block_n_tiles, nslice=b.nslice)
        return pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=1,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    return select


CASES = []
for shape in SHAPES:
    for lid in LAYOUTS:
        for br in ("bias_relu", "plain"):
            (H, W, c_in), c_out, p, s, d = shape[0]
            CASES.append(
                pytest.param(shape, lid, br, False, id=f"H{H}_W{W}_Ci{c_in}_Co{c_out}_p{p}_s{s}_d{d}-{lid}-{br}")
            )
# multi-M-block (one-tile-row M-blocks inside 4- and 7-tile-row shards; plain N-slice walk, slice CBs depth 2)
MULTI = [
    ((((16, 64, 64), 96, 1, 1, 1), {TL: (128, 32), RM: (128, 16)}), "tile_bf8b", "bias_relu"),
    ((((16, 64, 64), 96, 1, 1, 1), {TL: (128, 32), RM: (128, 16)}), "rm_bf16", "relu"),
    ((((16, 40, 64), 64, 1, 1, 1), {TL: (224, 32), RM: (224, 16)}), "tile_bf16", "bias_relu"),
    ((((16, 40, 64), 64, 1, 1, 1), {TL: (224, 32), RM: (224, 16)}), "tile_bf8b", "bias_relu"),
]
for shape, lid, br in MULTI:
    (H, W, c_in), c_out, p, s, d = shape[0]
    CASES.append(pytest.param(shape, lid, br, True, id=f"multi_H{H}_W{W}_Ci{c_in}_Co{c_out}-{lid}-{br}"))


def _run_once(shape, lid, br, device, seed):
    import torch

    torch.manual_seed(seed)
    ((H, W, c_in), c_out, p, s, d), shards = shape
    dtype, layout = LAYOUTS[lid]
    config = _block_config(H * W, c_in, layout, shards[layout])
    with_bias = br == "bias_relu"
    relu = br in ("bias_relu", "relu")
    torch_weight = torch.randn(c_out, c_in, 3, 3).to(torch.bfloat16)
    torch_bias = torch.randn(c_out).to(torch.bfloat16) if with_bias else None
    torch_input = torch.randn(1, 1, H * W, c_in).to(torch.bfloat16)
    kwargs = dict(
        stride=s,
        padding=p,
        dilation=d,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=ttnn.bfloat8_b,
        input_memory_config=config,
        device=device,
    )
    tt_weight = op.prepare_conv2d_weights(torch_weight, **kwargs)
    tt_bias = op.prepare_conv2d_bias(torch_bias, **kwargs) if with_bias else None
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=config)
    tt_out = op.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=p,
        stride=s,
        dilation=d,
        activation="relu" if relu else None,
    )
    out = ttnn.to_torch(tt_out).float()
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    ref = torch.nn.functional.conv2d(
        x, torch_weight.float(), None if torch_bias is None else torch_bias.float(), stride=s, padding=p, dilation=d
    )
    if relu:
        ref = torch.relu(ref)
    ref = ref.permute(0, 2, 3, 1).reshape(1, 1, -1, c_out)
    return out, ref


@pytest.mark.parametrize("shape,lid,br,multi", CASES)
def test_stream_bit_exact(shape, lid, br, multi, device, monkeypatch):
    import torch

    real_select = pd.select_blocking
    if multi:
        monkeypatch.setattr(op, "select_blocking", _one_tile_row_blocks(real_select))
    streams = {}

    def run(variant):
        for k, v in {**KNOBS, **VARIANTS.get(variant, {})}.items():
            monkeypatch.setattr(pd, k, v)
        pd.select_blocking.cache_clear()
        captured = {}
        sel = op.select_blocking

        def recording(g, grid, l1_budget):
            b = sel(g, grid, l1_budget)
            captured["stream"] = pd._nslice_stream_args(b, b.block_m_tiles, g.out_n_tiles) if b.nslice else None
            return b

        monkeypatch.setattr(op, "select_blocking", recording)
        try:
            out, ref = _run_once(shape, lid, br, device, seed=42)
        finally:
            monkeypatch.setattr(op, "select_blocking", sel)
        streams[variant] = captured.get("stream")
        return out, ref

    base, ref = run("baseline")
    ok, pcc = comp_pcc(ref, base, 0.99)
    assert ok and torch.isfinite(base).all(), f"baseline pcc {pcc}"
    bad = []
    for v in SEL:
        out, _ = run(v)
        if not torch.equal(out, base):
            bad.append((v, streams[v], float((out - base).abs().max())))
    print(f"streams: {streams}")
    assert not bad, f"not bit-identical to the baseline walk: {bad}"
