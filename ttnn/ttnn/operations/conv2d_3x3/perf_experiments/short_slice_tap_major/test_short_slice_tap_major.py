# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""short_slice_tap_major bake-off: tap-major im2col on a BLOCK K split whose last channel slice is short.

Variants (env, read at import of the op copy):
  SSTM_OP=real           the committed op (ttnn.operations.conv2d_3x3), the honest baseline
  SSTM_OP=copy           the op copy in op_copy/ with
    SSTM_SHORT_SLICE=off   the committed predicate (every column row-gathers when one slice is short)
    SSTM_SHORT_SLICE=on    candidate: the short slice's padded lanes are zero in the slab, tap-major everywhere
    SSTM_SHORT_SLICE=naive negative control: tap-major on the short slice WITHOUT zeroing its lanes
  SSTM_NO_ZONES=1        compile the permanent device zones out of the copy

test_sstm_perf: one dispatch per case (profile it with run_safe_pytest.sh --profile); PCC / rel-RMS printed and
gated. test_sstm_identity: the copy (off, on) vs the committed op on one input, with the shard padding of the input
pre-poisoned with NaN, so a lane that leaks into the result shows.
"""

import importlib
import os

import pytest
import ttnn

from tests.ttnn.utils_for_testing import comp_pcc

_COPY = "ttnn.operations.conv2d_3x3.perf_experiments.short_slice_tap_major.op_copy"
OP = os.environ.get("SSTM_OP", "copy")


def _op_modules(which):
    if which == "real":
        pkg = importlib.import_module("ttnn.operations.conv2d_3x3")
        return pkg, importlib.import_module("ttnn.operations.conv2d_3x3.conv2d_3x3_program_descriptor")
    pkg = importlib.import_module(_COPY)
    return pkg, importlib.import_module(_COPY + ".conv2d_3x3_program_descriptor")


BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
LOFI, HIFI4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi4
BF8, BF16 = ttnn.bfloat8_b, ttnn.bfloat16
TL, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT

# id: ((H, W, C_in), C_out, padding, stride, dilation), dtype, layout, shard, (gx, gy), bias+relu, fidelity, bm
CASES = {
    # focus: the perf-flagged LOOSE rows (feature_spec "[perf]"), exact config
    "aspp1": (((32, 64, 2048), 256, 6, 1, 6), BF8, TL, [256, 192], (11, 8), True, LOFI, None),
    "aspp2": (((32, 64, 2048), 256, 12, 1, 12), BF8, TL, [256, 192], (11, 8), True, LOFI, None),
    # domain
    "aspp3": (((32, 64, 1024), 256, 18, 1, 18), BF8, TL, [256, 96], (11, 8), False, HIFI4, None),
    "aspp3_lofi": (((32, 64, 1024), 256, 18, 1, 18), BF8, TL, [256, 96], (11, 8), False, LOFI, None),
    "aspp1_hifi4": (((32, 64, 2048), 256, 6, 1, 6), BF8, TL, [256, 192], (11, 8), True, HIFI4, None),
    "aspp1_bf16": (((32, 64, 2048), 256, 6, 1, 6), BF16, TL, [256, 192], (11, 8), True, LOFI, None),
    "aspp1_rm": (((32, 64, 2048), 256, 6, 1, 6), BF16, RM, [256, 192], (11, 8), True, LOFI, None),
    # no short slice (8 full slices of 32): tap-major already applies in the committed op; probes option zsrc only
    "res4_lofi": (((32, 64, 256), 256, 1, 1, 1), BF8, TL, [256, 32], (8, 8), True, LOFI, None),
    "res4_0_lofi": (((64, 128, 256), 256, 1, 2, 1), BF8, TL, [1024, 32], (8, 8), True, LOFI, None),
    # small: d 1, 4 slices of 96 with a 32-channel last slice (TILE whole-tile padding: 2 padding tiles)
    "c320_tile": (((16, 32, 320), 64, 1, 1, 1), BF8, TL, [128, 96], (4, 4), True, HIFI4, None),
    "c320_rm": (((16, 32, 320), 64, 1, 1, 1), BF16, RM, [128, 96], (4, 4), True, HIFI4, None),
    # one-tile-row M-blocks: 4 M-blocks per core, the cb_input_tiles ring and the slab reused across blocks
    "c320_tile_bm1": (((16, 32, 320), 64, 1, 1, 1), BF8, TL, [128, 96], (4, 4), True, HIFI4, 1),
    "c320_rm_bm1": (((16, 32, 320), 64, 1, 1, 1), BF16, RM, [128, 96], (4, 4), True, HIFI4, 1),
    # the unit suite's short slice (16 of 32): TILE lanes are tile padding (stays row gather), RM takes the zeroed slab
    "c48_tile": (((16, 34, 48), 96, 0, 1, 1), BF16, TL, [96, 32], (2, 6), False, HIFI4, None),
    "c48_rm": (((16, 34, 48), 96, 0, 1, 1), BF16, RM, [68, 32], (2, 8), False, HIFI4, None),
}

GATE = {"pcc": 0.996, "rel_rms": 0.08}  # focus soft gate (feature_spec LoFi [perf] tolerance)


def _memory_config(shard, grid_xy):
    gx, gy = grid_xy
    return ttnn.MemoryConfig(
        BS,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))]),
            shard,
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def _force_bm(pd, op_module, bm, monkeypatch):
    real_select = pd.select_blocking

    def select(g, grid, l1_budget):
        b = real_select(g, grid, l1_budget)
        fit = pd._footprint(g, b.core_m_tiles, bm, b.block_k_tiles, b.block_n_tiles)
        return pd.Conv3x3Blocking(
            grid_x=b.grid_x,
            core_m_tiles=b.core_m_tiles,
            num_m_groups=b.num_m_groups,
            block_m_tiles=bm,
            block_k_tiles=b.block_k_tiles,
            block_n_tiles=b.block_n_tiles,
            im2col_rm_depth_rows=b.im2col_rm_depth_rows,
            out_depth_row_groups=b.out_depth_row_groups,
            **fit,
        )

    monkeypatch.setattr(op_module, "select_blocking", select)


def _tensors(case_id, seed=0):
    import torch

    ((H, W, c_in), c_out, p, s, d), dtype, layout, shard, grid, bias_relu, fidelity, bm = CASES[case_id]
    g = torch.Generator().manual_seed(seed)
    torch_input = torch.randn(1, 1, H * W, c_in, generator=g).to(torch.bfloat16)
    torch_weight = torch.randn(c_out, c_in, 3, 3, generator=g).to(torch.bfloat16)
    torch_bias = torch.randn(c_out, generator=g).to(torch.bfloat16) if bias_relu else None
    return torch_input, torch_weight, torch_bias


def _reference(case_id, torch_input, torch_weight, torch_bias):
    import torch

    ((H, W, c_in), c_out, p, s, d), _, _, _, _, bias_relu, _, _ = CASES[case_id]
    x = torch_input.float().reshape(1, H, W, c_in).permute(0, 3, 1, 2)
    out = torch.nn.functional.conv2d(
        x, torch_weight.float(), None if torch_bias is None else torch_bias.float(), stride=s, padding=p, dilation=d
    )
    if bias_relu:
        out = torch.relu(out)
    return out.permute(0, 2, 3, 1).reshape(1, 1, -1, out.shape[1])


def _poison_shard_padding(case_id, device):
    """An L1 tensor on the same shard grid whose every shard is full of NaN; freed, so the next allocation of the
    same shard size reuses its pages and the input's physical shard padding holds NaN."""
    import torch

    ((H, W, c_in), *_), dtype, layout, shard, grid, *_ = CASES[case_id]
    cols = grid[0]
    nan = torch.full((1, 1, H * W, cols * shard[1]), float("nan"), dtype=torch.bfloat16)
    t = ttnn.from_torch(nan, dtype=dtype, layout=layout, device=device, memory_config=_memory_config(shard, grid))
    addr = t.buffer_address()
    ttnn.deallocate(t)
    return addr


def _lane_poisoned_weights(pkg, torch_weight, kwargs):
    """prepare_conv2d_weights, except the rows of the short slice's padded lanes are 1 instead of 0: a lane that is
    not exactly zero in the im2col block (or is NaN / Inf) now changes the result."""
    import torch.nn.functional as F

    op_module = importlib.import_module(pkg.__name__ + ".conv2d_3x3")
    c_out, c_in = int(torch_weight.shape[0]), int(torch_weight.shape[1])
    c_in_p = op_module.per_pixel_channels(c_in, kwargs["input_layout"], kwargs["input_memory_config"])
    slices = op_module.channel_slices(c_in, kwargs["input_layout"], kwargs["input_memory_config"])
    kh_rows = -(-3 * c_in_p // 32) * 32
    w = torch_weight.permute(2, 3, 1, 0)
    w = F.pad(w, (0, 0, 0, slices * c_in_p - c_in), value=1.0)
    w = w.reshape(3, 3, slices, c_in_p, c_out).permute(2, 0, 1, 3, 4).reshape(slices, 3, 3 * c_in_p, c_out)
    w = F.pad(w, (0, 0, 0, kh_rows - 3 * c_in_p)).reshape(1, 1, slices * 3 * kh_rows, c_out)
    return ttnn.from_torch(
        w,
        dtype=kwargs["weight_dtype"],
        layout=ttnn.TILE_LAYOUT,
        device=kwargs["device"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


PATHS = []  # (variant, tap_major, short slice lanes zeroed) per dispatch of the copy


def _run(which, case_id, torch_input, torch_weight, torch_bias, device, monkeypatch, poison=False):
    ((H, W, c_in), c_out, p, s, d), dtype, layout, shard, grid, bias_relu, fidelity, bm = CASES[case_id]
    pkg, pd = _op_modules(which)
    op_module = importlib.import_module(pkg.__name__ + ".conv2d_3x3")
    if bm is not None:
        _force_bm(pd, op_module, bm, monkeypatch)
    real_create = pd.create_program_descriptor

    def recording_create(*args, geometry, blocking, **kw):
        lanes = pd.short_slice_lanes_zeroed(geometry) if hasattr(pd, "short_slice_lanes_zeroed") else False
        PATHS.append(
            (which, pd.tap_major_im2col(geometry, blocking), lanes, blocking.block_k_tiles, blocking.block_m_tiles)
        )
        return real_create(*args, geometry=geometry, blocking=blocking, **kw)

    monkeypatch.setattr(op_module, "create_program_descriptor", recording_create)
    memory_config = _memory_config(shard, grid)
    kwargs = dict(
        stride=s,
        padding=p,
        dilation=d,
        input_dtype=dtype,
        input_layout=layout,
        weight_dtype=BF8,
        input_memory_config=memory_config,
        device=device,
    )
    tt_weight = (
        _lane_poisoned_weights(pkg, torch_weight, kwargs)
        if poison
        else pkg.prepare_conv2d_weights(torch_weight, **kwargs)
    )
    tt_bias = pkg.prepare_conv2d_bias(torch_bias, **kwargs) if torch_bias is not None else None
    poisoned_addr = _poison_shard_padding(case_id, device) if poison else None
    tt_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    if poison:
        assert tt_input.buffer_address() == poisoned_addr, "the input did not land on the poisoned pages"
    tt_output = pkg.conv2d_3x3(
        tt_input,
        tt_weight,
        input_height=H,
        input_width=W,
        bias=tt_bias,
        padding=p,
        stride=s,
        dilation=d,
        activation="relu" if bias_relu else None,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False
        ),
    )
    out = ttnn.to_torch(tt_output).float()
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)
    return out


def _metrics(ref, out):
    out = out.reshape(ref.shape)
    _, pcc = comp_pcc(ref, out)
    rel_rms = ((out - ref).pow(2).mean().sqrt() / ref.std()).item()
    return float(pcc), rel_rms


@pytest.mark.parametrize("case_id", list(CASES))
def test_sstm_perf(case_id, device, monkeypatch):
    import torch

    torch_input, torch_weight, torch_bias = _tensors(case_id)
    out = _run(OP, case_id, torch_input, torch_weight, torch_bias, device, monkeypatch)
    ref = _reference(case_id, torch_input, torch_weight, torch_bias)
    pcc, rel_rms = _metrics(ref, out)
    mode = os.environ.get("SSTM_SHORT_SLICE", "on") if OP == "copy" else "committed"
    print(f"\nSSTM_RESULT case={case_id} op={OP} mode={mode} pcc={pcc:.6f} rel_rms={rel_rms:.5f} path={PATHS[-1]}")
    if mode != "naive":
        assert torch.isfinite(out).all()
        assert pcc >= GATE["pcc"] and rel_rms <= GATE["rel_rms"], (pcc, rel_rms)


@pytest.mark.parametrize("case_id", list(CASES))
def test_sstm_identity(case_id, device, monkeypatch):
    """Committed op vs the copy (off / on), same input, NaN-poisoned shard padding. Prints max |diff|."""
    import torch

    pd = importlib.import_module(_COPY + ".conv2d_3x3_program_descriptor")
    torch_input, torch_weight, torch_bias = _tensors(case_id)
    ref = _reference(case_id, torch_input, torch_weight, torch_bias)
    outs = {"real": _run("real", case_id, torch_input, torch_weight, torch_bias, device, monkeypatch, poison=True)}
    paths = {"real": PATHS[-1]}
    for mode in ("off", "on", "naive"):
        monkeypatch.setattr(pd, "SHORT_SLICE_TAP_MAJOR", mode != "off")
        monkeypatch.setattr(pd, "SHORT_SLICE_NAIVE", mode == "naive")
        outs[mode] = _run("copy", case_id, torch_input, torch_weight, torch_bias, device, monkeypatch, poison=True)
        paths[mode] = PATHS[-1]
    for name, out in outs.items():
        pcc, rel_rms = _metrics(ref, out)
        diff = (out - outs["real"]).abs().max().item()
        print(
            f"\nSSTM_IDENTITY case={case_id} variant={name} pcc={pcc:.6f} rel_rms={rel_rms:.5f} "
            f"finite={bool(torch.isfinite(out).all())} max_abs_diff_vs_committed={diff} "
            f"bit_identical={torch.equal(out, outs['real'])} path(op,tap_major,lanes_zeroed,bk,bm)={paths[name]}"
        )
    for name, out in outs.items():
        if name == "naive":
            continue  # negative control: expected to pick up the poisoned lanes
        assert torch.isfinite(out).all(), name
        pcc, rel_rms = _metrics(ref, out)
        assert pcc >= 0.99, (name, pcc)


@pytest.mark.parametrize("poison", [False, True], ids=["clean", "poison"])
@pytest.mark.parametrize("case_id", list(CASES))
def test_sstm_naive_probe(case_id, poison, device, monkeypatch):
    """Negative-control probe: does the short slice's padding reach the slab when its lanes are not zeroed? Real
    input channels are zero and the padded lanes' weight rows are 1, so the output is exactly the padding's sum."""
    import torch

    pd = importlib.import_module(_COPY + ".conv2d_3x3_program_descriptor")
    torch_input, torch_weight, _ = _tensors(case_id)
    torch_input = torch.zeros_like(torch_input)
    monkeypatch.setattr(pd, "SHORT_SLICE_TAP_MAJOR", True)
    for mode in ("naive", "on", "naive"):
        monkeypatch.setattr(pd, "SHORT_SLICE_NAIVE", mode == "naive")
        ((H, W, c_in), c_out, p, s, d), dtype, layout, shard, grid, bias_relu, fidelity, bm = CASES[case_id]
        monkeypatch.setitem(
            CASES, case_id, (((H, W, c_in), c_out, p, s, d), dtype, layout, shard, grid, False, fidelity, bm)
        )
        out = (
            _run("copy", case_id, torch_input, torch_weight, None, device, monkeypatch, poison=True) if poison else None
        )
        if not poison:
            pkg, _ = _op_modules("copy")
            memory_config = _memory_config(shard, grid)
            kwargs = dict(
                stride=s,
                padding=p,
                dilation=d,
                input_dtype=dtype,
                input_layout=layout,
                weight_dtype=BF8,
                input_memory_config=memory_config,
                device=device,
            )
            tt_weight = _lane_poisoned_weights(pkg, torch_weight, kwargs)
            tt_input = ttnn.from_torch(
                torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config
            )
            tt_out = pkg.conv2d_3x3(
                tt_input,
                tt_weight,
                input_height=H,
                input_width=W,
                padding=p,
                stride=s,
                dilation=d,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            )
            out = ttnn.to_torch(tt_out).float()
        print(
            f"\nSSTM_PROBE case={case_id} poison={poison} mode={mode} path={PATHS[-1] if PATHS else None} "
            f"max_abs={out.abs().max().item()} nonzero={int((out != 0).sum())} finite={bool(torch.isfinite(out).all())}"
        )
