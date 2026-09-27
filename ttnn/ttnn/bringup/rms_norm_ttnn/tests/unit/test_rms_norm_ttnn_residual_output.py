# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""`return_residual_sum`: ttnn.bringup.rms_norm also returns t = x + r, the sum it normalizes.

    y, t = ttnn.bringup.rms_norm(x, residual_input_tensor=r, weight=w, return_residual_sum=True)

In a transformer t is the next residual add's input, so this one call replaces `ttnn.add` + the norm.  What is checked:

  * y against the torch reference (torch_rms_norm_ttnn), and BIT-IDENTICAL to the same call with the option off: the
    option only adds a write, it never changes y;
  * t BIT-IDENTICAL to `ttnn.add(x, r)` on the same device for bf16 and bf8b, i.e. to what the model runs today.  It is
    NOT bit-exact against torch's bf16 add: both are one FPU add of the same two bf16 operands, but the FPU rounds the
    sum its own way (mostly ties away from zero, where torch rounds to nearest even), so about 11% of elements of a
    randn pair sit one bf16 ulp from torch.  So t is checked against torch to <= 1 ulp, and against ttnn.add exactly.
    fp32 inputs: t is the op's own FPU sum (tf32 operands at fp32_dest_acc_en=True, a 16-bit DEST otherwise) -- the same
    t its statistics use -- so it is checked against torch to that precision, not to an fp32 add;
  * every scheme the TILE op has: the row split (RESIDENT, ROW_RESIDENT, STREAM, the ragged chunk), the interleaved
    width split, HEIGHT / WIDTH / BLOCK shards (identity and compact combine, the slot tree), t placed elsewhere than
    the input, `inplace`, masked widths, ranks 1 and 5, a zero-volume input;
  * the refusals, the return type with the option off, and a program-cache hit on fresh buffers.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

from eval.sharding import auto_shard_config, shard_config
from ttnn.bringup.rms_norm_ttnn import torch_rms_norm_ttnn
from ttnn.operations._op_contract import UnsupportedAxisValue

rms_norm = ttnn.bringup.rms_norm
_ML = ttnn.TensorMemoryLayout
_TILE = ttnn.TILE_LAYOUT
EPS = 1e-6

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def _cfg(fp32=False):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _bits(t):
    t = t.contiguous()
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def _ulp_bf16(t):
    """One bf16 ulp at each element's magnitude (2^(exponent - 7))."""
    a = t.float().abs().clamp_min(torch.finfo(torch.bfloat16).tiny)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


# ---------------------------------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------------------------------
# (id, shape, memory placement, shard, mode, extras)
#   placement: INTERLEAVED (DRAM) or a *_SHARDED layout; shard: None (auto) or (shard_shape, grid)
#   mode: "g" / "b" present (the residual always is)
#   extras: dtype, fp32 (fp32_dest_acc_en), t_mc ("dram" / "l1" / "height": t's own placement), inplace
CASES = [
    # MiMo-V2.6's call (tests/cases.py): [1,1,5120,4096] bf16, weight, HiFi4 / fp32 DEST, eps 1e-6.  ROW_RESIDENT.
    ("mimo_5120x4096_g", (1, 1, 5120, 4096), _ML.INTERLEAVED, None, "g", {"fp32": True}),
    ("int_64x128", (1, 1, 64, 128), _ML.INTERLEAVED, None, "", {}),
    ("int_64x128_gb", (1, 1, 64, 128), _ML.INTERLEAVED, None, "gb", {}),
    ("int_100x200_gb_masked", (1, 1, 100, 200), _ML.INTERLEAVED, None, "gb", {}),
    ("int_2x3x50x96_g_masked", (2, 3, 50, 96), _ML.INTERLEAVED, None, "g", {"fp32": True}),
    ("int_8192x5120_g_row_resident", (1, 1, 8192, 5120), _ML.INTERLEAVED, None, "g", {}),
    ("int_8192x7168_gb_compact_pc", (1, 1, 8192, 7168), _ML.INTERLEAVED, None, "gb", {}),
    ("int_3104x4064_g_ragged_chunk", (1, 1, 3104, 4064), _ML.INTERLEAVED, None, "g", {}),
    ("int_1024x16384_gb_stream", (1, 1, 1024, 16384), _ML.INTERLEAVED, None, "gb", {}),
    ("int_32x7168_g_wsplit", (1, 1, 32, 7168), _ML.INTERLEAVED, None, "g", {"fp32": True}),
    ("int_32x1024_gb_wsplit", (1, 1, 32, 1024), _ML.INTERLEAVED, None, "gb", {}),
    ("int_32x16384_g_wsplit_wide", (1, 1, 32, 16384), _ML.INTERLEAVED, None, "g", {}),
    ("int_256x1024_g_fp32", (1, 1, 256, 1024), _ML.INTERLEAVED, None, "g", {"dtype": ttnn.float32}),
    (
        "int_256x1024_g_fp32_fp32acc",
        (1, 1, 256, 1024),
        _ML.INTERLEAVED,
        None,
        "g",
        {"dtype": ttnn.float32, "fp32": True},
    ),
    ("int_256x1024_gb_bfp8", (1, 1, 256, 1024), _ML.INTERLEAVED, None, "gb", {"dtype": ttnn.bfloat8_b}),
    ("int_rank5_g", (2, 1, 2, 64, 256), _ML.INTERLEAVED, None, "g", {}),
    ("int_rank1", (96,), _ML.INTERLEAVED, None, "g", {}),
    ("int_256x512_g_t_l1", (1, 1, 256, 512), _ML.INTERLEAVED, None, "g", {"t_mc": "l1"}),
    ("int_256x512_g_t_height", (1, 1, 256, 512), _ML.INTERLEAVED, None, "g", {"t_mc": "height"}),
    ("h_256x512_gb", (1, 1, 256, 512), _ML.HEIGHT_SHARDED, None, "gb", {}),
    ("h_256x512_g_t_dram", (1, 1, 256, 512), _ML.HEIGHT_SHARDED, None, "g", {"t_mc": "dram"}),
    ("h_384x768_gb_inplace", (1, 1, 384, 768), _ML.HEIGHT_SHARDED, None, "gb", {"inplace": True}),
    ("w_32x1024_g", (1, 1, 32, 1024), _ML.WIDTH_SHARDED, ([32, 128], (8, 1)), "g", {}),
    ("w_32x5120_gb_tree", (1, 1, 32, 5120), _ML.WIDTH_SHARDED, ([32, 160], (8, 4)), "gb", {"fp32": True}),
    ("w_1024x512_g_compact", (1, 1, 1024, 512), _ML.WIDTH_SHARDED, ([1024, 128], (4, 1)), "g", {}),
    ("w_32x200_g_ragged_masked", (1, 1, 32, 200), _ML.WIDTH_SHARDED, None, "g", {}),
    ("blk_8192x1024_g", (1, 1, 8192, 1024), _ML.BLOCK_SHARDED, ([1024, 128], (8, 8)), "g", {}),
    ("blk_7168x1024_gb_t_dram", (1, 1, 7168, 1024), _ML.BLOCK_SHARDED, None, "gb", {"t_mc": "dram"}),
]


def _memory_config(shape, memory_layout, dtype, device, shard):
    if memory_layout == _ML.INTERLEAVED:
        return ttnn.DRAM_MEMORY_CONFIG
    if shard is not None:
        return shard_config(shard[0], shard[1], memory_layout, layout=_TILE, dtype=dtype, device=device)
    return auto_shard_config(list(shape), memory_layout, layout=_TILE, dtype=dtype, device=device)


def _t_memory_config(shape, dtype, device, which):
    if which is None:
        return None
    if which == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if which == "l1":
        return ttnn.L1_MEMORY_CONFIG
    assert which == "height"
    return auto_shard_config(list(shape), _ML.HEIGHT_SHARDED, layout=_TILE, dtype=dtype, device=device)


def _torch_dtype(dtype):
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


def _inputs(device, case, seed=0):
    _id, shape, memory_layout, shard, mode, extras = case
    dtype = extras.get("dtype", ttnn.bfloat16)
    mc = _memory_config(shape, memory_layout, dtype, device, shard)
    g = torch.Generator().manual_seed(seed)
    width = shape[-1]
    x = torch.randn(shape, generator=g).to(_torch_dtype(dtype))
    r = torch.randn(shape, generator=g).to(_torch_dtype(dtype))
    w = (1.0 + 0.5 * torch.randn((1, 1, 1, width), generator=g)).bfloat16() if "g" in mode else None
    b = torch.randn((1, 1, 1, width), generator=g).bfloat16() if "b" in mode else None
    if dtype == ttnn.bfloat8_b:  # the device sees bf8b; the references must see the same values
        x = ttnn.to_torch(ttnn.from_torch(x, dtype=dtype, layout=_TILE))
        r = ttnn.to_torch(ttnn.from_torch(r, dtype=dtype, layout=_TILE))

    def dev(t, dt, memory_config):
        return ttnn.from_torch(t, dtype=dt, layout=_TILE, device=device, memory_config=memory_config)

    tt = dict(
        x=dev(x, dtype, mc),
        r=dev(r, dtype, mc),
        w=dev(w, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG) if w is not None else None,
        b=dev(b, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG) if b is not None else None,
    )
    return dict(x=x, r=r, w=w, b=b), tt, mc


def _program_config(x):
    from ttnn.bringup.rms_norm_ttnn import RMSNormShardedMultiCoreProgramConfig

    spec = x.memory_config().shard_spec
    bbox = spec.grid.bounding_box()
    return RMSNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(bbox.end.x - bbox.start.x + 1, bbox.end.y - bbox.start.y + 1),
        subblock_w=1,
        block_h=spec.shape[0] // 32,
        block_w=spec.shape[1] // 32,
        inplace=True,
    )


def _call(tt, case, mc, *, on, t_mc=None):
    _id, shape, memory_layout, shard, mode, extras = case
    kwargs = dict(
        epsilon=EPS,
        weight=tt["w"],
        bias=tt["b"],
        residual_input_tensor=tt["r"],
        compute_kernel_config=_cfg(extras.get("fp32", False)),
    )
    if memory_layout != _ML.INTERLEAVED:
        kwargs["memory_config"] = mc
    if extras.get("inplace"):
        kwargs["program_config"] = _program_config(tt["x"])
    if on:
        kwargs["return_residual_sum"] = True
        if t_mc is not None:
            kwargs["residual_sum_memory_config"] = t_mc
    return rms_norm(tt["x"], **kwargs)


def _check_t(t, ref, tt, case, device):
    """t against the device's own add (bit-exact at bf16 / bf8b) and against torch."""
    _id, shape, memory_layout, shard, mode, extras = case
    dtype = extras.get("dtype", ttnn.bfloat16)
    x, r = ref["x"], ref["r"]
    assert t.shape == x.shape and t.dtype == x.dtype
    s = x.float() + r.float()
    if dtype == ttnn.float32:
        # The op's FPU sum.  fp32 DEST: tf32 operands (10 mantissa bits), an fp32 sum -> within 2^-9 of the operands'
        # magnitudes.  16-bit DEST: bf16 operands and a bf16 sum -> within 2^-6 (measured up to 2^-6.6).
        rel = 2.0**-9 if extras.get("fp32", False) else 2.0**-6
        err = (t - s).abs()
        bound = rel * (x.abs() + r.abs()) + 1e-30
        assert bool((err <= bound).all()), f"fp32 t off by {(err / bound).max().item():.3f}x its precision bound"
        return
    # bf16 / bf8b: ttnn.add of the same two tensors, from fresh interleaved copies (any placement; `inplace` has
    # overwritten the op's own x by now).
    xd = ttnn.from_torch(x, dtype=dtype, layout=_TILE, device=device)
    rd = ttnn.from_torch(r, dtype=dtype, layout=_TILE, device=device)
    added = ttnn.to_torch(ttnn.add(xd, rd, dtype=dtype))
    assert torch.equal(
        _bits(t), _bits(added)
    ), f"t differs from ttnn.add in {(t != added).sum().item()} of {t.numel()} elements"
    if dtype == ttnn.bfloat16:
        err = (t.float() - s.bfloat16().float()).abs()
        assert bool((err <= _ulp_bf16(s)).all()), "t is more than one bf16 ulp from torch's bf16 add"


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_residual_sum(device, case):
    _id, shape, memory_layout, shard, mode, extras = case
    dtype = extras.get("dtype", ttnn.bfloat16)
    ref, tt, mc = _inputs(device, case)
    t_mc = _t_memory_config(shape, dtype, device, extras.get("t_mc"))

    # Option off first (inplace overwrites x, so the off call uses its own copy of the inputs).
    if extras.get("inplace"):
        _, tt_off, _ = _inputs(device, case)
        y_off = ttnn.to_torch(_call(tt_off, case, mc, on=False))
    else:
        y_off = ttnn.to_torch(_call(tt, case, mc, on=False))

    res = _call(tt, case, mc, on=True, t_mc=t_mc)
    assert isinstance(res, tuple) and len(res) == 2, f"expected (y, t), got {type(res)}"
    y_dev, t_dev = res
    if extras.get("inplace"):
        assert y_dev is tt["x"], "program_config.inplace must still return the input tensor object itself"
    expected_t_mc = t_mc if t_mc is not None else tt["x"].memory_config()
    assert t_dev.memory_config() == expected_t_mc, f"t placed at {t_dev.memory_config()}, expected {expected_t_mc}"
    assert t_dev.layout == _TILE and t_dev.dtype == dtype
    y, t = ttnn.to_torch(y_dev), ttnn.to_torch(t_dev)

    y_ref, t_ref = torch_rms_norm_ttnn(
        ref["x"],
        epsilon=EPS,
        weight=ref["w"],
        bias=ref["b"],
        residual_input_tensor=ref["r"],
        return_residual_sum=True,
    )
    assert torch.equal(_bits(y), _bits(y_off)), "the option changed y"
    assert _pcc(y, y_ref) >= PCC[dtype], f"y pcc {_pcc(y, y_ref)}"
    assert _pcc(t, t_ref) >= 0.9999
    _check_t(t, ref, tt, case, device)


def test_zero_volume(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 0, 64).bfloat16(), layout=_TILE, device=device)
    r = ttnn.from_torch(torch.zeros(1, 1, 0, 64).bfloat16(), layout=_TILE, device=device)
    y, t = rms_norm(x, epsilon=EPS, residual_input_tensor=r, return_residual_sum=True)
    assert list(y.shape) == [1, 1, 0, 64] and list(t.shape) == [1, 1, 0, 64]


def test_off_returns_a_tensor(device):
    """With the option off (default or explicit False) the return type is exactly today's: one ttnn.Tensor."""
    x = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=_TILE, device=device)
    r = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=_TILE, device=device)
    assert isinstance(rms_norm(x, residual_input_tensor=r), ttnn.Tensor)
    assert isinstance(rms_norm(x, residual_input_tensor=r, return_residual_sum=False), ttnn.Tensor)
    assert isinstance(rms_norm(x), ttnn.Tensor)


# ---------------------------------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------------------------------


def test_refuses_without_residual(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=_TILE, device=device)
    with expect_error(ValueError, "return_residual_sum=True needs residual_input_tensor"):
        rms_norm(x, return_residual_sum=True)


def test_refuses_memory_config_without_the_option(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=_TILE, device=device)
    r = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=_TILE, device=device)
    with expect_error(ValueError, "residual_sum_memory_config was given without return_residual_sum"):
        rms_norm(x, residual_input_tensor=r, residual_sum_memory_config=ttnn.L1_MEMORY_CONFIG)


def test_refuses_row_major(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    r = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(UnsupportedAxisValue, "supports layout \\[Layout.TILE\\] only"):
        rms_norm(x, residual_input_tensor=r, return_residual_sum=True)
    assert issubclass(UnsupportedAxisValue, NotImplementedError)


def test_refuses_sharded_memory_config_without_shard_spec(device, expect_error):
    x = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=_TILE, device=device)
    r = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=_TILE, device=device)
    with expect_error(ValueError, "carries no shard spec"):
        rms_norm(
            x,
            residual_input_tensor=r,
            return_residual_sum=True,
            residual_sum_memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )


def test_usual_refusals_come_first(device, expect_error):
    """An input the op refuses anyway is refused the usual way, option or not (the option's checks run after)."""
    x = ttnn.from_torch(torch.randn(1, 1, 64, 128).bfloat16(), layout=_TILE, device=device)
    r = ttnn.from_torch(torch.randn(1, 1, 64, 256).bfloat16(), layout=_TILE, device=device)
    with expect_error(ValueError, "residual_input_tensor shape"):
        rms_norm(x, residual_input_tensor=r, return_residual_sum=True)


# ---------------------------------------------------------------------------------------------------
# the program cache
# ---------------------------------------------------------------------------------------------------


def test_cache_hit_at_new_addresses(device):
    """The second call is a program-cache HIT (no new program) on buffers at new addresses, incl. t's -- so
    override_runtime_arguments patched t's address -- and its results are right for its own data.  The option is in
    the key: an option-off call with the same tensors builds its own program."""
    case = ("int_8192x1024_g", (1, 1, 8192, 1024), _ML.INTERLEAVED, None, "g", {})
    addresses, entries = [], []
    keep = []
    for seed in (0, 1):
        if seed == 1:  # push the next round's tensors (and t) to different DRAM addresses
            keep.append(ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32 * 7]), ttnn.bfloat16, _TILE, device))
        ref, tt, mc = _inputs(device, case, seed=seed)
        keep.append(tt)
        before = device.num_program_cache_entries()
        y_dev, t_dev = _call(tt, case, mc, on=True)
        entries.append(device.num_program_cache_entries() - before)
        addresses.append((tt["x"].buffer_address(), t_dev.buffer_address()))
        keep.append(t_dev)
        y_ref, t_ref = torch_rms_norm_ttnn(
            ref["x"], epsilon=EPS, weight=ref["w"], residual_input_tensor=ref["r"], return_residual_sum=True
        )
        t = ttnn.to_torch(t_dev)
        assert _pcc(ttnn.to_torch(y_dev), y_ref) >= 0.995
        _check_t(t, ref, tt, case, device)
    assert entries[1] == 0, "the second call missed the program cache"
    assert addresses[0][1] != addresses[1][1], "t did not move, so the hit did not exercise the address patch"
    assert addresses[0][0] != addresses[1][0]
    before = device.num_program_cache_entries()
    _call(tt, case, mc, on=False)
    assert device.num_program_cache_entries() - before == 1, "the option is not in the program-cache key"


# ---------------------------------------------------------------------------------------------------
# the program with the option on (host only)
# ---------------------------------------------------------------------------------------------------


def test_program_adds_only_the_t_path(device):
    """On vs off on the same tensors: the option adds cb_residual_sum (index 26), the RMS_RESIDUAL_OUT define on the
    writer and compute, the writer's t block + t accessor CT args and its one common runtime arg (t's address) --
    and changes nothing else: every other CB, arg, define and semaphore is the option-off program's."""
    cpp = ttnn._ttnn.operations.bringup._rms_norm_ttnn_program_descriptor
    shape = (1, 1, 5120, 4096)

    def alloc():
        return ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, _TILE, device)

    x, r, out, t = alloc(), alloc(), alloc(), alloc()
    w = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 1, 4096]), ttnn.bfloat16, _TILE, device)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    off = cpp(x, out, weight=w, residual=r, epsilon=EPS, compute_kernel_config=cfg)
    on = cpp(x, out, weight=w, residual=r, epsilon=EPS, compute_kernel_config=cfg, residual_sum=t)

    cbs_off = {cb.format_descriptors[0].buffer_index: cb for cb in off.cbs}
    cbs_on = {cb.format_descriptors[0].buffer_index: cb for cb in on.cbs}
    assert set(cbs_on) - set(cbs_off) == {26}
    for idx, cb in cbs_off.items():
        assert cbs_on[idx].total_size == cb.total_size, f"CB {idx} changed size"
    t_blk = list(on.kernels[1].compile_time_args)[len(off.kernels[1].compile_time_args)]
    assert 1 <= t_blk <= 4, "t's DEST block is pass B's, at most 4 tiles at fp32 DEST"
    assert cbs_on[26].total_size == 2 * t_blk * 2048, "cb_residual_sum: two DEST blocks of bf16 tiles"

    names = ("reader", "writer", "compute")
    for name, k_off, k_on in zip(names, off.kernels, on.kernels):
        d_off = sorted((str(a), str(b)) for a, b in k_off.defines)
        d_on = sorted((str(a), str(b)) for a, b in k_on.defines)
        if name == "reader":
            assert d_on == d_off and list(k_on.compile_time_args) == list(k_off.compile_time_args)
            assert list(k_on.common_runtime_args) == []
        else:
            extra = [d for d in d_on if d not in d_off]
            assert [e[0] for e in extra] == ["RMS_RESIDUAL_OUT"], f"{name}: extra defines {extra}"
        ct_off, ct_on = list(k_off.compile_time_args), list(k_on.compile_time_args)
        if name == "writer":
            assert ct_on[: len(ct_off)] == ct_off, "writer: the option-off CT args must be a prefix"
            assert ct_on[len(ct_off)] == t_blk
            assert list(k_on.common_runtime_args) == [t.buffer_address()]
        else:
            assert ct_on == ct_off, f"{name}: CT args changed"
        for x_, y_ in ((c.x, c.y) for c in ttnn.corerange_to_cores(k_off.core_ranges, None, True)):
            assert list(k_on.runtime_args[x_][y_]) == list(k_off.runtime_args[x_][y_]), f"{name}: RT args changed"
    assert [s.id for s in on.semaphores] == [s.id for s in off.semaphores]
