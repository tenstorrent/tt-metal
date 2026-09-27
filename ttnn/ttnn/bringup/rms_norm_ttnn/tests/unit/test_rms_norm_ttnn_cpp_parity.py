# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The C++ host side of ttnn.bringup.rms_norm against the Python builder it was ported from.

`ttnn.bringup.rms_norm` builds its program in C++ (device/rms_norm_ttnn_program_factory.cpp); the Python
builder (rms_norm_ttnn_program_descriptor.py) stays as the reference.  Two checks:

  1. PROGRAM PARITY, on the host, no dispatch.  Both builders run on the same tensors and the two
     ProgramDescriptors are compared field by field: every kernel's source file, core ranges, config
     (processor / NoC / compute config), defines, compile-time args and per-core runtime args; every
     CB's index, size, page size, data format, tile, cores and shard backing; every semaphore.  Over a
     set that reaches every scheme and regime the builder has: the row split (RESIDENT, D42's one-row
     block, ROW_RESIDENT tiled and compact, STREAM, the ragged chunk), the interleaved width split
     (1-D line and packed 2-D group), HEIGHT / WIDTH / BLOCK shards (identity and compact combine, the
     slot tree, the NoC swap), the ROW_MAJOR BAND, both layouts, masked widths, block-float and fp32,
     the per-channel broadcast, every operand combination, `subblock_w`, `inplace`, ranks 0/1/5 and a
     zero-volume input.
  2. OUTPUT PARITY on the device: the C++ op and the Python op give BIT-IDENTICAL outputs, and so does
     a second C++ call on fresh buffers (a program-cache hit, which only re-patches addresses).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn

from eval.sharding import auto_shard_config, shard_config
from ttnn.bringup.rms_norm_ttnn import rms_norm_ttnn as rms_norm_python
from ttnn.bringup.rms_norm_ttnn.rms_norm_ttnn import normalize_compute_kernel_config
from ttnn.bringup.rms_norm_ttnn.rms_norm_ttnn_program_descriptor import (
    ResolvedProgramConfig,
    create_program_descriptor as python_descriptor,
)

_ML = ttnn.TensorMemoryLayout
_TILE = ttnn.TILE_LAYOUT
_RM = ttnn.ROW_MAJOR_LAYOUT
_REPO = Path(__file__).resolve().parents[6]

cpp_descriptor = ttnn._ttnn.operations.bringup._rms_norm_ttnn_program_descriptor


def _cfg(fidelity=ttnn.MathFidelity.HiFi4, fp32=False, approx=True):
    return ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=fp32, math_approx_mode=approx)


# ---------------------------------------------------------------------------------------------------
# the case set
# ---------------------------------------------------------------------------------------------------
# (id, shape, layout, memory_layout, shard, mode, extras)
#   shard:  None (interleaved / auto shard) or (shard_shape, grid)
#   mode:   which of weight / bias / residual are present
#   extras: dtype, operand dtype / layout, blocked operand, compute config, subblock_w, inplace
CASES = [
    # ---- interleaved TILE, the row split ------------------------------------------------------
    ("int_64x128_none", (1, 1, 64, 128), _TILE, _ML.INTERLEAVED, None, "", {}),
    ("int_64x128_g", (1, 1, 64, 128), _TILE, _ML.INTERLEAVED, None, "g", {}),
    ("int_64x128_b", (1, 1, 64, 128), _TILE, _ML.INTERLEAVED, None, "b", {}),
    ("int_64x128_r", (1, 1, 64, 128), _TILE, _ML.INTERLEAVED, None, "r", {}),
    ("int_64x128_gbr", (1, 1, 64, 128), _TILE, _ML.INTERLEAVED, None, "gbr", {}),
    ("int_8192x1024_g", (1, 1, 8192, 1024), _TILE, _ML.INTERLEAVED, None, "g", {}),
    ("int_8192x2304_gb", (1, 1, 8192, 2304), _TILE, _ML.INTERLEAVED, None, "gb", {}),
    ("int_5120x4096_g_fp32acc", (1, 1, 5120, 4096), _TILE, _ML.INTERLEAVED, None, "g", {"cfg": _cfg(fp32=True)}),
    ("int_8192x5120_g", (1, 1, 8192, 5120), _TILE, _ML.INTERLEAVED, None, "g", {}),  # ROW_RESIDENT
    ("int_8192x7168_gbr_fp32acc", (1, 1, 8192, 7168), _TILE, _ML.INTERLEAVED, None, "gbr", {"cfg": _cfg(fp32=True)}),
    ("int_1024x16384_gbr", (1, 1, 1024, 16384), _TILE, _ML.INTERLEAVED, None, "gbr", {}),
    ("int_3104x4064_g_prime", (1, 1, 3104, 4064), _TILE, _ML.INTERLEAVED, None, "g", {}),  # ragged chunk
    ("int_32x50_g_masked", (1, 1, 32, 50), _TILE, _ML.INTERLEAVED, None, "g", {}),
    ("int_8192x1000_g_masked", (1, 1, 8192, 1000), _TILE, _ML.INTERLEAVED, None, "g", {}),
    ("int_4x3x100x200_gb_fp32", (4, 3, 100, 200), _TILE, _ML.INTERLEAVED, None, "gb", {"dtype": ttnn.float32}),
    ("int_256x1024_gb_bfp8", (1, 1, 256, 1024), _TILE, _ML.INTERLEAVED, None, "gb", {"dtype": ttnn.bfloat8_b}),
    ("int_rank5_gr", (2, 1, 2, 64, 256), _TILE, _ML.INTERLEAVED, None, "gr", {}),
    ("int_rank1", (96,), _TILE, _ML.INTERLEAVED, None, "g", {}),
    ("int_zero_volume", (1, 1, 0, 64), _TILE, _ML.INTERLEAVED, None, "g", {}),
    ("int_eps0_g", (1, 1, 64, 256), _TILE, _ML.INTERLEAVED, None, "g", {"epsilon": 0.0}),
    (
        "int_mixed_operands",
        (1, 1, 128, 512),
        _TILE,
        _ML.INTERLEAVED,
        None,
        "gb",
        {"operand_dtype": ttnn.float32, "operand_layout": _RM},
    ),
    ("int_device_cfg_hifi2", (1, 1, 64, 256), _TILE, _ML.INTERLEAVED, None, "g", {"device_cfg": True}),
    # ---- interleaved TILE, the width split ---------------------------------------------------
    ("int_32x7168_g_wsplit", (1, 1, 32, 7168), _TILE, _ML.INTERLEAVED, None, "g", {}),
    ("int_32x1024_gbr_wsplit", (1, 1, 32, 1024), _TILE, _ML.INTERLEAVED, None, "gbr", {}),
    ("int_128x4096_g_wsplit_rect", (1, 1, 128, 4096), _TILE, _ML.INTERLEAVED, None, "g", {}),
    ("int_32x16384_gbr_wide", (1, 1, 32, 16384), _TILE, _ML.INTERLEAVED, None, "gbr", {}),
    # ---- interleaved ROW_MAJOR ----------------------------------------------------------------
    ("rm_64x128_none", (1, 1, 64, 128), _RM, _ML.INTERLEAVED, None, "", {}),
    ("rm_64x128_gbr", (1, 1, 64, 128), _RM, _ML.INTERLEAVED, None, "gbr", {}),
    ("rm_32x50_g", (1, 1, 32, 50), _RM, _ML.INTERLEAVED, None, "g", {}),
    ("rm_100x200_g_blocked", (1, 1, 100, 200), _RM, _ML.INTERLEAVED, None, "gb", {"blocked": True}),
    ("rm_32x4064_gbr_prime", (1, 1, 32, 4064), _RM, _ML.INTERLEAVED, None, "gbr", {}),
    ("rm_128x4096_g_fp32acc", (1, 1, 128, 4096), _RM, _ML.INTERLEAVED, None, "g", {"cfg": _cfg(fp32=True)}),
    ("rm_64x128_g_tile_operand", (1, 1, 64, 128), _RM, _ML.INTERLEAVED, None, "g", {"operand_layout": _TILE}),
    ("rm_rank0", (), _RM, _ML.INTERLEAVED, None, "", {}),
    # ---- HEIGHT -------------------------------------------------------------------------------
    ("h_256x512_gbr", (1, 1, 256, 512), _TILE, _ML.HEIGHT_SHARDED, None, "gbr", {}),
    ("h_2048x256_g", (1, 1, 2048, 256), _TILE, _ML.HEIGHT_SHARDED, None, "g", {}),
    ("h_384x768_g_subblock2", (1, 1, 384, 768), _TILE, _ML.HEIGHT_SHARDED, None, "g", {"subblock_w": 2}),
    ("h_384x768_gb_inplace", (1, 1, 384, 768), _TILE, _ML.HEIGHT_SHARDED, None, "gb", {"inplace": True}),
    ("h_256x512_g_out_interleaved", (1, 1, 256, 512), _TILE, _ML.HEIGHT_SHARDED, None, "g", {"out_dram": True}),
    ("h_rm_256x512_g", (1, 1, 256, 512), _RM, _ML.HEIGHT_SHARDED, None, "g", {}),
    # ---- WIDTH --------------------------------------------------------------------------------
    ("w_32x1024_g", (1, 1, 32, 1024), _TILE, _ML.WIDTH_SHARDED, ([32, 128], (8, 1)), "g", {}),
    ("w_32x7168_g", (1, 1, 32, 7168), _TILE, _ML.WIDTH_SHARDED, ([32, 256], (7, 4)), "g", {}),
    (
        "w_32x5120_gbr_tree",
        (1, 1, 32, 5120),
        _TILE,
        _ML.WIDTH_SHARDED,
        ([32, 160], (8, 4)),
        "gbr",
        {"cfg": _cfg(fp32=True)},
    ),
    ("w_32x4800_g_tree_ragged", (1, 1, 32, 4800), _TILE, _ML.WIDTH_SHARDED, ([32, 160], (10, 3)), "g", {}),
    ("w_1024x512_g_compact", (1, 1, 1024, 512), _TILE, _ML.WIDTH_SHARDED, ([1024, 128], (4, 1)), "g", {}),
    (
        "w_1024x512_g_compact_eps0",
        (1, 1, 1024, 512),
        _TILE,
        _ML.WIDTH_SHARDED,
        ([1024, 128], (4, 1)),
        "g",
        {"epsilon": 0.0},
    ),
    ("w_32x200_g_ragged_masked", (1, 1, 32, 200), _TILE, _ML.WIDTH_SHARDED, None, "g", {}),
    ("w_auto_224x3072_gbr", (1, 1, 224, 3072), _TILE, _ML.WIDTH_SHARDED, None, "gbr", {}),
    # ---- BLOCK --------------------------------------------------------------------------------
    ("blk_8192x1024_g", (1, 1, 8192, 1024), _TILE, _ML.BLOCK_SHARDED, ([1024, 128], (8, 8)), "g", {}),
    ("blk_7168x1024_gbr", (1, 1, 7168, 1024), _TILE, _ML.BLOCK_SHARDED, None, "gbr", {}),
    ("blk_auto_256x512_g", (1, 1, 256, 512), _TILE, _ML.BLOCK_SHARDED, None, "g", {}),
    # ---- ROW_MAJOR BAND -----------------------------------------------------------------------
    ("band_blk_256x512_g", (1, 1, 256, 512), _RM, _ML.BLOCK_SHARDED, None, "g", {}),
    ("band_w_256x512_gbr", (1, 1, 256, 512), _RM, _ML.WIDTH_SHARDED, None, "gbr", {}),
    ("band_w_256x512_g_tile_operand", (1, 1, 256, 512), _RM, _ML.WIDTH_SHARDED, None, "g", {"operand_layout": _TILE}),
    ("band_blk_128x8192_gbr_fp32", (1, 1, 128, 8192), _RM, _ML.BLOCK_SHARDED, None, "gbr", {"dtype": ttnn.float32}),
]

_IDS = [c[0] for c in CASES]


def _torch_dtype(dtype):
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


def _memory_config(shape, memory_layout, layout, dtype, device, shard):
    if memory_layout == _ML.INTERLEAVED:
        return ttnn.DRAM_MEMORY_CONFIG
    if shard is not None:
        return shard_config(shard[0], shard[1], memory_layout, layout=layout, dtype=dtype, device=device)
    return auto_shard_config(list(shape), memory_layout, layout=layout, dtype=dtype, device=device)


def _build_inputs(device, shape, layout, memory_layout, shard, mode, extras, *, seed=0, with_data=False):
    """The op's tensors for one case.  `with_data` fills them with random values (for dispatch)."""
    dtype = extras.get("dtype", ttnn.bfloat16)
    operand_dtype = extras.get("operand_dtype", dtype)
    operand_layout = extras.get("operand_layout", layout)
    mc = _memory_config(shape, memory_layout, layout, dtype, device, shard)
    width = shape[-1] if len(shape) else 1
    torch.manual_seed(seed)

    def tensor(torch_shape, dt, lay, memory_config):
        if not with_data:  # a descriptor build reads no values: allocate only
            return ttnn.allocate_tensor_on_device(ttnn.Shape(list(torch_shape)), dt, lay, device, memory_config)
        data = torch.randn(torch_shape, dtype=torch.float32)
        return ttnn.from_torch(
            data.to(_torch_dtype(dt)), dtype=dt, layout=lay, device=device, memory_config=memory_config
        )

    def per_channel():
        if extras.get("blocked"):
            wt = (width + 31) // 32
            return tensor((wt, 32), operand_dtype, _RM, ttnn.DRAM_MEMORY_CONFIG)
        return tensor((1, 1, 1, width), operand_dtype, operand_layout, ttnn.DRAM_MEMORY_CONFIG)

    x = tensor(shape, dtype, layout, mc)
    w = per_channel() if "g" in mode else None
    b = per_channel() if "b" in mode else None
    r = tensor(shape, dtype, layout, mc) if "r" in mode else None
    if extras.get("out_dram"):
        out_mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        out_mc = mc
    return x, w, b, r, out_mc


def _compute_config(device, extras):
    if extras.get("device_cfg"):
        return ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True
        )
    return extras.get("cfg", _cfg())


# ---------------------------------------------------------------------------------------------------
# descriptor → comparable plain data
# ---------------------------------------------------------------------------------------------------


def _cores(crs):
    return sorted((c.x, c.y) for c in ttnn.corerange_to_cores(crs, None, True))


def _kernel_file(path: str) -> Path:
    p = Path(path)
    return (p if p.is_absolute() else _REPO / p).resolve()


def _config_of(cfg):
    if isinstance(cfg, ttnn.ComputeConfigDescriptor):
        return (
            "compute",
            str(cfg.math_fidelity),
            cfg.fp32_dest_acc_en,
            cfg.dst_full_sync_en,
            [str(m) for m in cfg.unpack_to_dest_mode],
            cfg.bfp8_pack_precise,
            cfg.math_approx_mode,
            cfg.enable_trisc2_rvv,
        )
    if isinstance(cfg, ttnn.DataMovementConfigDescriptor):
        # `.value`: the NOC enum's NOC_0 / NOC_1 are aliases that do not compare equal to themselves.
        return ("dm", str(cfg.processor), cfg.noc.value, str(cfg.noc_mode))
    return (type(cfg).__name__,)


def _kernel_of(k):
    rt = {}
    view = k.runtime_args
    for x, y in _cores(k.core_ranges):
        rt[(x, y)] = list(view[x][y])
    assert len(view) == len(rt), "runtime args set on a core outside the kernel's core ranges"
    return {
        "source": _kernel_file(k.kernel_source),
        "source_type": str(k.source_type),
        "cores": _cores(k.core_ranges),
        "config": _config_of(k.config),
        "defines": sorted((str(a), str(b)) for a, b in k.defines),
        "ct": list(k.compile_time_args),
        "named_ct": list(k.named_compile_time_args),
        "common_rt": list(k.common_runtime_args),
        "rt": rt,
    }


def _cb_of(cb):
    fds = []
    for fd in cb.format_descriptors:
        tile = None if fd.tile is None else (fd.tile.height, fd.tile.width, fd.tile.transpose)
        fds.append((fd.buffer_index, fd.data_format_as_uint8, fd.page_size, tile))
    return {
        "formats": fds,
        "total_size": cb.total_size,
        "cores": _cores(cb.core_ranges),
        "has_buffer": cb.has_buffer(),
        "buffer_address": cb.buffer_address(),
        "address_offset": cb.address_offset,
    }


def _describe(desc):
    return {
        "kernels": [_kernel_of(k) for k in desc.kernels],
        "cbs": {cb.format_descriptors[0].buffer_index: _cb_of(cb) for cb in desc.cbs},
        "cb_order": [cb.format_descriptors[0].buffer_index for cb in desc.cbs],
        "semaphores": [(s.id, str(s.core_type), _cores(s.core_ranges), s.initial_value) for s in desc.semaphores],
    }


def _assert_same(py, cpp):
    assert len(py["kernels"]) == len(cpp["kernels"]) == 3
    for name, a, b in zip(("reader", "writer", "compute"), py["kernels"], cpp["kernels"]):
        for field in a:
            if field == "rt":
                assert set(a["rt"]) == set(b["rt"]), f"{name}: runtime args on different cores"
                for core in a["rt"]:
                    assert a["rt"][core] == b["rt"][core], f"{name}: runtime args differ on core {core}"
            else:
                assert a[field] == b[field], f"{name}.{field}: python={a[field]} cpp={b[field]}"
    assert py["cb_order"] == cpp["cb_order"], f"CB set/order: python={py['cb_order']} cpp={cpp['cb_order']}"
    for idx in py["cbs"]:
        assert py["cbs"][idx] == cpp["cbs"][idx], f"CB {idx}: python={py['cbs'][idx]} cpp={cpp['cbs'][idx]}"
    assert py["semaphores"] == cpp["semaphores"], "semaphores differ"


def _output_tensor(device, x, out_mc, extras):
    if extras.get("inplace"):
        return x
    return ttnn.allocate_tensor_on_device(ttnn.Shape(list(x.shape)), x.dtype, x.layout, device, out_mc)


@pytest.mark.parametrize("case", CASES, ids=_IDS)
def test_program_descriptor_parity(device, case):
    _id, shape, layout, memory_layout, shard, mode, extras = case
    x, w, b, r, out_mc = _build_inputs(device, shape, layout, memory_layout, shard, mode, extras)
    out = _output_tensor(device, x, out_mc, extras)
    cfg = _compute_config(device, extras)
    eps = extras.get("epsilon", 1e-6)
    subblock_w = extras.get("subblock_w", 0)
    py = python_descriptor(
        x,
        out,
        weight=w,
        bias=b,
        residual=r,
        epsilon=eps,
        compute_kernel_config=normalize_compute_kernel_config(cfg),
        program_config=ResolvedProgramConfig(subblock_w=subblock_w, inplace=bool(extras.get("inplace"))),
    )
    cpp = cpp_descriptor(
        x,
        out,
        weight=w,
        bias=b,
        residual=r,
        epsilon=eps,
        compute_kernel_config=cfg,
        subblock_w=subblock_w,
    )
    _assert_same(_describe(py), _describe(cpp))


# ---------------------------------------------------------------------------------------------------
# device outputs, bit for bit
# ---------------------------------------------------------------------------------------------------

_DEVICE_CASES = [
    c
    for c in CASES
    if c[0]
    in (
        "int_64x128_gbr",
        "int_8192x1024_g",
        "int_32x7168_g_wsplit",
        "int_32x50_g_masked",
        "int_256x1024_gb_bfp8",
        "rm_100x200_g_blocked",
        "h_256x512_gbr",
        "h_384x768_gb_inplace",
        "w_32x7168_g",
        "w_1024x512_g_compact",
        "blk_8192x1024_g",
        "band_w_256x512_gbr",
        "int_zero_volume",
    )
]


def _program_config(x, extras):
    if not (extras.get("inplace") or extras.get("subblock_w")):
        return None
    from ttnn.bringup.rms_norm_ttnn import RMSNormShardedMultiCoreProgramConfig

    spec = x.memory_config().shard_spec
    bbox = spec.grid.bounding_box()
    return RMSNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(bbox.end.x - bbox.start.x + 1, bbox.end.y - bbox.start.y + 1),
        subblock_w=extras.get("subblock_w", 1),
        block_h=spec.shape[0] // 32,
        block_w=spec.shape[1] // 32,
        inplace=bool(extras.get("inplace")),
    )


def _run(op, device, case, seed):
    _id, shape, layout, memory_layout, shard, mode, extras = case
    x, w, b, r, out_mc = _build_inputs(
        device, shape, layout, memory_layout, shard, mode, extras, seed=seed, with_data=True
    )
    kwargs = dict(
        epsilon=1e-6, weight=w, bias=b, residual_input_tensor=r, compute_kernel_config=_compute_config(device, extras)
    )
    if memory_layout != _ML.INTERLEAVED or extras.get("out_dram"):
        kwargs["memory_config"] = out_mc
    pc = _program_config(x, extras)
    if pc is not None:
        kwargs["program_config"] = pc
    out = op(x, **kwargs)
    if extras.get("inplace"):
        assert out is x, "program_config.inplace must return the input tensor object itself"
    # The tensors' addresses are handed back so the caller can check that the next call's moved.
    return ttnn.to_torch(out), _addresses([x, w, b, r, out])


def _addresses(tensors):
    return [t.buffer_address() for t in tensors if t is not None and t.is_allocated() and t.volume()]


def _spacers(device):
    """Small allocations that push the next tensors to different DRAM and L1 addresses.  (Holding a whole
    round of L1 shards alive instead would crowd the CB region the op sized for its own tensors.)"""
    return [
        ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 32, 32 * n]), ttnn.bfloat16, _TILE, device, mc)
        for n, mc in ((7, ttnn.DRAM_MEMORY_CONFIG), (3, ttnn.L1_MEMORY_CONFIG))
    ]


@pytest.mark.parametrize("case", _DEVICE_CASES, ids=[c[0] for c in _DEVICE_CASES])
def test_device_output_is_bit_identical(device, case):
    cpp_addresses = []
    cache_entries = []
    spacers = []
    for seed in (0, 1):  # the second C++ call is a program-cache HIT on fresh buffers
        if seed == 1:
            spacers = _spacers(device)
        expected, _ = _run(rms_norm_python, device, case, seed)
        before = device.num_program_cache_entries()
        actual, addresses = _run(ttnn.bringup.rms_norm, device, case, seed)
        cpp_addresses.append(addresses)
        cache_entries.append(device.num_program_cache_entries() - before)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        if expected.numel():
            assert torch.equal(_bits(actual), _bits(expected)), (
                f"seed {seed}: C++ and Python outputs differ (max abs diff "
                f"{(actual.float() - expected.float()).abs().max().item()})"
            )
    # The second C++ call built no new program, and its tensors really did move, so the hit exercised
    # override_runtime_arguments.
    assert cache_entries[1] == 0, "the second C++ call missed the program cache"
    if cpp_addresses[0]:
        assert cpp_addresses[0] != cpp_addresses[1], "the second round reused the first round's buffers"
    del spacers


def _bits(t):
    """The raw bit pattern, so the comparison is exact (NaN == NaN, -0 != +0)."""
    t = t.contiguous()
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.int32)


def test_binding_is_the_cpp_op():
    """`ttnn.bringup.rms_norm` resolves to the C++ binding, not the Python registration."""
    op = ttnn.bringup.rms_norm
    assert op.is_cpp_operation
    assert op.python_fully_qualified_name == "ttnn.bringup.rms_norm"
