# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.pack_fp8_rows: per-token fp8 rows -> Blaze decode's 192-byte indexer key cache rows
(deepseek_v41_flash_d_p P.39, INDEX_K_SERVED=fp8_row192).

    e4m3  [.., M, 128] FP8_E4M3 ROW_MAJOR  and  scale [.., M, 1] FLOAT32 ROW_MAJOR
          (ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(x, round_scale_to_power_of_two=False) of a 128-wide x)
    ->    out [.., M, 192] UINT8 ROW_MAJOR, row m = [e4m3 row m (128 B) | scale m as fp32 little endian (4 B) | 60 zero B]

The layout of tt-blaze's DsaIndexerDirectKCacheUpdate (blaze/ops/dsa_indexer_direct_k_cache_update, SCALED_ROW_BYTES
192; tests/blaze/micro_ops/dsa/test_dsa_indexer_direct_k_cache_update.py checks bytes 0..131 against
per_token_cast_to_fp8). A row whose 128 values are all +-0 (a zeroed pad row) is written all zero: the stock packer of
the GLM rows (ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache: 512 B latent | 4 scales | 64 bf16 RoPE)
copies the clamp-floor scale (1e-4 / 448) there. A data-movement program only (one kernel, NoC 0): rows split over
the cores by common runtime args (no per-core args, DS41-145); always dispatched as a prepared program
(ttnn.bringup.prepare_program / run_prepared: the three addresses patched, OV-23).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import ttnn

KERNELS = Path(__file__).parent / "kernels"
VALUES, ROW_BYTES = 128, 192
CB_STAGE = 0
ROWS_BLOCK = 16  # rows staged per read / write batch (16 x 192 B = 3 KiB of L1 a core)
_PREPARED_CAP = 64
_PREPARED = OrderedDict()  # key -> (mesh device, PreparedProgram), LRU


def _grid(device, max_grid=None) -> tuple[int, int]:
    """The compute grid the op uses: the device's, cut to ``max_grid`` (a Galaxy's 110-core cap, agent rule 18)."""
    g = device.compute_with_storage_grid_size()
    if max_grid is None:
        return g.x, g.y
    mx, my = (max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid)
    return max(1, min(g.x, mx)), max(1, min(g.y, my))


def _mg_key(max_grid):
    if max_grid is None:
        return None
    return (max_grid.x, max_grid.y) if hasattr(max_grid, "x") else tuple(max_grid)


def _rows(t: ttnn.Tensor) -> int:
    s = t.shape
    r = 1
    for i in range(len(s) - 1):
        r *= s[i]
    return r


def plan(e4m3: ttnn.Tensor, max_grid=None) -> tuple[int, int, int]:
    """(rows, cores, grid width) of the call: at least ROWS_BLOCK rows a core, at most the grid."""
    rows = _rows(e4m3)
    gx, gy = _grid(e4m3.device(), max_grid)
    cores = max(1, min(gx * gy, -(-rows // ROWS_BLOCK)))
    return rows, cores, gx


def program(e4m3: ttnn.Tensor, scale: ttnn.Tensor, out: ttnn.Tensor, max_grid=None) -> ttnn.ProgramDescriptor:
    rows, ncores, gx = plan(e4m3, max_grid)
    cores = [(i % gx, i // gx) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    crs = crs.merge(crs)
    per, extra = divmod(rows, ncores)
    stage = ROWS_BLOCK * ROW_BYTES + 64
    cbs = [
        ttnn.CBDescriptor(
            total_size=stage,
            core_ranges=crs,
            format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=CB_STAGE, data_format=ttnn.uint8, page_size=stage)
            ],
        )
    ]
    ct = [CB_STAGE, ROWS_BLOCK, gx]
    for t in (e4m3, scale, out):
        ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "pack_fp8_rows.cpp"),
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[e4m3.buffer_address(), scale.buffer_address(), out.buffer_address(), per, extra],
            config=ttnn.WriterConfigDescriptor(),
        )
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _prepared(e4m3: ttnn.Tensor, scale: ttnn.Tensor, out: ttnn.Tensor, max_grid=None):
    """The program for this (mesh, shape, memory configs, grid), built and prepared once (the 3 addresses patched)."""
    mesh = e4m3.device()
    key = (
        mesh.id(),
        tuple(e4m3.shape[i] for i in range(len(e4m3.shape))),
        str(e4m3.memory_config()),
        str(scale.memory_config()),
        str(out.memory_config()),
        _mg_key(max_grid),
    )
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = program(e4m3, scale, out, max_grid)
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    pp = ttnn.bringup.prepare_program(md, [e4m3, scale, out])
    assert pp.patch_count == 3, f"pack_fp8_rows: {pp.patch_count} patched slots (a frozen arg equals an io address)"
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = (mesh, pp)
    return pp


def _interleaved_dram(t: ttnn.Tensor, what: str) -> None:
    mc = t.memory_config()
    assert (
        mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"pack_fp8_rows: {what} must be DRAM interleaved, got {mc}"


def pack_fp8_rows(
    e4m3: ttnn.Tensor,
    scale: ttnn.Tensor,
    *,
    output_tensor: ttnn.Tensor | None = None,
    memory_config=None,
    max_grid=None,
) -> ttnn.Tensor:
    """e4m3 [.., M, 128] FP8_E4M3 ROW_MAJOR + scale [.., M, 1] FLOAT32 ROW_MAJOR (both DRAM interleaved, equal leading
    dims) -> [.., M, 192] UINT8 ROW_MAJOR rows [values | fp32 scale | zeros] (all zero where the values are all +-0).
    On a mesh, the same program on every chip. ``output_tensor``: the caller's buffer of exactly that spec (DRAM
    interleaved), returned as is. ``max_grid`` (default None = the whole compute grid): a ttnn.CoreCoord or (x, y);
    the cores are taken from the rectangle (0, 0) .. (max_grid - 1) (a Galaxy's 110-core cap)."""
    assert (
        e4m3.dtype == ttnn.fp8_e4m3 and e4m3.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"pack_fp8_rows: e4m3 must be FP8_E4M3 ROW_MAJOR, got {e4m3.dtype} {e4m3.layout}"
    assert (
        scale.dtype == ttnn.float32 and scale.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"pack_fp8_rows: scale must be FLOAT32 ROW_MAJOR, got {scale.dtype} {scale.layout}"
    assert (
        e4m3.shape[-1] == VALUES and scale.shape[-1] == 1
    ), f"pack_fp8_rows: values [.., M, {VALUES}] and one scale a row, got {e4m3.shape} / {scale.shape}"
    assert list(e4m3.shape)[:-1] == list(scale.shape)[:-1], (e4m3.shape, scale.shape)
    _interleaved_dram(e4m3, "e4m3")
    _interleaved_dram(scale, "scale")
    shape = list(e4m3.shape)
    shape[-1] = ROW_BYTES
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape),
            ttnn.uint8,
            ttnn.ROW_MAJOR_LAYOUT,
            e4m3.device(),
            memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        o = output_tensor
        assert (
            o.dtype == ttnn.uint8 and o.layout == ttnn.ROW_MAJOR_LAYOUT and list(o.shape) == shape
        ), f"pack_fp8_rows: output_tensor must be uint8 ROW_MAJOR {shape}, got {o.dtype} {o.layout} {list(o.shape)}"
    _interleaved_dram(output_tensor, "output")
    pp = _prepared(e4m3, scale, output_tensor, max_grid)
    ttnn.bringup.run_prepared(pp, [e4m3, scale, output_tensor])
    return output_tensor
