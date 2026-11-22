# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.pair_pool: the softmax-gated pooling of row pairs in one program (deepseek_v41_flash_d_p P.24,
TINY_KERNELS; the compressor's ratio-2 pooling, reference compressor kv * softmax(gate) summed over each pair).

    x [.., R, 2 hd] fp32 TILE = [kv | gate] per row  ->  out [.., R / 2, hd] fp32 TILE
    out[t] = kv[2t] w0 + kv[2t + 1] w1,  (w0, w1) = softmax(gate[2t], gate[2t + 1]) per column

The op chain it replaces (tt/compressor.py TtCompressor._pool: reshape, 4 slices, maximum, 2 subtract, 2 exp, add,
2 divide, 2 multiply, add) in the same fp32 steps with the stock ops' SFPU bodies (max; subtract; the fp32 accurate
exp; add; the fp32 SFPU divide; multiply; add), each step one fp32 rounding. Units (output tile-row, column tile) split
over the grid by common runtime args only (no per-core args: host dispatch on a 32-chip mesh, DS41-145); always
dispatched as a prepared program (ttnn.bringup.prepare_program / run_prepared, the two addresses patched; OV-23).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import ttnn

KERNELS = Path(__file__).parent / "kernels"
CB_STAGE, CB_IN, CB_OUT = 0, 1, 16
NUM_CB_SLOTS = 64
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
    s = t.padded_shape
    r = 1
    for i in range(len(s) - 1):
        r *= s[i]
    return r


def plan(x: ttnn.Tensor, max_grid=None) -> tuple[int, int, int, int]:
    """(units, cores, column tiles HT, grid width) of the call."""
    ht = x.shape[-1] // 2 // 32
    units = _rows(x) // 64 * ht
    gx, gy = _grid(x.device(), max_grid)
    return units, max(1, min(units, gx * gy)), ht, gx


def program(x: ttnn.Tensor, out: ttnn.Tensor, max_grid=None) -> ttnn.ProgramDescriptor:
    units, ncores, ht, gx = plan(x, max_grid)
    cores = [(i % gx, i // gx) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    crs = crs.merge(crs)  # (merged rectangles: whole grid rows + the rest of the last row)
    per, extra = divmod(units, ncores)
    page = ttnn.tile_size(ttnn.float32)

    def cb(idx, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=page)],
        )

    cbs = [cb(CB_STAGE, 4), cb(CB_IN, 8), cb(CB_OUT, 2)]  # (cb_in double-buffered: the next unit's shuffle overlaps)
    gxd = [("PAIR_POOL_GX", str(gx))]
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, dst_full_sync_en=True
    )
    modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    modes[CB_IN] = ttnn.UnpackToDestMode.UnpackToDestFp32  # exact fp32 copies into DEST
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "pair_pool_reader.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_STAGE, CB_IN, ht] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()),
            defines=gxd,
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[x.buffer_address(), per, extra],
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "pair_pool_writer.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_OUT, ht] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            defines=gxd,
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[out.buffer_address(), per, extra],
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "pair_pool_compute.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_IN, CB_OUT],
            defines=gxd,
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[per, extra],
            config=cfg,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _prepared(x: ttnn.Tensor, out: ttnn.Tensor, max_grid=None):
    """The program for this (mesh, shape, memory configs, grid), built and prepared once (x / out addresses patched)."""
    mesh = x.device()
    key = (
        mesh.id(),  # (MeshDevice::id: unique per opened mesh)
        tuple(x.padded_shape[i] for i in range(len(x.padded_shape))),
        str(x.memory_config()),
        str(out.memory_config()),
        _mg_key(max_grid),
    )
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = program(x, out, max_grid)
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    pp = ttnn.bringup.prepare_program(md, [x, out])
    # the two common address slots (reader: x, writer: out); anything else matched a frozen value
    assert pp.patch_count == 2, f"pair_pool: {pp.patch_count} patched slots (a frozen arg equals an io address)"
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = (mesh, pp)
    return pp


def pair_pool(
    x: ttnn.Tensor, *, output_tensor: ttnn.Tensor | None = None, memory_config=None, max_grid=None
) -> ttnn.Tensor:
    """x [.., R, 2 hd] float32 TILE DRAM interleaved (R % 64 == 0, hd % 32 == 0; columns [kv | gate]) ->
    [.., R / 2, hd] float32 TILE: out[t] = kv[2t] w0 + kv[2t+1] w1, (w0, w1) = softmax(gate[2t], gate[2t+1]) per
    column. On a mesh, the same program on every chip. ``output_tensor`` (default None = allocate): the caller's buffer
    of exactly that spec, returned as is. ``max_grid`` (default None = the whole compute grid): a ttnn.CoreCoord or
    (x, y); the cores are taken from the rectangle (0, 0) .. (max_grid - 1) (a Galaxy's 110-core cap)."""
    assert x.dtype == ttnn.float32 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
    mc = x.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    w = x.shape[-1]
    assert w % 64 == 0, f"pair_pool: last dim {w} must be 2 hd with hd % 32 == 0"
    assert x.shape[-2] % 64 == 0 and x.padded_shape[-2] == x.shape[-2], f"pair_pool: rows {x.shape[-2]} % 64 != 0"
    shape = list(x.shape)
    shape[-2] //= 2
    shape[-1] //= 2
    if output_tensor is None:
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), ttnn.float32, ttnn.TILE_LAYOUT, x.device(), memory_config or ttnn.DRAM_MEMORY_CONFIG
        )
    else:
        o = output_tensor
        assert o.dtype == ttnn.float32 and o.layout == ttnn.TILE_LAYOUT and list(o.shape) == shape, (o.dtype, o.shape)
        omc = o.memory_config()
        assert omc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, omc
    pp = _prepared(x, output_tensor, max_grid)
    # (looked up on ttnn.bringup at each call: the bring-up capture wraps run_prepared, F201)
    ttnn.bringup.run_prepared(pp, [x, output_tensor])
    return output_tensor
