# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.block_gather / ttnn.bringup.block_keys: the two data-movement steps of an EXACT block pre-selected
top-k (deepseek_v41_flash_d_p PB.BM, INDEXER_BLOCK_SELECT): per row, the top-K blocks of B adjacent scores by block max
(a top-k over the block maxima), then the top-k keys over only those blocks' scores.

    block_gather   scores [.., R, W] bf16 RM, ids [.., R, K] uint32 RM (block ids, e.g. a top-k's output; >= W / B,
                   0xFFFFFFFF included: none) -> [.., R, K B] bf16 RM: row r = the K blocks' B scores each, in ids'
                   order (a missing block: B x -inf)
    block_keys     pos [.., R, KP] uint32 RM (positions in block_gather's row, e.g. a top-k of it; 0xFFFFFFFF: none),
                   ids [.., R, K] uint32 RM (the same ids) -> [.., R, KP] uint32 RM: ids[pos / B] B + pos % B (none:
                   0xFFFFFFFF)

Why not ttnn.gather: its ROW_MAJOR kernel holds each core's whole input row in L1 (a 1M-key row is 2 MiB) and reads it
once a core; here a block is one 64-byte DRAM read. One data-movement kernel each (NoC 0), rows split over the cores by
common runtime args (no per-core args, DS41-145); always dispatched as a prepared program (ttnn.bringup.prepare_program
/ run_prepared: the three addresses patched; the program depends on shapes only, never on a value).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import ttnn

KERNELS = Path(__file__).parent / "kernels"
CB = 0
GATHER_PARTS = 4  # block_gather: a row's blocks split into this many work units (both RISCs of every core)
_PREPARED_CAP = 64
_PREPARED = OrderedDict()  # key -> (mesh device, PreparedProgram), LRU
SENTINEL = 0xFFFFFFFF


def _grid(device, max_grid=None) -> tuple[int, int]:
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
    r = 1
    for i in range(len(t.shape) - 1):
        r *= t.shape[i]
    return r


def _dram_rm(t: ttnn.Tensor, what: str, dtype) -> None:
    mc = t.memory_config()
    assert (
        mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"block_select: {what} must be DRAM interleaved, got {mc}"
    assert t.layout == ttnn.ROW_MAJOR_LAYOUT and t.dtype == dtype, f"block_select: {what} {t.dtype} {t.layout}"


def _program(kernel: str, ios: list, ct_head: list, scratch: int, rows: int, max_grid) -> ttnn.ProgramDescriptor:
    gx, gy = _grid(ios[0].device(), max_grid)
    ncores = max(1, min(gx * gy, rows))
    cores = [(i % gx, i // gx) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    crs = crs.merge(crs)
    per, extra = divmod(rows, ncores)
    cbs = [
        ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB, data_format=ttnn.uint8, page_size=scratch)],
        )
    ]
    ct = [CB] + ct_head
    for t in ios:
        ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / kernel),
            core_ranges=crs,
            compile_time_args=ct,
            runtime_args=ttnn.RuntimeArgs(),
            common_runtime_args=[int(t.buffer_address()) for t in ios] + [per, extra],
            config=ttnn.ReaderConfigDescriptor(),
        )
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _prepared(kind: str, ios: list, params: tuple, make):
    """The program for (kind, mesh, shapes, memory configs, params), built and prepared once (3 addresses patched)."""
    mesh = ios[0].device()
    key = (kind, mesh.id(), params) + tuple(
        (tuple(t.shape[i] for i in range(len(t.shape))), str(t.memory_config())) for t in ios
    )
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = make()
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    pp = ttnn.bringup.prepare_program(md, ios)
    assert pp.patch_count >= 3, f"block_select {kind}: {pp.patch_count} patched slots (a frozen arg equals an address)"
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = (mesh, pp)
    return pp


def _out(like: ttnn.Tensor, width: int, dtype, output_tensor, memory_config, what: str) -> ttnn.Tensor:
    shape = list(like.shape)
    shape[-1] = width
    if output_tensor is None:
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), dtype, ttnn.ROW_MAJOR_LAYOUT, like.device(), memory_config or ttnn.DRAM_MEMORY_CONFIG
        )
    o = output_tensor
    assert (
        o.dtype == dtype and o.layout == ttnn.ROW_MAJOR_LAYOUT and list(o.shape) == shape
    ), f"block_select {what}: output_tensor must be {dtype} ROW_MAJOR {shape}, got {o.dtype} {o.layout} {list(o.shape)}"
    return o


def block_gather(
    scores: ttnn.Tensor,
    ids: ttnn.Tensor,
    *,
    block: int,
    output_tensor: ttnn.Tensor | None = None,
    memory_config=None,
    max_grid=None,
) -> ttnn.Tensor:
    """scores [.., R, W] bf16 ROW_MAJOR + block ids [.., R, K] uint32 ROW_MAJOR (equal leading dims, DRAM interleaved)
    -> [.., R, K block] bf16 ROW_MAJOR: row r holds scores[r, ids[r, j] block + o] at j block + o (an id >= W / block:
    -inf). ``block`` a power of two <= 32 dividing W (a multiple of 32). ``max_grid``: the cores' rectangle. Work
    units of K / GATHER_PARTS blocks of a row over both RISCs of every core."""
    B = int(block)
    assert 1 <= B <= 32 and B & (B - 1) == 0, f"block_gather: block {B} must be a power of two <= 32"
    _dram_rm(scores, "scores", ttnn.bfloat16)
    _dram_rm(ids, "ids", ttnn.uint32)
    W, K = int(scores.shape[-1]), int(ids.shape[-1])
    assert W % 32 == 0 and W % B == 0, f"block_gather: width {W} must be a multiple of 32 and of the block {B}"
    assert list(scores.shape)[:-1] == list(ids.shape)[:-1], (scores.shape, ids.shape)
    # (a part's ids are read at a 64-byte aligned offset: Blackhole's DRAM read alignment)
    parts = next(p for p in (GATHER_PARTS, 2, 1) if p == 1 or (K % p == 0 and (K // p) % 16 == 0))
    out = _out(scores, K * B, ttnn.bfloat16, output_tensor, memory_config, "block_gather")
    _dram_rm(out, "output", ttnn.bfloat16)
    rows = _rows(scores)
    gx, gy = _grid(scores.device(), max_grid)
    ncores = gx * gy
    kp = K // parts
    span = -(-4 * kp // 64) * 64 + -(-2 * B * kp // 64) * 64 + (0 if 2 * B == 64 else 64 * kp)
    scratch = 64 + 2 * span

    def make():
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        cbs = [
            ttnn.CBDescriptor(
                total_size=scratch,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB, data_format=ttnn.uint8, page_size=scratch)
                ],
            )
        ]
        acc = []
        for t in (scores, ids, out):
            acc += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        common = [int(scores.buffer_address()), int(ids.buffer_address()), int(out.buffer_address()), rows]
        ks = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNELS / "block_gather.cpp"),
                core_ranges=crs,
                compile_time_args=[CB, B, K, W // B, gx, parts, half, ncores] + acc,
                runtime_args=ttnn.RuntimeArgs(),
                common_runtime_args=list(common),
                config=cfg,
            )
            for half, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor()))
        ]
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)

    pp = _prepared("gather", [scores, ids, out], (B, _mg_key(max_grid)), make)
    ttnn.bringup.run_prepared(pp, [scores, ids, out])
    return out


def block_keys(
    pos: ttnn.Tensor,
    ids: ttnn.Tensor,
    *,
    block: int,
    width: int,
    output_tensor: ttnn.Tensor | None = None,
    memory_config=None,
    max_grid=None,
) -> ttnn.Tensor:
    """pos [.., R, KP] uint32 ROW_MAJOR (positions in block_gather's [.., R, K block] row) + the block ids [.., R, K]
    uint32 ROW_MAJOR it read -> [.., R, KP] uint32 ROW_MAJOR key ids ids[pos / block] block + pos % block
    (0xFFFFFFFF where pos is none or its block was: an id >= width / block). ``width``: the scores' W."""
    B = int(block)
    assert 1 <= B <= 32 and B & (B - 1) == 0, f"block_keys: block {B} must be a power of two <= 32"
    _dram_rm(pos, "pos", ttnn.uint32)
    _dram_rm(ids, "ids", ttnn.uint32)
    assert list(pos.shape)[:-1] == list(ids.shape)[:-1], (pos.shape, ids.shape)
    KP, K = int(pos.shape[-1]), int(ids.shape[-1])
    out = _out(pos, KP, ttnn.uint32, output_tensor, memory_config, "block_keys")
    _dram_rm(out, "output", ttnn.uint32)
    rows = _rows(pos)
    gx, _ = _grid(pos.device(), max_grid)
    scratch = 64 + ((4 * KP + 63) // 64) * 64 + ((4 * K + 63) // 64) * 64

    nbw = int(width) // B

    def make():
        return _program("block_keys.cpp", [pos, ids, out], [B, K, KP, nbw, gx], scratch, rows, max_grid)

    pp = _prepared("keys", [pos, ids, out], (B, nbw, _mg_key(max_grid)), make)
    ttnn.bringup.run_prepared(pp, [pos, ids, out])
    return out


COMMON_ARG = 1 << 31  # (prepare_program: a runtime-arg index with bit 31 names a COMMON runtime arg)
CAST_CHUNK = 2048  # input values a work unit (8 KiB of L1)


def block_max_bf16(
    block_max: ttnn.Tensor,
    *,
    factor: int = 1,
    n: int | None = None,
    n_tensor: ttnn.Tensor | None = None,
    output_tensor: ttnn.Tensor | None = None,
    memory_config=None,
    max_grid=None,
) -> ttnn.Tensor:
    """block_max [.., R, NI] fp32 ROW_MAJOR (bf16 values widened: the indexer score's block_max_out) -> [.., R,
    NI / factor] bf16 ROW_MAJOR, column b = the max of input columns [b factor, (b + 1) factor) (exact), for the first
    ``n`` output columns (rounded up to 8; the rest not written). ``n`` a runtime value (one program for every n) or
    ``n_tensor`` ([1] uint32 ROW_MAJOR DRAM a device, read on the device). ``factor`` a power of two <= 16."""
    F = int(factor)
    assert 1 <= F <= 16 and F & (F - 1) == 0, f"block_max_bf16: factor {F}"
    _dram_rm(block_max, "block_max", ttnn.float32)
    NI = int(block_max.shape[-1])
    assert NI % (8 * F) == 0, f"block_max_bf16: width {NI} must be a multiple of 8 x factor {F}"
    assert (n is None) != (n_tensor is None), "block_max_bf16: exactly one of n / n_tensor"
    out = _out(block_max, NI // F, ttnn.bfloat16, output_tensor, memory_config, "block_max_bf16")
    _dram_rm(out, "output", ttnn.bfloat16)
    rows = _rows(block_max)
    gx, gy = _grid(block_max.device(), max_grid)
    ncores = gx * gy
    has_len = n_tensor is not None
    ios = [block_max, out] + ([n_tensor] if has_len else [])
    span = -(-(CAST_CHUNK * 4 + CAST_CHUNK // F * 2 + 64) // 64) * 64
    scratch = 64 + 4 * span  # (two RISCs x two buffers)

    def make():
        cores = [(i % gx, i // gx) for i in range(ncores)]
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        assert len(cores) == gx * gy
        cbs = [
            ttnn.CBDescriptor(
                total_size=scratch,
                core_ranges=crs,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=CB, data_format=ttnn.uint8, page_size=scratch)
                ],
            )
        ]
        ct = [CB, F, NI, gx, ncores, CAST_CHUNK, int(has_len)]
        for t in (block_max, out, n_tensor if has_len else out):
            ct += list(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        common = [int(block_max.buffer_address()), int(out.buffer_address())]
        common += [int(n_tensor.buffer_address()) if has_len else 0, 0, rows]
        ks = []
        for half, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor())):
            cth = ct[:7] + [half] + ct[7:]
            ks.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNELS / "block_max_bf16.cpp"),
                    core_ranges=crs,
                    compile_time_args=cth,
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=list(common),
                    config=cfg,
                )
            )
        return ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=cbs)

    mesh = block_max.device()
    key = ("bmax", mesh.id(), F, has_len, _mg_key(max_grid)) + tuple(
        (tuple(t.shape[i] for i in range(len(t.shape))), str(t.memory_config())) for t in ios
    )
    hit = _PREPARED.get(key)
    if hit is None:
        shape = mesh.shape
        md = ttnn.MeshProgramDescriptor()
        md[
            ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))
        ] = make()
        slots = [] if has_len else [(0, COMMON_ARG | 3), (1, COMMON_ARG | 3)]
        pp = ttnn.bringup.prepare_program(md, ios, slots)
        assert pp.patch_count >= len(ios), f"block_max_bf16: {pp.patch_count} patched slots"
        if len(_PREPARED) >= _PREPARED_CAP:
            _PREPARED.popitem(last=False)
        _PREPARED[key] = hit = (mesh, pp)
    else:
        _PREPARED.move_to_end(key)
    if has_len:
        ttnn.bringup.run_prepared(hit[1], ios)
    else:
        ttnn.bringup.run_prepared(hit[1], ios, [int(n), int(n)])
    return out
