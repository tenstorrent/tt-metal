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


# ---------------------------------------------------------------------------------------------------------------------
# ttnn.bringup.unpack_fp8_rows: the inverse reader (deepseek_v41_flash_d_p P.76, INDEX_K_SOURCE=served)

CB_E, CB_S, CB_OUT, CB_RSTAGE = 0, 1, 2, 3
NUM_CB_SLOTS = 64  # (ComputeConfigDescriptor.unpack_to_dest_mode's length)
K_READER, K_WRITER, K_COMPUTE = 0, 1, 2
# value slots (prepare_program): reader src_page0, nbands; writer out_page0, nbands; compute nbands
_UNPACK_SLOTS = [(K_READER, 0), (K_READER, 1), (K_WRITER, 0), (K_WRITER, 1), (K_COMPUTE, 0)]
COMMON_ARG = 1 << 31  # (prepare_program: a runtime-arg index with bit 31 names a COMMON runtime arg)
# option common_values: the same five values as common args (reader / writer 1, 2; compute 0), one patch a program
_UNPACK_COMMON_SLOTS = [
    (K_READER, COMMON_ARG | 1),
    (K_READER, COMMON_ARG | 2),
    (K_WRITER, COMMON_ARG | 1),
    (K_WRITER, COMMON_ARG | 2),
    (K_COMPUTE, COMMON_ARG | 0),
]


def _unpack_cores(device, nbands: int, max_grid=None) -> int:
    """The program's core count: one band a core up to the grid (the per-core value slots cost host time a core, so
    a small call runs on as many cores as it has bands; a call of at least the grid's bands on the whole grid)."""
    gx, gy = _grid(device, max_grid)
    return max(1, min(gx * gy, nbands))


def unpack_program(
    rows: ttnn.Tensor, out: ttnn.Tensor, ncores: int, max_grid=None, common_values: bool = False
) -> ttnn.ProgramDescriptor:
    """One program for every call of this (rows, out) spec and core count: the work (src_page0, out_page0, nbands) is
    a value slot on every core (the same values on every core and chip; each core takes its share of the bands by its
    index, cores y GX + x < ncores of the grid). ``common_values``: the values are COMMON runtime args (reader / writer
    args 1, 2, compute arg 0; no per-core args), one patch a program instead of one a core (prepare_program's common
    value slots)."""
    gx, gy = _grid(rows.device(), max_grid)
    assert 1 <= ncores <= gx * gy, (ncores, gx, gy)
    cores = [(i % gx, i // gx) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])
    crs = crs.merge(crs)
    fp8_tile, f32_tile, b8_tile = (ttnn.tile_size(d) for d in (ttnn.fp8_e4m3, ttnn.float32, ttnn.bfloat8_b))
    stage = 32 * ROW_BYTES + 64

    def cb(idx, fmt, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=fmt, page_size=page)],
        )

    # (double-buffered: the next band's read and shuffle overlap this band's compute)
    cbs = [
        cb(CB_E, ttnn.fp8_e4m3, fp8_tile, 8),
        cb(CB_S, ttnn.float32, f32_tile, 2),
        cb(CB_OUT, ttnn.bfloat8_b, b8_tile, 8),
        cb(CB_RSTAGE, ttnn.uint8, stage, 1),
    ]
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    if not common_values:
        for cx, cy in cores:  # (value slots: placeholders, written by every run)
            rd[cx][cy] = [0, 0]
            wr[cx][cy] = [0, 0]
            cp[cx][cy] = [0]
    vals = [0, 0] if common_values else []  # (common value slots: placeholders, written by every run)
    defs = [("UNPACK_COMMON_VALUES", "1")] if common_values else []
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
        dst_full_sync_en=True,
        bfp8_pack_precise=True,
    )
    modes = [ttnn.UnpackToDestMode.Default] * NUM_CB_SLOTS
    modes[CB_S] = ttnn.UnpackToDestMode.UnpackToDestFp32  # the scales exact in DEST
    cfg.unpack_to_dest_mode = modes
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "unpack_fp8_rows_reader.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_E, CB_S, CB_RSTAGE, gx, ncores]
            + list(ttnn.TensorAccessorArgs(rows).get_compile_time_args()),
            runtime_args=rd,
            common_runtime_args=[rows.buffer_address()] + vals,
            defines=defs,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "unpack_fp8_rows_writer.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_OUT, gx, ncores] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr,
            common_runtime_args=[out.buffer_address()] + vals,
            defines=defs,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "unpack_fp8_rows_compute.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_E, CB_S, CB_OUT, gx, ncores],
            runtime_args=cp,
            common_runtime_args=[0] if common_values else [],
            defines=defs,
            config=cfg,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def _prepared_unpack(rows: ttnn.Tensor, out: ttnn.Tensor, ncores: int, max_grid=None, common_values: bool = False):
    """The program for this (mesh, shapes, memory configs, grid), built and prepared once: the two addresses patched,
    the work a value slot (``common_values``: common value slots)."""
    mesh = rows.device()
    key = (
        "unpack",
        mesh.id(),
        tuple(rows.shape[i] for i in range(len(rows.shape))),
        tuple(out.shape[i] for i in range(len(out.shape))),
        str(rows.memory_config()),
        str(out.memory_config()),
        ncores,
        _mg_key(max_grid),
        bool(common_values),
    )
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = unpack_program(rows, out, ncores, max_grid, common_values)
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    slots = _UNPACK_COMMON_SLOTS if common_values else _UNPACK_SLOTS
    pp = ttnn.bringup.prepare_program(md, [rows, out], slots)
    # (the two common addresses + every core's value slots, or the common value slots once)
    want = 2 + (len(slots) if common_values else ncores * len(slots))
    assert (
        pp.patch_count == want
    ), f"unpack_fp8_rows: {pp.patch_count} patched slots, not {want} (a frozen arg equals an io address)"
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = (mesh, pp)
    return pp


def unpack_fp8_rows(
    rows: ttnn.Tensor,
    out: ttnn.Tensor,
    *,
    row_lo: int,
    row_hi: int,
    batch: int = 0,
    out_row0: int | None = None,
    max_grid=None,
    whole_grid: bool = False,
    common_values: bool = False,
) -> ttnn.Tensor:
    """The inverse of pack_fp8_rows, onto the bfloat8_b grid: rows [row_lo, row_hi) of batch ``batch`` of ``rows``
    ([B, .., R, 192] UINT8 ROW_MAJOR in DRAM, interleaved or ND-sharded: one 192-byte row a page; [e4m3 x 128 | fp32
    scale | ..]) -> e4m3 x scale (fp32 on the SFPU) rounded to nearest even onto the bfloat8_b grid (the host packer's
    rule, ckv_bfp4_ttnn's round), written in place into ``out`` ([.., M, 128] BFLOAT8_B TILE, DRAM interleaved) at
    rows [out_row0, out_row0 + row_hi - row_lo) (default out_row0 = row_lo). row_lo, row_hi, out_row0 multiples of 32
    (whole tile rows). On a mesh, the same program and rows on every chip (each chip decodes its own rows). Always a
    prepared program (OV-23): built once per (shapes, memory configs, grid), the addresses patched and the rows a
    value slot, so every range of at least the grid's bands runs one program (a smaller range: one program per band
    count, a core a band: the per-core value slots cost host time a core). ``max_grid``: as pack_fp8_rows' (the cores
    inside it; each core its share of the 32-row bands). ``whole_grid`` (default False: the behaviour before): every
    call on the whole grid, cores past the band count idle, so ONE program serves every row count (no prepare a band
    count; deepseek_v41_flash_d_p MB.2, POS_DYNAMIC: serving never builds a program after warm-up). ``common_values``
    (default False: the behaviour before): the per-run values (first page, bands, out page) are common runtime args
    (prepare_program's common value slots), so a run patches five words a program instead of five a core (the
    whole grid's 110 cores x 32 chips cost ~0.33 ms of host a call; deepseek_v41_flash_d_p PB.9X). The same bytes.
    Returns ``out``."""
    assert (
        rows.dtype == ttnn.uint8 and rows.layout == ttnn.ROW_MAJOR_LAYOUT and rows.shape[-1] == ROW_BYTES
    ), f"unpack_fp8_rows: rows must be uint8 ROW_MAJOR [.., {ROW_BYTES}], got {rows.dtype} {rows.layout} {rows.shape}"
    assert (
        out.dtype == ttnn.bfloat8_b and out.layout == ttnn.TILE_LAYOUT and out.shape[-1] == VALUES
    ), f"unpack_fp8_rows: out must be bfloat8_b TILE [.., {VALUES}], got {out.dtype} {out.layout} {out.shape}"
    assert rows.memory_config().buffer_type == ttnn.BufferType.DRAM, "unpack_fp8_rows: rows must be in DRAM"
    _interleaved_dram(out, "out")
    r = int(rows.shape[-2])
    lead = 1
    for i in range(len(rows.shape) - 2):
        lead *= int(rows.shape[i])
    m = _rows(out)
    out_row0 = row_lo if out_row0 is None else out_row0
    n = row_hi - row_lo
    assert (
        0 <= batch < lead and 0 <= row_lo <= row_hi <= r
    ), f"unpack_fp8_rows: rows [{row_lo}, {row_hi}) of batch {batch} outside [{lead}, .., {r}]"
    assert (
        row_lo % 32 == 0 and n % 32 == 0 and out_row0 % 32 == 0
    ), f"unpack_fp8_rows: whole tile rows only (row_lo {row_lo}, row_hi {row_hi}, out_row0 {out_row0})"
    assert out_row0 + n <= m, f"unpack_fp8_rows: out rows [{out_row0}, {out_row0 + n}) past its {m} rows"
    if n == 0:
        return out
    nb = n // 32
    gx, gy = _grid(rows.device(), max_grid)
    ncores = gx * gy if whole_grid else _unpack_cores(rows.device(), nb, max_grid)
    pp = _prepared_unpack(rows, out, ncores, max_grid, common_values)
    ttnn.bringup.run_prepared(pp, [rows, out], [batch * r + row_lo, nb, out_row0 // 32 * (VALUES // 32), nb, nb])
    return out
