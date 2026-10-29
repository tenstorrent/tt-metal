# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ckv_bfp4: bf16 TILE rows [.., n, 512] -> 320-byte bfloat4_b (or 576-byte bfloat8_b) row records written
into a row table at row0, rounded to nearest even like tt-metal's host packer (an SFPU round before the pack: the
packer alone truncates bfp4 in effect, and rounds bfp8 half away from zero), plus optionally the same rounded values
as bf16 TILE.
ttnn.bringup.bfp_round: bf16 TILE -> bfloat4_b / bfloat8_b TILE by the same rounding, as a new tensor or in place
into a wider table at a tile row.

Row record (320 B): the 32 shared exponents (one per 16 values, the group's max fp32 biased exponent) twice, then 256
bytes of sign-magnitude nibbles (value 2i in the low nibble of byte i) = tt/ckv_codec.py format B with its exponents
repeated. bfloat8_b: the same with 512 bytes of sign-magnitude bytes. The second copy lets a reader's NoC read put a row's exponents at either 32 B half of a 64 B line, so a
gathered block of rows is a row-major bfloat4_b tensor the unpacker reads (sparse_sdpa ROW_BFP4).
The packer writes the records (two streams a row: exponents at +0 and +32, values at +64); the data-movement RISCs
only move tiles and records.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import ttnn

KERNELS = Path(__file__).parent / "kernels"
_RD = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
WIDTH, ROW_BYTES, GROUP = 512, 320, 16
# bfloat8_b window blocks (blocks=True; sparse_sdpa window_block_*): 32 rows a page, [their exponents | 32 B | the same
# | to 64 B | their values]
BLOCK_EXP2 = 32 * 32 + 32
BLOCK_DATA = (BLOCK_EXP2 + 32 * 32 + 63) // 64 * 64
BLOCK_BYTES = BLOCK_DATA + 32 * WIDTH
CB_IN, CB_ROWS, CB_ROUNDED = 0, 16, 17
K_WRITER, WR_ROW0 = 1, 3  # (program(): the writer's kernel index and row0's runtime-arg index)
RW_PAGE0 = 3  # (round_program(): page0's runtime-arg index in the writer, kernel K_WRITER)
_PREPARED_CAP = 64
_PREPARED = OrderedDict()  # prepared=True: key -> (mesh device, PreparedProgram | None) (could not prepare), LRU
_FMT = {}  # dtype -> (record bytes, magnitude bits); filled lazily (ttnn dtypes are hashable enum values)


def _fmt(dtype):
    if not _FMT:
        _FMT.update({ttnn.bfloat4_b: (ROW_BYTES, 3), ttnn.bfloat8_b: (64 + WIDTH, 7)})
    assert dtype in _FMT, f"dtype must be bfloat4_b or bfloat8_b (got {dtype})"
    return _FMT[dtype]


def _cores(device, bands: int):
    g = device.compute_with_storage_grid_size()
    n = max(1, min(bands, g.x * g.y))
    return [(i % g.x, i // g.x) for i in range(n)]


def program(
    ckv: ttnn.Tensor, rows: ttnn.Tensor, row0: int, rounded: ttnn.Tensor | None, dtype=None, blocks: bool = False
) -> ttnn.ProgramDescriptor:
    dtype = dtype or ttnn.bfloat4_b
    row_bytes, mbits = _fmt(dtype)
    page_bytes = BLOCK_BYTES if blocks else 32 * row_bytes
    n = ckv.shape[-2]
    bands, tpb = n // 32, WIDTH // 32
    cores = _cores(ckv.device(), bands)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    per, extra = divmod(bands, len(cores))
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    b0 = 0
    for i, (x, y) in enumerate(cores):
        nb = per + (i < extra)
        rd[x][y] = [ckv.buffer_address(), nb * tpb, b0 * tpb]
        cp[x][y] = [nb]
        wr[x][y] = [rows.buffer_address(), b0, nb, row0] + ([rounded.buffer_address()] if rounded is not None else [])
        b0 += nb
    t_bf16 = ttnn.tile_size(ttnn.bfloat16)

    def cb(idx, dtype, page, pages):
        return ttnn.CBDescriptor(
            total_size=page * pages,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
        )

    # a band's 16 input tiles at once (full-sync DEST holds them), double-buffered; the packer writes a band's 32
    # records into one cb_rows page (its format is the packer's output format, bfloat4_b)
    cbs = [cb(CB_IN, ttnn.bfloat16, t_bf16, 2 * tpb), cb(CB_ROWS, dtype, page_bytes, 2)]
    defines = [("CKV_BLOCKS", "1")] if blocks else []
    block_defines = [("CKV_BLOCK_EXP2", str(BLOCK_EXP2)), ("CKV_BLOCK_DATA", str(BLOCK_DATA))] if blocks else []
    fmt_defines = [] if dtype == ttnn.bfloat4_b else [("CKV_MBITS", str(mbits)), ("CKV_ROW_BYTES", str(row_bytes))]
    if rounded is not None:
        cbs.append(cb(CB_ROUNDED, ttnn.bfloat16, t_bf16, 2 * tpb))
        defines = [("CKV_ROUNDED_OUT", "1")]
    wr_ct = [CB_ROWS, CB_ROUNDED, tpb, BLOCK_BYTES if blocks else row_bytes] + list(
        ttnn.TensorAccessorArgs(rows).get_compile_time_args()
    )
    if rounded is not None:
        wr_ct += list(ttnn.TensorAccessorArgs(rounded).get_compile_time_args())
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_RD,
            core_ranges=crs,
            compile_time_args=list(ttnn.TensorAccessorArgs(ckv).get_compile_time_args()),
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "ckv_bfp4_writer.cpp"),
            core_ranges=crs,
            compile_time_args=wr_ct,
            defines=defines,
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "ckv_bfp4_compute.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_IN, CB_ROWS, CB_ROUNDED],
            defines=defines + block_defines + fmt_defines + [("CKV_PACK_PRECISE", "1")],
            runtime_args=cp,
            # full-sync DEST holds a band's 16 tiles; precise pack: the packer reads DEST as bf16, the format the row
            # pack's strides are set for (no bfloat8_b step). CKV_PACK_PRECISE only gives the binary its own cache key:
            # ComputeKernel::config_hash() leaves bfp8_pack_precise out (WH/BH).
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=True, bfp8_pack_precise=True),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def ckv_bfp4(
    ckv: ttnn.Tensor,
    rows: ttnn.Tensor,
    row0: int = 0,
    *,
    rounded: bool = False,
    dtype=None,
    blocks: bool = False,
    memory_config=None,
    prepared: bool = False,
) -> ttnn.Tensor | None:
    """ckv [1, 1, n, 512] bf16 TILE (n % 32 == 0; on a mesh, the same program on every chip) -> rows [row0, row0 + n)
    of ``rows`` ([1, 1, R, 320] uint8 ROW_MAJOR, written in place). ``rounded``: also return the rounded values as a new
    [1, 1, n, 512] bf16 TILE tensor (else None). ``dtype``: bfloat4_b (default, 320 B records) or bfloat8_b (576 B
    records: [32 exponents | the same 32 | 512 sign-magnitude bytes]). ``blocks`` (bfloat8_b only): ``rows`` is
    [1, 1, R / 32, BLOCK_BYTES], 32 consecutive rows a page (the sparse_sdpa window: two NoC reads a block), row0 % 32
    == 0. ``prepared`` (default False: a fresh descriptor through generic_op every call): the program is built once per
    (mesh, n, rounded, dtype, blocks, memory configs) and dispatched as a prepared program (ttnn.bringup.run_prepared,
    row0 a value slot): the same program and bytes without the per-call descriptor build."""
    n = ckv.shape[-2]
    row_bytes = _fmt(dtype or ttnn.bfloat4_b)[0]
    if blocks:
        assert dtype == ttnn.bfloat8_b and row0 % 32 == 0, (dtype, row0)
        row_bytes = BLOCK_BYTES
    assert ckv.dtype == ttnn.bfloat16 and ckv.layout == ttnn.TILE_LAYOUT, (ckv.dtype, ckv.layout)
    assert ckv.shape[-1] == WIDTH and n % 32 == 0, ckv.shape
    assert rows.dtype == ttnn.uint8 and rows.layout == ttnn.ROW_MAJOR_LAYOUT and rows.shape[-1] == row_bytes, (
        rows.dtype,
        rows.layout,
        rows.shape,
    )
    assert 0 <= row0 and row0 + n <= rows.shape[-2] * (32 if blocks else 1), (row0, n, rows.shape)
    out = None
    if rounded:
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(ckv.shape)),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            ckv.device(),
            memory_config or ttnn.DRAM_MEMORY_CONFIG,
        )
    io = [ckv, rows] + ([out] if out is not None else [])
    pp = _prepared(ckv, rows, out, row0, io, dtype, blocks) if prepared else None
    if pp is not None:
        # (looked up on ttnn.bringup at each call: the bring-up capture wraps run_prepared, F201)
        ttnn.bringup.run_prepared(pp, io, [row0])
    else:
        ttnn.generic_op(io, program(ckv, rows, row0, out, dtype, blocks))
    return out


BLOCK = 16  # tiles per full-sync DEST block


def _vol(t: ttnn.Tensor) -> int:
    v = 1
    for d in t.padded_shape:
        v *= d
    return v


def round_program(x: ttnn.Tensor, out: ttnn.Tensor, page0: int) -> ttnn.ProgramDescriptor:
    nt = _vol(x) // 1024
    g = x.device().compute_with_storage_grid_size()
    ncores = max(1, min(nt, g.x * g.y))  # every core (the SFPU round is the cost); blocks of up to BLOCK tiles
    cores = [(i % g.x, i // g.x) for i in range(ncores)]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x_, y_), ttnn.CoreCoord(x_, y_)) for x_, y_ in cores])
    per, extra = divmod(nt, ncores)
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    t0 = 0
    for i, (cx, cy) in enumerate(cores):
        n = per + (i < extra)
        rd[cx][cy] = [x.buffer_address(), n, t0]
        cp[cx][cy] = [n]
        wr[cx][cy] = [out.buffer_address(), n, t0, page0]
        t0 += n
    cbs = [
        ttnn.CBDescriptor(
            total_size=ttnn.tile_size(dt) * 2 * BLOCK,
            core_ranges=crs,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt, page_size=ttnn.tile_size(dt))],
        )
        for i, dt in ((CB_IN, ttnn.bfloat16), (CB_ROWS, out.dtype))
    ]
    mbits = _fmt(out.dtype)[1]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_RD,
            core_ranges=crs,
            compile_time_args=list(ttnn.TensorAccessorArgs(x).get_compile_time_args()),
            runtime_args=rd,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "bfp_round_writer.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_ROWS, BLOCK] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            runtime_args=wr,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "bfp_round_compute.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_IN, CB_ROWS, BLOCK],
            # precise pack: one conversion of the bf16 DEST at the group's exponent (exact on the grid); the define
            # gives the binary its own cache key (config_hash leaves bfp8_pack_precise out)
            defines=[("CKV_MBITS", str(mbits)), ("CKV_PACK_PRECISE", "1")],
            runtime_args=cp,
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=True, bfp8_pack_precise=True),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def bfp_round(
    x: ttnn.Tensor,
    dtype=None,
    *,
    out: ttnn.Tensor | None = None,
    row0: int = 0,
    memory_config=None,
    prepared: bool = False,
) -> ttnn.Tensor:
    """x bf16 TILE (any rank, its tiles in order) -> its values rounded to nearest even onto the ``dtype`` grid
    (bfloat4_b default, or bfloat8_b; one exponent per 16 values of a tile's face row), as a new TILE tensor of x's
    shape, or written in place into ``out`` (TILE, ``dtype``, x's width) at row ``row0`` (a multiple of 32) of its
    flattened rows; returns that tensor. On a mesh, the same program on every chip. ``prepared`` (default False: a
    fresh descriptor through generic_op every call): built once per (mesh, tiles, dtype, memory configs) and dispatched
    by ttnn.bringup.run_prepared, the table's first page a value slot (as ckv_bfp4's)."""
    dtype = dtype or (out.dtype if out is not None else ttnn.bfloat4_b)
    _fmt(dtype)
    assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
    w = x.shape[-1]
    page0 = 0
    if out is None:
        assert row0 == 0, row0
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(x.shape)), dtype, ttnn.TILE_LAYOUT, x.device(), memory_config or ttnn.DRAM_MEMORY_CONFIG
        )
    else:
        assert out.dtype == dtype and out.layout == ttnn.TILE_LAYOUT and out.shape[-1] == w, (out.dtype, out.shape)
        rows_x, rows_out = _vol(x) // w, _vol(out) // w
        assert row0 % 32 == 0 and 0 <= row0 and row0 + rows_x <= rows_out, (row0, rows_x, rows_out)
        page0 = row0 // 32 * (w // 32)
    io = [x, out]
    pp = _prepared_round(x, out, page0, io) if prepared else None
    if pp is not None:
        ttnn.bringup.run_prepared(pp, io, [page0])
    else:
        ttnn.generic_op(io, round_program(x, out, page0))
    return out


def _prepared_round(x, out, page0, io):
    """bfp_round(prepared=True): the round program for this (mesh, tiles, dtype, memory configs), prepared once (page0 a
    value slot), or None when it cannot be prepared (the patch count says a frozen arg matched an io address)."""
    mesh = x.device()
    nt = _vol(x) // 1024
    key = ("round", mesh.id(), nt, str(out.dtype), str(x.memory_config()), str(out.memory_config()))
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = round_program(x, out, page0)
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    pp = ttnn.bringup.prepare_program(md, io, [(K_WRITER, RW_PAGE0)])
    g = mesh.compute_with_storage_grid_size()
    cores = max(1, min(nt, g.x * g.y))
    # reader: x; writer: out, page0 - one slot each a core; anything else matched a frozen value
    entry = (mesh, pp if pp.patch_count == cores * 3 else None)
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = entry
    return entry[1]


def _prepared(ckv, rows, out, row0, io, dtype=None, blocks=False):
    """prepared=True: the program for this (mesh, n, rounded, dtype, blocks, memory configs), built and prepared once (row0 a
    value slot, every tensor address patched), or None when it cannot be prepared (a frozen runtime arg equal to an io
    address: the patch count says so) - the caller then takes the plain generic_op path."""
    mesh = ckv.device()
    key = (
        mesh.id(),  # (MeshDevice::id: unique per opened mesh, a reopened mesh never hits an old program)
        ckv.shape[-2],
        out is not None,
        str(dtype or ttnn.bfloat4_b),
        blocks,
        str(ckv.memory_config()),
        str(rows.memory_config()),
        str(out.memory_config()) if out is not None else None,
    )
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = program(ckv, rows, row0, out, dtype, blocks)
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    pp = ttnn.bringup.prepare_program(md, io, [(K_WRITER, WR_ROW0)])
    cores = len(_cores(mesh, ckv.shape[-2] // 32))
    # reader: ckv; writer: rows [, out], row0 - one slot each a core; anything else matched a frozen value
    entry = (mesh, pp if pp.patch_count == cores * (3 + (out is not None)) else None)
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = entry
    return entry[1]
