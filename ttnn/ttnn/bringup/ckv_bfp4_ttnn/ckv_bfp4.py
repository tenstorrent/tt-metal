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


def _crs(cores, compact: bool = False):
    """The cores' CoreRangeSet: one single-core range each (default, the fork's original program), or ``compact``
    (option, P.12): merged into rectangles (whole rows of the grid + the rest of the last row), the same cores, kernels
    and runtime args - the device program is the same, the host dispatch walks a few ranges instead of one per core."""
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    return crs.merge(crs) if compact else crs  # (merge: CoreRangeSet::merge_ranges, the bound name)


def _cores(device, bands: int):
    g = device.compute_with_storage_grid_size()
    n = max(1, min(bands, g.x * g.y))
    return [(i % g.x, i // g.x) for i in range(n)]


def _common_defines(device, compact: bool):
    """compact: the grid width the kernels' row-wise core index needs (ckv_share.hpp)."""
    return [("CKV_COMMON_GX", str(device.compute_with_storage_grid_size().x))] if compact else []


def _reader(src: ttnn.Tensor, prefix: ttnn.Tensor | None, unit: int, compact: bool):
    """(kernel source, compile-time args, defines) of the reader: the stock reader_unary_interleaved_start_id by
    default; the fork's ckv_bfp4_reader.cpp for the options prefix (two sources) and compact (common args)."""
    acc = list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
    if prefix is None and not compact:
        return _RD, acc, []
    split = prefix.shape[-2] // 32 * (WIDTH // 32) if prefix is not None else 0
    ct = [unit, split] + (list(ttnn.TensorAccessorArgs(prefix).get_compile_time_args()) if prefix is not None else [])
    defines = ([("CKV_PREFIX", "1")] if prefix is not None else []) + _common_defines(src.device(), compact)
    return str(KERNELS / "ckv_bfp4_reader.cpp"), ct + acc, defines


def program(
    ckv: ttnn.Tensor,
    rows: ttnn.Tensor,
    row0: int,
    rounded: ttnn.Tensor | None,
    dtype=None,
    blocks: bool = False,
    prefix: ttnn.Tensor | None = None,
    compact: bool = False,
) -> ttnn.ProgramDescriptor:
    dtype = dtype or ttnn.bfloat4_b
    row_bytes, mbits = _fmt(dtype)
    page_bytes = BLOCK_BYTES if blocks else 32 * row_bytes
    npre = prefix.shape[-2] if prefix is not None else 0
    n = npre + ckv.shape[-2]
    bands, tpb = n // 32, WIDTH // 32
    cores = _cores(ckv.device(), bands)
    crs = _crs(cores, compact)
    per, extra = divmod(bands, len(cores))
    src = [ckv.buffer_address()] if prefix is None else [prefix.buffer_address(), ckv.buffer_address()]
    out_addr = [rounded.buffer_address()] if rounded is not None else []
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd_c, cp_c, wr_c = [], [], []
    if compact:  # every split from the common args (ckv_share.hpp); no per-core args, row0 a common arg
        rd_c, cp_c, wr_c = src + [per, extra], [per, extra], [rows.buffer_address(), per, extra, row0] + out_addr
    else:
        b0 = 0
        for i, (x, y) in enumerate(cores):
            nb = per + (i < extra)
            rd[x][y] = src + [nb * tpb, b0 * tpb]
            cp[x][y] = [nb]
            wr[x][y] = [rows.buffer_address(), b0, nb, row0] + out_addr
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
    rd_src, rd_ct, rd_def = _reader(ckv, prefix, tpb, compact)
    common = _common_defines(ckv.device(), compact)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=rd_src,
            core_ranges=crs,
            compile_time_args=rd_ct,
            defines=rd_def,
            runtime_args=rd,
            common_runtime_args=rd_c,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "ckv_bfp4_writer.cpp"),
            core_ranges=crs,
            compile_time_args=wr_ct,
            defines=defines + common,
            runtime_args=wr,
            common_runtime_args=wr_c,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "ckv_bfp4_compute.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_IN, CB_ROWS, CB_ROUNDED],
            defines=defines + block_defines + fmt_defines + common + [("CKV_PACK_PRECISE", "1")],
            runtime_args=cp,
            common_runtime_args=cp_c,
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
    prefix: ttnn.Tensor | None = None,
    compact: bool = False,
) -> ttnn.Tensor | None:
    """ckv [1, 1, n, 512] bf16 TILE (n % 32 == 0; on a mesh, the same program on every chip) -> rows [row0, row0 + n)
    of ``rows`` ([1, 1, R, 320] uint8 ROW_MAJOR, written in place). ``rounded``: also return the rounded values as a new
    [1, 1, n, 512] bf16 TILE tensor (else None). ``dtype``: bfloat4_b (default, 320 B records) or bfloat8_b (576 B
    records: [32 exponents | the same 32 | 512 sign-magnitude bytes]). ``blocks`` (bfloat8_b only): ``rows`` is
    [1, 1, R / 32, BLOCK_BYTES], 32 consecutive rows a page (the sparse_sdpa window: two NoC reads a block), row0 % 32
    == 0. ``prepared`` (default False: a fresh descriptor through generic_op every call): the program is built once per
    (mesh, n, rounded, dtype, blocks, memory configs) and dispatched as a prepared program (ttnn.bringup.run_prepared,
    row0 a value slot): the same program and bytes without the per-call descriptor build. ``prefix`` (option, default
    None; not with ``rounded``): a second bf16 TILE input [1, 1, m, 512] (m % 32 == 0) whose rows go first - one call
    writes [prefix | ckv] at [row0, row0 + m + n), the same records as two calls (prefix at row0, ckv at row0 + m).
    ``compact`` (option, default False): the program's cores as merged rectangles instead of one range per core (the
    same cores, kernels and runtime args; less host dispatch work a call)."""
    if prefix is not None:
        assert not rounded, "prefix: no rounded output"
        assert prefix.dtype == ttnn.bfloat16 and prefix.layout == ttnn.TILE_LAYOUT, (prefix.dtype, prefix.layout)
        assert prefix.shape[-1] == WIDTH and prefix.shape[-2] % 32 == 0, prefix.shape
    n = ckv.shape[-2] + (prefix.shape[-2] if prefix is not None else 0)
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
    io = ([prefix] if prefix is not None else []) + [ckv, rows] + ([out] if out is not None else [])
    pp = _prepared(ckv, rows, out, row0, io, dtype, blocks, prefix, compact) if prepared else None
    if pp is not None:
        # (looked up on ttnn.bringup at each call: the bring-up capture wraps run_prepared, F201)
        ttnn.bringup.run_prepared(pp, io, [] if compact else [row0])
    else:
        ttnn.generic_op(io, program(ckv, rows, row0, out, dtype, blocks, prefix, compact))
    return out


BLOCK = 16  # tiles per full-sync DEST block


def _vol(t: ttnn.Tensor) -> int:
    v = 1
    for d in t.padded_shape:
        v *= d
    return v


def round_program(x: ttnn.Tensor, out: ttnn.Tensor, page0: int, compact: bool = False) -> ttnn.ProgramDescriptor:
    nt = _vol(x) // 1024
    g = x.device().compute_with_storage_grid_size()
    ncores = max(1, min(nt, g.x * g.y))  # every core (the SFPU round is the cost); blocks of up to BLOCK tiles
    cores = [(i % g.x, i // g.x) for i in range(ncores)]
    crs = _crs(cores, compact)
    per, extra = divmod(nt, ncores)
    rd, cp, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd_c, cp_c, wr_c = [], [], []
    if compact:  # every split from the common args (ckv_share.hpp); no per-core args, page0 a common arg
        rd_c, cp_c = [x.buffer_address(), per, extra], [per, extra]
        wr_c = [out.buffer_address(), per, extra, page0]
    else:
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
    rd_src, rd_ct, rd_def = _reader(x, None, 1, compact)
    common = _common_defines(x.device(), compact)
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=rd_src,
            core_ranges=crs,
            compile_time_args=rd_ct,
            defines=rd_def,
            runtime_args=rd,
            common_runtime_args=rd_c,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "bfp_round_writer.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_ROWS, BLOCK] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()),
            defines=common,
            runtime_args=wr,
            common_runtime_args=wr_c,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNELS / "bfp_round_compute.cpp"),
            core_ranges=crs,
            compile_time_args=[CB_IN, CB_ROWS, BLOCK],
            # precise pack: one conversion of the bf16 DEST at the group's exponent (exact on the grid); the define
            # gives the binary its own cache key (config_hash leaves bfp8_pack_precise out)
            defines=[("CKV_MBITS", str(mbits)), ("CKV_PACK_PRECISE", "1")] + common,
            runtime_args=cp,
            common_runtime_args=cp_c,
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
    compact: bool = False,
) -> ttnn.Tensor:
    """x bf16 TILE (any rank, its tiles in order) -> its values rounded to nearest even onto the ``dtype`` grid
    (bfloat4_b default, or bfloat8_b; one exponent per 16 values of a tile's face row), as a new TILE tensor of x's
    shape, or written in place into ``out`` (TILE, ``dtype``, x's width) at row ``row0`` (a multiple of 32) of its
    flattened rows; returns that tensor. On a mesh, the same program on every chip. ``prepared`` (default False: a
    fresh descriptor through generic_op every call): built once per (mesh, tiles, dtype, memory configs) and dispatched
    by ttnn.bringup.run_prepared, the table's first page a value slot (as ckv_bfp4's). ``compact``: as ckv_bfp4's."""
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
    pp = _prepared_round(x, out, page0, io, compact) if prepared else None
    if pp is not None:
        ttnn.bringup.run_prepared(pp, io, [] if compact else [page0])
    else:
        ttnn.generic_op(io, round_program(x, out, page0, compact))
    return out


def _prepared_round(x, out, page0, io, compact=False):
    """bfp_round(prepared=True): the round program for this (mesh, tiles, dtype, memory configs), prepared once (page0 a
    value slot), or None when it cannot be prepared (the patch count says a frozen arg matched an io address)."""
    mesh = x.device()
    nt = _vol(x) // 1024
    # (compact: page0 is frozen in the program, so it is in the key)
    key = (
        "round",
        mesh.id(),
        nt,
        str(out.dtype),
        str(x.memory_config()),
        str(out.memory_config()),
        compact,
        page0 if compact else None,
    )
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = round_program(x, out, page0, compact)
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    pp = ttnn.bringup.prepare_program(md, io, [] if compact else [(K_WRITER, RW_PAGE0)])
    g = mesh.compute_with_storage_grid_size()
    cores = max(1, min(nt, g.x * g.y))
    # reader: x; writer: out, page0 - one slot each a core (compact: x and out one common slot each); anything else
    # matched a frozen value
    entry = (mesh, pp if pp.patch_count == (2 if compact else cores * 3) else None)
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = entry
    return entry[1]


def _prepared(ckv, rows, out, row0, io, dtype=None, blocks=False, prefix=None, compact=False):
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
        prefix.shape[-2] if prefix is not None else None,
        str(prefix.memory_config()) if prefix is not None else None,
        row0 if compact else None,  # (compact: row0 is frozen in the program)
        compact,
    )
    hit = _PREPARED.get(key)
    if hit is not None:
        _PREPARED.move_to_end(key)
        return hit[1]
    desc = program(ckv, rows, row0, out, dtype, blocks, prefix, compact)
    shape = mesh.shape
    md = ttnn.MeshProgramDescriptor()
    md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(shape[0] - 1, shape[1] - 1))] = desc
    pp = ttnn.bringup.prepare_program(md, io, [] if compact else [(K_WRITER, WR_ROW0)])
    npre = prefix.shape[-2] if prefix is not None else 0
    cores = len(_cores(mesh, (npre + ckv.shape[-2]) // 32))
    # reader: ckv [, prefix]; writer: rows [, out], row0 - one slot each a core (compact: the tensors one common slot
    # each, row0 frozen); anything else matched a frozen value
    nio = 2 + (out is not None) + (prefix is not None)
    entry = (mesh, pp if pp.patch_count == (nio if compact else cores * (nio + 1)) else None)
    if len(_PREPARED) >= _PREPARED_CAP:
        _PREPARED.popitem(last=False)
    _PREPARED[key] = entry
    return entry[1]
