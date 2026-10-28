# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.ckv_bfp4: bf16 TILE rows [.., n, 512] -> 320-byte bfloat4_b row records written into a row table at
row0, rounded to nearest even like tt-metal's host packer (an SFPU round before the pack: the packer alone truncates
bfp4 in effect), plus optionally the same rounded values as bf16 TILE.

Row record (320 B): the 32 shared exponents (one per 16 values, the group's max fp32 biased exponent) twice, then 256
bytes of sign-magnitude nibbles (value 2i in the low nibble of byte i) = tt/ckv_codec.py format B with its exponents
repeated. The second copy lets a reader's NoC read put a row's exponents at either 32 B half of a 64 B line, so a
gathered block of rows is a row-major bfloat4_b tensor the unpacker reads (sparse_sdpa ROW_BFP4).
The packer writes the records (two streams a row: exponents at +0 and +32, values at +64); the data-movement RISCs
only move tiles and records.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNELS = Path(__file__).parent / "kernels"
_RD = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp"
WIDTH, ROW_BYTES, GROUP = 512, 320, 16
CB_IN, CB_ROWS, CB_ROUNDED = 0, 16, 17


def _cores(device, bands: int):
    g = device.compute_with_storage_grid_size()
    n = max(1, min(bands, g.x * g.y))
    return [(i % g.x, i // g.x) for i in range(n)]


def program(ckv: ttnn.Tensor, rows: ttnn.Tensor, row0: int, rounded: ttnn.Tensor | None) -> ttnn.ProgramDescriptor:
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
    cbs = [cb(CB_IN, ttnn.bfloat16, t_bf16, 2 * tpb), cb(CB_ROWS, ttnn.bfloat4_b, 32 * ROW_BYTES, 2)]
    defines = []
    if rounded is not None:
        cbs.append(cb(CB_ROUNDED, ttnn.bfloat16, t_bf16, 2 * tpb))
        defines = [("CKV_ROUNDED_OUT", "1")]
    wr_ct = [CB_ROWS, CB_ROUNDED, tpb, ROW_BYTES] + list(ttnn.TensorAccessorArgs(rows).get_compile_time_args())
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
            defines=defines + [("CKV_PACK_PRECISE", "1")],
            runtime_args=cp,
            # full-sync DEST holds a band's 16 tiles; precise pack: the packer reads DEST as bf16, the format the row
            # pack's strides are set for (no bfloat8_b step). CKV_PACK_PRECISE only gives the binary its own cache key:
            # ComputeKernel::config_hash() leaves bfp8_pack_precise out (WH/BH).
            config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False, dst_full_sync_en=True, bfp8_pack_precise=True),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


def ckv_bfp4(
    ckv: ttnn.Tensor, rows: ttnn.Tensor, row0: int = 0, *, rounded: bool = False, memory_config=None
) -> ttnn.Tensor | None:
    """ckv [1, 1, n, 512] bf16 TILE (n % 32 == 0; on a mesh, the same program on every chip) -> rows [row0, row0 + n)
    of ``rows`` ([1, 1, R, 320] uint8 ROW_MAJOR, written in place). ``rounded``: also return the rounded values as a new
    [1, 1, n, 512] bf16 TILE tensor (else None)."""
    n = ckv.shape[-2]
    assert ckv.dtype == ttnn.bfloat16 and ckv.layout == ttnn.TILE_LAYOUT, (ckv.dtype, ckv.layout)
    assert ckv.shape[-1] == WIDTH and n % 32 == 0, ckv.shape
    assert rows.dtype == ttnn.uint8 and rows.layout == ttnn.ROW_MAJOR_LAYOUT and rows.shape[-1] == ROW_BYTES, (
        rows.dtype,
        rows.layout,
        rows.shape,
    )
    assert 0 <= row0 and row0 + n <= rows.shape[-2], (row0, n, rows.shape)
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
    ttnn.generic_op(io, program(ckv, rows, row0, out))
    return out
