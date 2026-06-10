# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for groupnorm_sc_N_1_HW_C.

Single core (0,0). Per (n, g) group: three streaming passes over the
Ht x Wg tile slab (mean, variance, normalize+affine); the reader re-reads
the slab from DRAM each pass so HW can be arbitrarily large.

CB layout (per op_design.md):
  cb_input_tiles  (0):  2*Wg pages, in_dtype — pass 1/2/3 stream
  cb_gamma_tiles  (1):  Wt pages, affine dtype — read once, HeldBulk
  cb_beta_tiles   (2):  Wt pages, affine dtype — read once, HeldBulk
  cb_scaler       (8):  1 page, bf16 — 1/sqrt(HW*Cg), pushed once
  cb_output_tiles (16): 2*Wg pages — compute -> writer stream
  cb_mean         (24): 1 page — per-group mean, HeldBulk in passes 2/3
  cb_var          (25): 1 page — variance accumulator -> rstd
  cb_centered     (26): 2*Wg pages — (x - mean) per row chunk
  cb_xhat         (27): 2*Wg pages — (x - mean)*rstd (HAS_GAMMA only)
  cb_scaled       (28): 2*Wg pages — xhat*gamma (HAS_GAMMA && HAS_BETA only)
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", value))[0]


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    gamma_tensor,  # TILE-layout (1,1,1,C) or None
    beta_tensor,  # TILE-layout (1,1,1,C) or None
    output_tensor: ttnn.Tensor,
    num_groups: int,
    eps: float,
) -> ttnn.ProgramDescriptor:
    N, _, HW, C = list(input_tensor.shape)
    G = num_groups
    Cg = C // G
    Ht = HW // TILE_DIM
    Wt = C // TILE_DIM
    Wg = Cg // TILE_DIM

    has_gamma = gamma_tensor is not None
    has_beta = beta_tensor is not None

    # REDUCE_SCALAR applies the scaler twice (row then col), so 1/sqrt(N_grp)
    # turns the SUM into a mean over the HW x Cg group slab.
    n_grp = HW * Cg
    inv_sqrt_n_bits = _f32_bits(1.0 / math.sqrt(float(n_grp)))
    eps_bits = _f32_bits(eps)

    in_page = input_tensor.buffer_page_size()
    out_page = output_tensor.buffer_page_size()
    bf16_page = ttnn.tile_size(ttnn.bfloat16)

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    CB_INPUT_TILES = 0
    CB_GAMMA_TILES = 1
    CB_BETA_TILES = 2
    CB_SCALER = 8
    CB_OUTPUT_TILES = 16
    CB_MEAN = 24
    CB_VAR = 25
    CB_CENTERED = 26
    CB_XHAT = 27
    CB_SCALED = 28

    def cb(index, pages, page_size, dtype):
        return ttnn.CBDescriptor(
            total_size=pages * page_size,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    in_dtype = input_tensor.dtype
    stream_pages = 2 * Wg

    cbs = [
        cb(CB_INPUT_TILES, stream_pages, in_page, in_dtype),
        cb(CB_SCALER, 1, bf16_page, ttnn.bfloat16),
        cb(CB_OUTPUT_TILES, stream_pages, out_page, in_dtype),
        cb(CB_MEAN, 1, in_page, in_dtype),
        cb(CB_VAR, 1, in_page, in_dtype),
        cb(CB_CENTERED, stream_pages, in_page, in_dtype),
    ]
    if has_gamma:
        gamma_page = gamma_tensor.buffer_page_size()
        cbs.append(cb(CB_GAMMA_TILES, Wt, gamma_page, gamma_tensor.dtype))
        cbs.append(cb(CB_XHAT, stream_pages, in_page, in_dtype))
    if has_beta:
        beta_page = beta_tensor.buffer_page_size()
        cbs.append(cb(CB_BETA_TILES, Wt, beta_page, beta_tensor.dtype))
        cbs.append(cb(CB_SCALED, stream_pages, in_page, in_dtype))

    # --- Reader: scalars first, TensorAccessorArgs at the end (input, gamma, beta) ---
    reader_ct_args = [Ht, Wt, Wg, G, N, int(has_gamma), int(has_beta), inv_sqrt_n_bits]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma_tensor).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(beta_tensor).get_compile_time_args()
        if has_beta
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma_tensor.buffer_address() if has_gamma else 0,
        beta_tensor.buffer_address() if has_beta else 0,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    writer_ct_args = [Ht, Wt, Wg, G, N]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [output_tensor.buffer_address()]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---
    compute_ct_args = [Ht, Wt, Wg, G, N, int(has_gamma), int(has_beta), eps_bits]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
