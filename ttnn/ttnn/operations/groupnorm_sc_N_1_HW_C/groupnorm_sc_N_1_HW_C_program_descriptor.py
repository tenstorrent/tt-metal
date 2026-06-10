# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Program descriptor for groupnorm_sc_N_1_HW_C.

Multi-core: N*G (n, g) groups are split across the full compute grid via
ttnn.split_work_to_cores (work unit = one group; embarrassingly parallel over
interleaved DRAM — group slabs are disjoint, gamma/beta read by each core).
Per group the compute kernel makes three streaming passes over the Ht x Wg
tile slab (mean, variance, normalize+affine); the reader re-reads the slab
from DRAM each pass so HW can be arbitrarily large.

CB layout (per op_design.md, refined dtype handling):
  cb_input_tiles  (0):  2*Wg pages, in_dtype — pass 1/2/3 stream
  cb_gamma_tiles  (1):  Wt pages, affine dtype — read once, HeldBulk
  cb_beta_tiles   (2):  Wt pages, affine dtype — read once, HeldBulk
  cb_scaler       (8):  1 page, bf16 — 1/sqrt(HW*Cg), pushed once
  cb_output_tiles (16): 2*Wg pages, out_dtype — compute -> writer stream
  cb_mean         (24): 1 page, stat fmt — per-group mean, HeldBulk passes 2/3
  cb_var          (25): 1 page, stat fmt — variance accumulator -> rstd
  cb_centered     (26): 2*Wg pages, stat fmt — (x - mean) per row chunk
  cb_xhat         (27): 2*Wg pages, stat fmt — (x - mean)*rstd (HAS_GAMMA only)
  cb_scaled       (28): 2*Wg pages, stat fmt — xhat*gamma (HAS_GAMMA && HAS_BETA)

Intermediate (stat) format: Float32 when fp32_dest_acc_en is set (accumulation
crosses these CBs between passes — packing to bf16 would erase the fp32 dest
gain) or when the input itself is fp32; bfloat16 otherwise (incl. bf8b input —
block-float intermediates would lose precision for no L1 win). No CB carries
UnpackToDestMode::UnpackToDestFp32: every intermediate feeds FPU helpers
(sub/mul/square/reduce), which the tag forbids.
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
    compute_kernel_config=None,
) -> ttnn.ProgramDescriptor:
    N, _, HW, C = list(input_tensor.shape)
    G = num_groups
    Cg = C // G
    Ht = HW // TILE_DIM
    Wt = C // TILE_DIM
    Wg = Cg // TILE_DIM

    has_gamma = gamma_tensor is not None
    has_beta = beta_tensor is not None

    # --- compute config ---
    # Defaults reproduce Phase-0 ComputeConfigDescriptor() for bf16/bf8b inputs.
    # fp32 input defaults fp32_dest_acc_en=True (dtype-driven; fp32 was never
    # supported before so there is no prior behavior to preserve, and TF32
    # dest rounding otherwise dominates the fp32 stat error budget — measured
    # rel_rms 0.0112 -> 0.0075 on (1,1,1024,256) G=8 + bf8b gamma).
    if compute_kernel_config is not None:
        math_fidelity = compute_kernel_config.math_fidelity
        fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en
        math_approx_mode = compute_kernel_config.math_approx_mode
        dst_full_sync_en = compute_kernel_config.dst_full_sync_en
    else:
        math_fidelity = ttnn.MathFidelity.HiFi4
        fp32_dest_acc_en = input_tensor.dtype == ttnn.float32
        math_approx_mode = False
        dst_full_sync_en = False

    # REDUCE_SCALAR applies the scaler twice (row then col), so 1/sqrt(N_grp)
    # turns the SUM into a mean over the HW x Cg group slab.
    n_grp = HW * Cg
    inv_sqrt_n_bits = _f32_bits(1.0 / math.sqrt(float(n_grp)))
    eps_bits = _f32_bits(eps)

    in_dtype = input_tensor.dtype
    in_page = input_tensor.buffer_page_size()
    out_page = output_tensor.buffer_page_size()
    bf16_page = ttnn.tile_size(ttnn.bfloat16)

    # Intermediate statistics format (see module docstring).
    if fp32_dest_acc_en or in_dtype == ttnn.float32:
        stat_dtype = ttnn.float32
    else:
        stat_dtype = ttnn.bfloat16
    stat_page = ttnn.tile_size(stat_dtype)

    # --- multi-core split: one work unit = one (n, g) group ---
    grid = input_tensor.device().compute_with_storage_grid_size()
    full_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    num_units = N * G
    (
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_g1,
        units_per_core_g2,
    ) = ttnn.split_work_to_cores(full_grid, num_units)

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
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    stream_pages = 2 * Wg

    cbs = [
        cb(CB_INPUT_TILES, stream_pages, in_page, in_dtype),
        cb(CB_SCALER, 1, bf16_page, ttnn.bfloat16),
        cb(CB_OUTPUT_TILES, stream_pages, out_page, in_dtype),
        cb(CB_MEAN, 1, stat_page, stat_dtype),
        cb(CB_VAR, 1, stat_page, stat_dtype),
        cb(CB_CENTERED, stream_pages, stat_page, stat_dtype),
    ]
    if has_gamma:
        gamma_page = gamma_tensor.buffer_page_size()
        cbs.append(cb(CB_GAMMA_TILES, Wt, gamma_page, gamma_tensor.dtype))
        cbs.append(cb(CB_XHAT, stream_pages, stat_page, stat_dtype))
    if has_beta:
        beta_page = beta_tensor.buffer_page_size()
        cbs.append(cb(CB_BETA_TILES, Wt, beta_page, beta_tensor.dtype))
        cbs.append(cb(CB_SCALED, stream_pages, stat_page, stat_dtype))

    # --- per-core runtime args: [start_group, num_groups_here] + addresses ---
    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    compute_rt_args = ttnn.RuntimeArgs()
    start_group = 0
    for group, units_per_core in (
        (core_group_1, units_per_core_g1),
        (core_group_2, units_per_core_g2),
    ):
        for core_range in group.ranges():
            for x in range(core_range.start.x, core_range.end.x + 1):
                for y in range(core_range.start.y, core_range.end.y + 1):
                    reader_rt_args[x][y] = [
                        input_tensor.buffer_address(),
                        gamma_tensor.buffer_address() if has_gamma else 0,
                        beta_tensor.buffer_address() if has_beta else 0,
                        start_group,
                        units_per_core,
                    ]
                    writer_rt_args[x][y] = [
                        output_tensor.buffer_address(),
                        start_group,
                        units_per_core,
                    ]
                    compute_rt_args[x][y] = [start_group, units_per_core]
                    start_group += units_per_core
    assert start_group == num_units, f"work split mismatch: {start_group} != {num_units}"

    # --- Reader: scalars first, TensorAccessorArgs at the end (input, gamma, beta) ---
    reader_ct_args = [Ht, Wt, Wg, G, int(has_gamma), int(has_beta), inv_sqrt_n_bits]
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

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # --- Writer ---
    writer_ct_args = [Ht, Wt, Wg, G]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # --- Compute ---
    compute_ct_args = [Ht, Wt, Wg, G, int(has_gamma), int(has_beta), eps_bits]
    compute_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=math_approx_mode,
        dst_full_sync_en=dst_full_sync_en,
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "groupnorm_sc_N_1_HW_C_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
