// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Merged reader kernel for layernorm / dit_rms_norm_unary_fused (standard / non-large-tensor path).
//
// Handles both TILE-layout input (default) and ROW_MAJOR input (#ifdef TILIZE_IN).
// The loop structure, scaler/eps generation, gamma/beta reads, and FUSE_PRE_ADD reads
// are shared between both paths.  Only the input accessor setup and the per-ncht input
// read call branch on TILIZE_IN.
//
// Replaces the two separate files:
//   - reader_unary_interleaved_ln.cpp          (TILE path, 105 lines)
//   - reader_unary_interleaved_ln_rm_input.cpp (ROW_MAJOR path, 143 lines)
//
// Compile-time args:
//   CTA[0]    = block_size
//   CTA[1]    = use_welford  (0 for TILIZE_IN / RMSNORM)
//   CTA[2..]  = TensorAccessorArgs for input a
//   ...       = TensorAccessorArgs for b / residual  (may be null)
//   ...       = TensorAccessorArgs for gamma          (may be null)
//   ...       = TensorAccessorArgs for beta           (may be null)
//   CTA[N]    = W                (logical width in elements)
//   CTA[N+1]  = elem_size_bytes  (TILIZE_IN only; unused for TILE path)
//
// Runtime args:
//   arg[0] = src_addr
//   arg[1] = NCHt              (number of tile-rows assigned to this core)
//   arg[2] = Wt                (width in tiles)
//   arg[3] = start_tile_row    (tile-row index of first row for this core)
//                              TILE:  previously passed as tile_offset = start_tile_row * Wt
//                              RM:    previously passed as start_row;  start_tile_row = start_row / TILE_H
//   arg[4] = packed_one_value  (scaler value for reduce)
//   arg[5] = eps               (epsilon as bit-cast uint32)
//   arg[6] = gamma_dram_addr
//   arg[7] = beta_dram_addr
//   arg[8] = b_dram_addr       (residual, unused if no FUSE_PRE_ADD)
//   arg[9] = W                 (width in elements)
//   arg[10] = H_logical        (TILIZE_IN only: total valid rows; unused for TILE path)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/kernel_util/dataflow/custom_tiles.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t NCHt = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t start_tile_row = get_arg_val<uint32_t>(3);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t beta_addr = get_arg_val<uint32_t>(7);
    const uint32_t b_addr = get_arg_val<uint32_t>(8);
    const uint32_t W = get_arg_val<uint32_t>(9);
    const uint32_t tile_width = get_arg_val<uint32_t>(10);
    const uint32_t tile_height = get_arg_val<uint32_t>(11);
#ifdef TILIZE_IN
    const uint32_t H_logical = get_arg_val<uint32_t>(12);
#endif

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in");
    constexpr uint32_t cb_id_in1 = get_named_compile_time_arg_val("cb_inb");
    constexpr uint32_t cb_id_gamma = get_named_compile_time_arg_val("cb_gamma");
    constexpr uint32_t cb_id_beta = get_named_compile_time_arg_val("cb_beta");

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);
#ifdef FUSE_PRE_ADD
    experimental::CircularBuffer cb_in1(cb_id_in1);
#endif
#ifdef FUSE_GAMMA
    experimental::CircularBuffer cb_gamma(cb_id_gamma);
#endif
#ifdef FUSE_BETA
    experimental::CircularBuffer cb_beta(cb_id_beta);
#endif

    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr bool use_welford = get_compile_time_arg_val(1) == 1;
    constexpr auto src0_args = TensorAccessorArgs<2>();
    [[maybe_unused]] constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

#ifdef TILIZE_IN
    // ROW_MAJOR path: input a is a row-major tensor.
    // The compute kernel tilizes cb_in_rm (c_27) → cb_in (c_0) before processing.
    constexpr uint32_t elem_size_bytes = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());

    constexpr uint32_t rm_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    constexpr uint32_t cb_id_in_rm = get_named_compile_time_arg_val("cb_in_rm");

    const uint32_t src0_page_bytes = W * elem_size_bytes;
#else
    // TILE path: input a is already in tile layout.
    const uint32_t src0_page_bytes = get_tile_size(cb_id_in0);
#endif

    const auto src_a = TensorAccessor(src0_args, src_addr, src0_page_bytes);

#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = get_tile_size(cb_id_gamma);
    const auto addrg = TensorAccessor(gamma_args, gamma_addr, gamma_tile_bytes);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = get_tile_size(cb_id_beta);
    const auto addrb = TensorAccessor(beta_args, beta_addr, beta_tile_bytes);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = get_tile_size(cb_id_in1);
    const auto src_b = TensorAccessor(src1_args, b_addr, src1_tile_bytes);
#endif

    // Generate constant tiles for layernorm compute
    constexpr uint32_t cb_scaler = get_named_compile_time_arg_val("cb_scaler");
    constexpr uint32_t cb_eps = get_named_compile_time_arg_val("cb_eps");

    if constexpr (!use_welford) {
        // Scaler(s) for reduce
        uint32_t scaler = get_arg_val<uint32_t>(4);
        generate_reduce_scaler(cb_scaler, scaler);
        const auto partial_last_tile_cols = W % tile_width;
        if (partial_last_tile_cols > 0 && !use_welford) {
            norm::kernel_util::dataflow::generate_partial_reduce_scaler(
                cb_scaler, scaler, partial_last_tile_cols, tile_height, tile_width);
        }
    }

    const uint32_t eps = get_arg_val<uint32_t>(5);
    generate_bcast_col_scalar(cb_eps, eps);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        const uint32_t curr_tile_row = start_tile_row + ncht;

        // --- Input read: branches on layout ---
#ifdef TILIZE_IN
        // ROW_MAJOR: push one tile-row of row-major data into cb_in_rm (block-by-block).
        // The compute kernel's TILIZE_IN block converts cb_in_rm → cb_in before processing.
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            cb_id_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);

#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#else
        // TILE: read input a and b (if present) interleaved per block.
        for (auto block : generic::blocks(Wt, block_size)) {
            const uint32_t flat_offset = curr_tile_row * Wt + block.start();
            layernorm_dataflow_utils::read_block_to_cb(noc, cb_in0, src_a, src0_page_bytes, flat_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(noc, cb_in1, src_b, src1_tile_bytes, flat_offset, block);
#endif
        }
#endif

        // --- Gamma / beta (shared): read once at ncht == 0 ---
#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, block_size)) {
#ifdef FUSE_GAMMA
                layernorm_dataflow_utils::read_block_to_cb(noc, cb_gamma, addrg, gamma_tile_bytes, block.start(), block);
#endif
#ifdef FUSE_BETA
                layernorm_dataflow_utils::read_block_to_cb(noc, cb_beta, addrb, beta_tile_bytes, block.start(), block);
#endif
            }  // wt loop
        }
#endif
    }  // ncht loop
}
