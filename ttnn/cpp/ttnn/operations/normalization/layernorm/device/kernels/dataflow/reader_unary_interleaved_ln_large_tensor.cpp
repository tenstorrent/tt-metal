// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Merged reader kernel for the layernorm large-tensor path.
//
// Handles both TILE-layout input (default) and ROW_MAJOR input (#ifdef TILIZE_IN).
// The loop structure (three-pass for non-RMSNORM, two-pass for RMSNORM), gamma/beta
// interleaving in pass 2, and FUSE_PRE_ADD reads are shared between both paths.
// Only the input accessor setup and the per-pass input-read calls branch on TILIZE_IN.
//
// Replaces the two separate files:
//   - reader_unary_interleaved_ln_large_tensor.cpp           (TILE path)
//   - reader_unary_interleaved_ln_large_tensor_rm_input.cpp  (ROW_MAJOR path)
//
// Compile-time args:
//   CTA[0]    = block_size
//   CTA[1..]  = TensorAccessorArgs for input a
//   ...       = TensorAccessorArgs for b / residual  (may be null)
//   ...       = TensorAccessorArgs for gamma          (may be null)
//   ...       = TensorAccessorArgs for beta           (may be null)
//   CTA[last] = elem_size_bytes  (TILIZE_IN only; unused for TILE path)
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
//   arg[9] = W / W_logical     (logical width in elements)
//   arg[10] = H_logical        (TILIZE_IN only: total valid rows; unused for TILE path)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // factory [0]
    const uint32_t NCHt = get_arg_val<uint32_t>(1);        // factory [1]
    const uint32_t Wt = get_arg_val<uint32_t>(2);          // factory [2]
    const uint32_t start_tile_row = get_arg_val<uint32_t>(3); // factory [3]
    // [4] = eps, read below after scaler generation
    const uint32_t gamma_addr = get_arg_val<uint32_t>(5);  // factory [5]
    const uint32_t beta_addr = get_arg_val<uint32_t>(6);   // factory [6]
    const uint32_t b_addr = get_arg_val<uint32_t>(7);      // factory [7]
#ifdef TILIZE_IN
    const uint32_t H_logical = get_arg_val<uint32_t>(8);   // factory [8]
#endif

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_beta = tt::CBIndex::c_6;

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

    // No use_welford slot (large-tensor + Welford uses a separate kernel).
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr auto src0_args = TensorAccessorArgs<1>();
    [[maybe_unused]] constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();
    constexpr uint32_t W = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

#ifdef TILIZE_IN
    // ROW_MAJOR path: input a is a row-major tensor.
    // The compute kernel tilizes cb_in_rm (c_27) → cb_in (c_0) before each pass.
    constexpr uint32_t elem_size_bytes = get_compile_time_arg_val(beta_args.next_compile_time_args_offset());

    constexpr uint32_t rm_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    constexpr uint32_t cb_id_in_rm = tt::CBIndex::c_27;

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

    // Generate constant tiles (scaler and epsilon) — shared between TILE and RM paths.
    {
        // Scaler(s) for reduce: full tile, then optional partial tile for last tile
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        dataflow_kernel_lib::
            calculate_and_prepare_reduce_scaler<cb_in_2, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
        constexpr uint32_t partial_tile_columns = W % tt::constants::TILE_WIDTH;
        if constexpr (partial_tile_columns > 0) {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                cb_in_2,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                partial_tile_columns>();
        }
    }
    constexpr uint32_t eps_cb_id = tt::CBIndex::c_3;
    const uint32_t eps = get_arg_val<uint32_t>(4);          // factory [4]
    generate_bcast_col_scalar(eps_cb_id, eps);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        const uint32_t curr_tile_row = start_tile_row + ncht;

#ifndef RMSNORM
        // Pass 0: Data for calculating E[X]
#ifdef TILIZE_IN
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            cb_id_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);
#else
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in0, src_a, src0_page_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#endif

// Pass 1: Data for calculating variance.
// RM path: push all in_rm first, then all in1 (separate) — the tilize step in
//   compute provides enough pipeline slack.
// TILE path: in0 and in1 MUST be interleaved per block to avoid deadlock.
//   cb_in0 holds only 2*block_size tiles; filling all in0 before any in1 stalls
//   once the buffer is full while compute waits for in1 — circular wait.
#ifdef TILIZE_IN
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            cb_id_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);
#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#else  // TILE path: interleaved per block
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in0, src_a, src0_page_bytes, curr_tile_row * Wt + block.start(), block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
#endif
        }
#endif

        // Pass 2: Data for the final normalization step.
        // For the ROW_MAJOR path, input MUST be interleaved with gamma/beta per block.
        // Pushing all pass-2 input first and then all gamma would deadlock:
        //   cb_in_rm capacity = 1 block; the compute's normalization loop reads gamma
        //   INSIDE the same per-block loop that drains cb_in_rm.  Once compute finishes
        //   the x/sqrt(var+eps) step it blocks on cb_gamma, while the reader is still
        //   blocked on cb_in_rm for the next block — circular wait.
        // For the TILE path the same block-interleaved order is used for consistency.
#ifdef TILIZE_IN
        const uint32_t abs_row_base = curr_tile_row * TILE_H;
        uint32_t num_valid_rows_pass2 = TILE_H;
        if (abs_row_base >= H_logical) {
            num_valid_rows_pass2 = 0;
        } else if (H_logical - abs_row_base < TILE_H) {
            num_valid_rows_pass2 = H_logical - abs_row_base;
        }
#endif

        for (auto block : generic::blocks(Wt, block_size)) {
            // Pass 2 input for this block
#ifdef TILIZE_IN
            layernorm_dataflow_utils::read_row_major_block_to_cb<decltype(src_a), decltype(block), TILE_W, TILE_H>(
                cb_id_in_rm,
                src_a,
                curr_tile_row,
                num_valid_rows_pass2,
                TILE_W * elem_size_bytes,
                rm_row_stride_bytes,
                block);
#else
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in0, src_a, src0_page_bytes, curr_tile_row * Wt + block.start(), block);
#endif

            // Gamma/beta and b-tensor for this block — pushed immediately after input so
            // compute finds them in the CB when it reaches the per-block multiply step.
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
#endif
#ifdef FUSE_GAMMA
            layernorm_dataflow_utils::read_block_to_cb(noc, cb_gamma, addrg, gamma_tile_bytes, block.start(), block);
#endif
#ifdef FUSE_BETA
            layernorm_dataflow_utils::read_block_to_cb(noc, cb_beta, addrb, beta_tile_bytes, block.start(), block);
#endif
        }  // wt loop
    }  // ncht loop
}
