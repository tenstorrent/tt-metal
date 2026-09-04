// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
//   block_size
//   W                 (logical width in elements)
//   elem_size_bytes   (TILIZE_IN only; unused for TILE path)
//
// Runtime args:
//   NCHt              (number of tile-rows assigned to this core)
//   Wt                (width in tiles)
//   reader_start      (tile-row index of the first row for this core)
//   eps               (epsilon as bit-cast uint32)
//   H_logical         (TILIZE_IN only: total valid rows; unused for TILE path)
//
// Tensors are bound rather than passed as addresses: tensor::src is the input, tensor::src_b the
// residual, tensor::gamma and tensor::beta the weight and bias.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    const uint32_t NCHt = get_arg(args::NCHt);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t start_tile_row = get_arg(args::reader_start);
#ifdef TILIZE_IN
    const uint32_t H_logical = get_arg(args::H_logical);
#endif

    Noc noc;
#ifndef TILIZE_IN
    // On the ROW_MAJOR path the compute kernel fills dfb_in itself, by tilizing dfb_in_rm, so
    // this kernel has no producer role on it and no binding to construct from.
    DataflowBuffer dfb_in0(dfb::in);
#endif
#ifdef FUSE_PRE_ADD
    DataflowBuffer dfb_in1(dfb::inb);
#endif
#ifdef FUSE_GAMMA
    DataflowBuffer dfb_gamma(dfb::gamma);
#endif
#ifdef FUSE_BETA
    DataflowBuffer dfb_beta(dfb::beta);
#endif

    // No use_welford gate (large-tensor + Welford uses a separate kernel).
    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto W_logical = get_arg(args::W);

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

#ifdef TILIZE_IN
    // ROW_MAJOR path: input a is a row-major tensor.
    // The compute kernel tilizes dfb_in_rm → dfb_in before each pass.
    constexpr auto elem_size_bytes = get_arg(args::elem_size_bytes);

    constexpr uint32_t rm_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    DataflowBuffer dfb_in_rm(dfb::in_rm);

    const uint32_t src0_page_bytes = W_logical * elem_size_bytes;
#else
    // TILE path: input a is already in tile layout.
    const uint32_t src0_page_bytes = dfb_in0.get_tile_size();
#endif

    const auto src_a = TensorAccessor(tensor::src);

#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = dfb_gamma.get_tile_size();
    const auto addrg = TensorAccessor(tensor::gamma);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = dfb_beta.get_tile_size();
    const auto addrb = TensorAccessor(tensor::beta);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = dfb_in1.get_tile_size();
    const auto src_b = TensorAccessor(tensor::src_b);
#endif

    // Generate constant tiles (scaler and epsilon) — shared between TILE and RM paths.
    {
        constexpr uint32_t partial_last_tile_cols = W_logical % tt::constants::TILE_WIDTH;

        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

        if constexpr (partial_last_tile_cols > 0) {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                dfb::scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>(partial_last_tile_cols);
        }
    }
    const uint32_t eps = get_arg(args::eps);
    DataflowBuffer dfb_eps(dfb::eps);
    generate_bcast_col_scalar(dfb_eps, eps);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        const uint32_t curr_tile_row = start_tile_row + ncht;

#ifndef RMSNORM
        // Pass 0: Data for calculating E[X]
#ifdef TILIZE_IN
        layernorm_dataflow_utils::push_row_major_blocks_to_dfb<decltype(src_a), TILE_W, TILE_H>(
            noc, dfb_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);
#else
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in0, src_a, src0_page_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#endif

// Pass 1: Data for calculating variance.
// RM path: push all in_rm first, then all in1 (separate) — the tilize step in
//   compute provides enough pipeline slack.
// TILE path: in0 and in1 MUST be interleaved per block to avoid deadlock.
//   dfb_in0 holds only 2*block_size tiles; filling all in0 before any in1 stalls
//   once the buffer is full while compute waits for in1 — circular wait.
#ifdef TILIZE_IN
        layernorm_dataflow_utils::push_row_major_blocks_to_dfb<decltype(src_a), TILE_W, TILE_H>(
            noc, dfb_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);
#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#else  // TILE path: interleaved per block
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in0, src_a, src0_page_bytes, curr_tile_row * Wt + block.start(), block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
#endif
        }
#endif

        // Pass 2: Data for the final normalization step.
        // For the ROW_MAJOR path, input MUST be interleaved with gamma/beta per block.
        // Pushing all pass-2 input first and then all gamma would deadlock:
        //   dfb_in_rm capacity = 1 block; the compute's normalization loop reads gamma
        //   INSIDE the same per-block loop that drains dfb_in_rm. Once compute finishes
        //   the x/sqrt(var+eps) step it blocks on dfb_gamma, while the reader is still
        //   blocked on dfb_in_rm for the next block — circular wait.
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
            layernorm_dataflow_utils::read_row_major_block_to_dfb<decltype(src_a), decltype(block), TILE_W, TILE_H>(
                noc,
                dfb_in_rm,
                src_a,
                curr_tile_row,
                num_valid_rows_pass2,
                TILE_W * elem_size_bytes,
                rm_row_stride_bytes,
                block);
#else
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in0, src_a, src0_page_bytes, curr_tile_row * Wt + block.start(), block);
#endif

            // Gamma/beta and b-tensor for this block — pushed immediately after input so
            // compute finds them in the buffer when it reaches the per-block multiply step.
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
#endif
#ifdef FUSE_GAMMA
            layernorm_dataflow_utils::read_block_to_dfb(noc, dfb_gamma, addrg, gamma_tile_bytes, block.start(), block);
#endif
#ifdef FUSE_BETA
            layernorm_dataflow_utils::read_block_to_dfb(noc, dfb_beta, addrb, beta_tile_bytes, block.start(), block);
#endif
        }  // wt loop
    }  // ncht loop
}
