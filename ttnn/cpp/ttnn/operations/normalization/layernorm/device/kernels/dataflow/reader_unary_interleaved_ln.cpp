// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
// Metal 2.0 bindings (see the program factory):
//   DFBs   : cb_in (produced; ROW_MAJOR fills cb_in_rm instead), cb_inb (FUSE_PRE_ADD), cb_gamma,
//            cb_beta, cb_scaler (!use_welford), cb_eps, cb_in_rm (TILIZE_IN),
//            cb_x_welford (non-fused TILE welford-fp32 alias, #ifdef WELFORD_FP32_ALIAS)
//   Tensors: input (a), residual (b, FUSE_PRE_ADD), gamma (FUSE_GAMMA), beta (FUSE_BETA)
//   CTAs    : block_size, use_welford, W, elem_size_bytes (TILIZE_IN)
//   RTAs    : NCHt, Wt, start_tile_row, eps, H_logical (TILIZE_IN)
//
// The legacy buffer-address RTAs (a/gamma/beta/b) are now TensorBindings (TensorAccessor(ta::*));
// the legacy dead packed_one_value RTA is dropped.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    const uint32_t NCHt = get_arg(args::NCHt);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t start_tile_row = get_arg(args::start_tile_row);
#ifdef TILIZE_IN
    const uint32_t H_logical = get_arg(args::H_logical);
#endif

    Noc noc;
    DataflowBuffer cb_in0(dfb::cb_in);
#ifdef FUSE_PRE_ADD
    DataflowBuffer cb_in1(dfb::cb_inb);
#endif
#ifdef FUSE_GAMMA
    DataflowBuffer cb_gamma(dfb::cb_gamma);
#endif
#ifdef FUSE_BETA
    DataflowBuffer cb_beta(dfb::cb_beta);
#endif

    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr bool use_welford = get_arg(args::use_welford) == 1;
    constexpr uint32_t W = get_arg(args::W);

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

#ifdef TILIZE_IN
    // ROW_MAJOR path: input a is a row-major tensor.
    // The compute kernel tilizes cb_in_rm (c_27) → cb_in (c_0) before processing.
    constexpr uint32_t elem_size_bytes = get_arg(args::elem_size_bytes);

    constexpr uint32_t rm_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    DataflowBuffer cb_in_rm(dfb::cb_in_rm);

    const uint32_t src0_page_bytes = W * elem_size_bytes;
#else
    // TILE path: input a is already in tile layout.
    const uint32_t src0_page_bytes = get_tile_size(dfb::cb_in);
#endif

    const auto src_a = TensorAccessor(ta::input);

#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = get_tile_size(dfb::cb_gamma);
    const auto addrg = TensorAccessor(ta::gamma);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = get_tile_size(dfb::cb_beta);
    const auto addrb = TensorAccessor(ta::beta);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = get_tile_size(dfb::cb_inb);
    const auto src_b = TensorAccessor(ta::residual);
#endif

    // Generate constant tiles for layernorm compute
    if constexpr (!use_welford) {
        constexpr uint32_t partial_last_tile_cols = W % tt::constants::TILE_WIDTH;

        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR,
            /*compute_uses_reduce_tile=*/true>();

        if constexpr (partial_last_tile_cols > 0) {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                dfb::cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR,
                /*compute_uses_reduce_tile=*/true>(partial_last_tile_cols);
        }
    }

    const uint32_t eps = get_arg(args::eps);
    generate_bcast_col_scalar(dfb::cb_eps, eps);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        const uint32_t curr_tile_row = start_tile_row + ncht;

        // --- Input read: branches on layout ---
#ifdef TILIZE_IN
        // ROW_MAJOR: push one tile-row of row-major data into cb_in_rm (block-by-block).
        // The compute kernel's TILIZE_IN block converts cb_in_rm → cb_in before processing.
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            noc, cb_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);

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
#else
            // Non-fused welford-fp32 alias: cb_x_welford shares cb_in0's memory but has its own
            // read/write pointers. After the data lands in cb_in0, push cb_x_welford by the same
            // amount so compute can wait_front on the alias separately for welford reads. The alias
            // DFB is bound (and WELFORD_FP32_ALIAS defined) only on this non-fused TILE path; in the
            // unaliased build the gate is off and cb_x_welford is not referenced.
#ifdef WELFORD_FP32_ALIAS
            DataflowBuffer cb_x_welford(dfb::cb_x_welford);
            cb_x_welford.reserve_back(block.full_block_size());
            cb_x_welford.push_back(block.full_block_size());
#endif
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
