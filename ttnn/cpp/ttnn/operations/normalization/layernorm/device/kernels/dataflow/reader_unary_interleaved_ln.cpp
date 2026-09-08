// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merged reader kernel for layernorm / dit_rms_norm_unary_fused (standard / non-large-tensor path).
//
// Handles both TILE-layout input (default) and ROW_MAJOR input (#ifdef TILIZE_IN).
// The loop structure, scaler/eps generation, gamma/beta reads, and FUSE_PRE_ADD reads
// are shared between both paths. Only the input accessor setup and the per-ncht input
// read call branch on TILIZE_IN.
//
// Replaces the two separate files:
//   - reader_unary_interleaved_ln.cpp          (TILE path, 105 lines)
//   - reader_unary_interleaved_ln_rm_input.cpp (ROW_MAJOR path, 143 lines)
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
#include "ttnn/operations/normalization/layernorm/device/kernels/layernorm_scaler_tiles.h"
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
    // Welford-fp32 alias of dfb_in (non-fused) or dfb_x (fused). Shares SRAM with the
    // primary buffer but has its own read/write pointers, so we must push_back on it whenever we
    // push to the primary buffer. Absent when the alias is inactive: the compute kernel then
    // reads the primary buffer directly and a duplicate push would double-count its semaphore.
#if defined(WELFORD_FP32_ALIAS) && !defined(FUSE_PRE_ADD)
    DataflowBuffer dfb_x_welford(dfb::x_welford);
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

    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto W = get_arg(args::W);

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

#ifdef TILIZE_IN
    // ROW_MAJOR path: input a is a row-major tensor.
    // The compute kernel tilizes dfb_in_rm → dfb_in before processing.
    constexpr auto elem_size_bytes = get_arg(args::elem_size_bytes);

    constexpr uint32_t rm_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    DataflowBuffer dfb_in_rm(dfb::in_rm);

    const uint32_t src0_page_bytes = W * elem_size_bytes;
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

    // Generate constant tiles for layernorm compute
#ifndef USE_WELFORD
    {
        constexpr uint32_t partial_last_tile_cols = W % tt::constants::TILE_WIDTH;
        // Push count shared with the compute kernel's dfb_scaler pop count (issue #48487).
        constexpr uint32_t num_scaler_tiles = norm::layernorm::reduce_scaler_tile_count(W, tt::constants::TILE_WIDTH);

        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();

        if constexpr (num_scaler_tiles == 2) {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                dfb::scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>(partial_last_tile_cols);
        }
    }
#endif

    const uint32_t eps = get_arg(args::eps);
    DataflowBuffer dfb_eps(dfb::eps);
    generate_bcast_col_scalar(dfb_eps, eps);

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        const uint32_t curr_tile_row = start_tile_row + ncht;

        // --- Input read: branches on layout ---
#ifdef TILIZE_IN
        // ROW_MAJOR: push one tile-row of row-major data into dfb_in_rm (block-by-block).
        // The compute kernel's TILIZE_IN block converts dfb_in_rm → dfb_in before processing.
        layernorm_dataflow_utils::push_row_major_blocks_to_dfb<decltype(src_a), TILE_W, TILE_H>(
            noc, dfb_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);

#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_dfb(
                noc, dfb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#else
        // TILE: read input a and b (if present) interleaved per block.
        for (auto block : generic::blocks(Wt, block_size)) {
            const uint32_t flat_offset = curr_tile_row * Wt + block.start();
            layernorm_dataflow_utils::read_block_to_dfb(noc, dfb_in0, src_a, src0_page_bytes, flat_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_dfb(noc, dfb_in1, src_b, src1_tile_bytes, flat_offset, block);
#else
            // Non-fused welford-fp32 alias: dfb_x_welford shares dfb_in0's memory but has its own
            // read/write pointers. After the data lands in dfb_in0, push
            // dfb_x_welford by the same amount so compute can wait_front on the alias separately
            // for welford reads. Absent when no alias is active; the duplicate push would then
            // double-count dfb_in0's semaphore.
#ifdef WELFORD_FP32_ALIAS
            dfb_x_welford.reserve_back(block.full_block_size());
            dfb_x_welford.push_back(block.full_block_size());
#endif
#endif
        }
#endif

        // --- Gamma / beta (shared): read once at ncht == 0 ---
#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, block_size)) {
#ifdef FUSE_GAMMA
                layernorm_dataflow_utils::read_block_to_dfb(
                    noc, dfb_gamma, addrg, gamma_tile_bytes, block.start(), block);
#endif
#ifdef FUSE_BETA
                layernorm_dataflow_utils::read_block_to_dfb(
                    noc, dfb_beta, addrb, beta_tile_bytes, block.start(), block);
#endif
            }  // wt loop
        }
#endif
    }  // ncht loop
}
