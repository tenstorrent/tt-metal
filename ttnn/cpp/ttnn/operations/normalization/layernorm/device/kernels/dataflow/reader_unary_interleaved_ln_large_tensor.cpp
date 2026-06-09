// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merged reader kernel for the layernorm large-tensor path.
//
// Handles both TILE-layout input (default) and ROW_MAJOR input (#ifdef TILIZE_IN).

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
    auto NCHt = get_arg(args::NCHt);
    auto Wt = get_arg(args::Wt);
    auto start_tile_row = get_arg(args::start_tile_row);
#ifdef TILIZE_IN
    auto H_logical = get_arg(args::H_logical);
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

    constexpr auto block_size = get_arg(args::block_size);
    constexpr auto W_logical = get_arg(args::W);

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;

#ifdef TILIZE_IN
    constexpr auto elem_size_bytes = get_arg(args::elem_size_bytes);
    constexpr uint32_t rm_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    DataflowBuffer cb_in_rm(dfb::cb_in_rm);
    const uint32_t src0_page_bytes = W_logical * elem_size_bytes;
#else
    const uint32_t src0_page_bytes = get_tile_size(dfb::cb_in);
#endif

    const auto src_a = TensorAccessor(ta::src_a);

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
    const auto src_b = TensorAccessor(ta::src_b);
#endif

    {
        constexpr uint32_t partial_last_tile_cols = W_logical % tt::constants::TILE_WIDTH;

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

#ifndef RMSNORM
        // Pass 0: Data for calculating E[X]
#ifdef TILIZE_IN
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            noc, cb_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);
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

// Pass 1
#ifdef TILIZE_IN
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            noc, cb_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);
#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
        }
#endif
#else
        for (auto block : generic::blocks(Wt, block_size)) {
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in0, src_a, src0_page_bytes, curr_tile_row * Wt + block.start(), block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
#endif
        }
#endif

        // Pass 2
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
#ifdef TILIZE_IN
            layernorm_dataflow_utils::read_row_major_block_to_cb<decltype(src_a), decltype(block), TILE_W, TILE_H>(
                noc,
                cb_in_rm,
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
        }
    }
}
