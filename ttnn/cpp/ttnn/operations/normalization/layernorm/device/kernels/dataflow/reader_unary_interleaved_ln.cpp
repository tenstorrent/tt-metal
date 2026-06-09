// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Merged reader kernel for layernorm / dit_rms_norm_unary_fused (standard / non-large-tensor path).
//
// Handles both TILE-layout input (default) and ROW_MAJOR input (#ifdef TILIZE_IN).
// The loop structure, scaler/eps generation, gamma/beta reads, and FUSE_PRE_ADD reads
// are shared between both paths.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "layernorm_dataflow_utils.h"

namespace generic = norm::kernel_util::generic;
namespace layernorm_dataflow_utils = norm::layernorm::device::kernels::dataflow;

void kernel_main() {
    WAYPOINT("LR0");
    DPRINT("[reader_unary_interleaved_ln] START\n");

    auto NCHt = get_arg(args::NCHt);
    auto Wt = get_arg(args::Wt);
    auto start_tile_row = get_arg(args::start_tile_row);
#ifdef TILIZE_IN
    auto H_logical = get_arg(args::H_logical);
#endif

    DPRINT("[reader_unary_interleaved_ln] START2\n");
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
    DPRINT("[reader_unary_interleaved_ln] START3\n");

    constexpr auto block_size = get_arg(args::block_size);
    constexpr bool use_welford = get_arg(args::use_welford) == 1;
    constexpr auto W = get_arg(args::W);

    constexpr uint32_t TILE_H = tt::constants::TILE_HEIGHT;
    constexpr uint32_t TILE_W = tt::constants::TILE_WIDTH;
    DPRINT("[reader_unary_interleaved_ln] START4\n");

#ifdef TILIZE_IN
    constexpr auto elem_size_bytes = get_arg(args::elem_size_bytes);
    constexpr uint32_t rm_row_stride_bytes = block_size * TILE_W * elem_size_bytes;
    DataflowBuffer cb_in_rm(dfb::cb_in_rm);
    const uint32_t src0_page_bytes = W * elem_size_bytes;
#else
    const uint32_t src0_page_bytes = get_tile_size(dfb::cb_in);
#endif
    DPRINT("[reader_unary_interleaved_ln] START5\n");

    const auto src_a = TensorAccessor(ta::src_a);
    DPRINT("[reader_unary_interleaved_ln] START6\n");

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
    DPRINT("[reader_unary_interleaved_ln] AFTER START\n");

#ifndef USE_WELFORD
    {
        constexpr uint32_t partial_last_tile_cols = W % tt::constants::TILE_WIDTH;

        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::cb_scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR,
            /*compute_uses_reduce_tile=*/true>();
        DPRINT("[reader_unary_interleaved_ln] AFTER REDUCE SCALER\n");

        if constexpr (partial_last_tile_cols > 0) {
            dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                dfb::cb_scaler,
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR,
                /*compute_uses_reduce_tile=*/true>(partial_last_tile_cols);
        }
        DPRINT("[reader_unary_interleaved_ln] AFTER SECOND REDUCE SCALER\n");
    }
#endif

    const uint32_t eps = get_arg(args::eps);
    DPRINT("[reader_unary_interleaved_ln] BEFORE BCAST SCALER\n");
    generate_bcast_col_scalar(dfb::cb_eps, eps);
    DPRINT("[reader_unary_interleaved_ln] AFTER BCAST SCALER\n");
    WAYPOINT("LR1");

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        WAYPOINT("LR2");
        DPRINT("[reader_unary_interleaved_ln] LOOP_BEGIN ncht ncht={} NCHt={}\n", ncht, NCHt);
        const uint32_t curr_tile_row = start_tile_row + ncht;

#ifdef TILIZE_IN
        DPRINT("[reader_unary_interleaved_ln] BEFORE push_row_major_blocks_to_cb ncht={}\n", ncht);
        layernorm_dataflow_utils::push_row_major_blocks_to_cb<decltype(src_a), TILE_W, TILE_H>(
            noc, cb_in_rm, src_a, Wt, block_size, curr_tile_row, elem_size_bytes, rm_row_stride_bytes, H_logical);
        DPRINT("[reader_unary_interleaved_ln] AFTER push_row_major_blocks_to_cb ncht={}\n", ncht);

#ifdef FUSE_PRE_ADD
        for (auto block : generic::blocks(Wt, block_size)) {
            DPRINT(
                "[reader_unary_interleaved_ln] LOOP_BEGIN fuse_pre_add_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
            layernorm_dataflow_utils::read_block_to_cb(
                noc, cb_in1, src_b, src1_tile_bytes, curr_tile_row * Wt + block.start(), block);
            DPRINT(
                "[reader_unary_interleaved_ln] LOOP_END fuse_pre_add_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
        }
#endif
#else
        for (auto block : generic::blocks(Wt, block_size)) {
            DPRINT(
                "[reader_unary_interleaved_ln] LOOP_BEGIN input_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
            const uint32_t flat_offset = curr_tile_row * Wt + block.start();
            layernorm_dataflow_utils::read_block_to_cb(noc, cb_in0, src_a, src0_page_bytes, flat_offset, block);
#ifdef FUSE_PRE_ADD
            layernorm_dataflow_utils::read_block_to_cb(noc, cb_in1, src_b, src1_tile_bytes, flat_offset, block);
#endif
            DPRINT(
                "[reader_unary_interleaved_ln] LOOP_END input_block ncht={} start={} size={} full={}\n",
                ncht,
                block.start(),
                block.size(),
                block.full_block_size());
        }
#endif

#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, block_size)) {
                DPRINT(
                    "[reader_unary_interleaved_ln] LOOP_BEGIN gamma_beta_block ncht={} start={} size={} full={}\n",
                    ncht,
                    block.start(),
                    block.size(),
                    block.full_block_size());
#ifdef FUSE_GAMMA
                layernorm_dataflow_utils::read_block_to_cb(noc, cb_gamma, addrg, gamma_tile_bytes, block.start(), block);
#endif
#ifdef FUSE_BETA
                layernorm_dataflow_utils::read_block_to_cb(noc, cb_beta, addrb, beta_tile_bytes, block.start(), block);
#endif
                DPRINT(
                    "[reader_unary_interleaved_ln] LOOP_END gamma_beta_block ncht={} start={} size={} full={}\n",
                    ncht,
                    block.start(),
                    block.size(),
                    block.full_block_size());
            }
        }
#endif
        DPRINT("[reader_unary_interleaved_ln] LOOP_END ncht ncht={} NCHt={}\n", ncht, NCHt);
    }
}
