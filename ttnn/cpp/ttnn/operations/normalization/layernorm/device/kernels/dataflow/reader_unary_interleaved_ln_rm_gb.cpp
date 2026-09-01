// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "ttnn/operations/normalization/kernel_util/generic/blocked_range.h"
#include "ttnn/operations/normalization/layernorm/device/kernels/layernorm_scaler_tiles.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"

namespace generic = norm::kernel_util::generic;

void kernel_main() {
    uint32_t NCHt = get_arg(args::NCHt);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t tile_offset = get_arg(args::reader_start);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in);
    // Welford-fp32 alias of dfb_in (non-fused) or dfb_x (fused). Shares SRAM with the
    // primary buffer but has its own read/write pointers, so we must push_back on it whenever we
    // push to the primary buffer. Absent when the alias is inactive: compute then reads the
    // primary buffer itself, and a duplicate push would double-count its semaphore.
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

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = dfb_in0.get_tile_size();

    constexpr auto blk = get_arg(args::block_size);  // needed for correctness of softmax/LN kernels
    constexpr auto W = get_arg(args::W);

    const auto src_a = TensorAccessor(tensor::src);

    // Byte offsets within a tile scale with the datum size (2B for bf16, 4B for fp32):
    //   row_bytes      = one tile-width row  = TILE_WIDTH (32) datums
    //   face_bytes     = one 16x16 tile face = FACE_HW (256) datums; face 1 starts here within the tile
    //   half_row_bytes = first FACE_WIDTH (16) datums of a row = the face boundary in a row-major stick
#ifdef FUSE_GAMMA
    const uint32_t gamma_tile_bytes = dfb_gamma.get_tile_size();
    const uint32_t gamma_datum_bytes = gamma_tile_bytes / tt::constants::TILE_HW;
    const uint32_t gamma_row_bytes = tt::constants::TILE_WIDTH * gamma_datum_bytes;
    const uint32_t gamma_face_bytes = tt::constants::FACE_HW * gamma_datum_bytes;
    const uint32_t gamma_half_row_bytes = tt::constants::FACE_WIDTH * gamma_datum_bytes;
    const auto addrg = TensorAccessor(tensor::gamma);
#endif
#ifdef FUSE_BETA
    const uint32_t beta_tile_bytes = dfb_beta.get_tile_size();
    const uint32_t beta_datum_bytes = beta_tile_bytes / tt::constants::TILE_HW;
    const uint32_t beta_row_bytes = tt::constants::TILE_WIDTH * beta_datum_bytes;
    const uint32_t beta_face_bytes = tt::constants::FACE_HW * beta_datum_bytes;
    const uint32_t beta_half_row_bytes = tt::constants::FACE_WIDTH * beta_datum_bytes;
    const auto addrb = TensorAccessor(tensor::beta);
#endif
#ifdef FUSE_PRE_ADD
    const uint32_t src1_tile_bytes = dfb_in1.get_tile_size();
    const auto src_b = TensorAccessor(tensor::src_b);
#endif

    // Generate constant tiles for layernorm compute
#ifndef USE_WELFORD
    {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            dfb::scaler,
            ckernel::PoolType::SUM,
            ckernel::ReduceDim::REDUCE_ROW,
            dataflow_kernel_lib::SUM_AND_MAX_REDUCE_FACTOR>();
        // Push count shared with the compute kernel's dfb_scaler pop count (issue #48487).
        constexpr uint32_t partial_last_tile_cols = W % tt::constants::TILE_WIDTH;
        constexpr uint32_t num_scaler_tiles = norm::layernorm::reduce_scaler_tile_count(W, tt::constants::TILE_WIDTH);
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

    // read a ublock of tiles from src to the input buffer, and then push the ublock to unpacker
    uint32_t offs = 0;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        for (auto block : generic::blocks(Wt, blk)) {
            dfb_in0.reserve_back(block.full_block_size());
            uint32_t idx = 0;
            for (auto r : block.local()) {
                noc.async_read(
                    src_a,
                    dfb_in0,
                    src0_tile_bytes,
                    {.page_id = offs + block.start() + r + tile_offset},
                    {.offset_bytes = idx * src0_tile_bytes});
                idx++;
            }
            noc.async_read_barrier();
            dfb_in0.push_back(block.full_block_size());

#ifdef FUSE_PRE_ADD
            dfb_in1.reserve_back(block.full_block_size());
            idx = 0;
            for (auto r : block.local()) {
                noc.async_read(
                    src_b,
                    dfb_in1,
                    src1_tile_bytes,
                    {.page_id = offs + block.start() + r + tile_offset},
                    {.offset_bytes = idx * src1_tile_bytes});
                idx++;
            }
            noc.async_read_barrier();
            dfb_in1.push_back(block.full_block_size());
#else
            // Non-fused welford-fp32 alias: dfb_x_welford shares dfb_in0's memory but has its own
            // read/write pointers. After the data lands in dfb_in0, push dfb_x_welford by the same
            // amount so compute can wait_front on the alias separately for welford reads. Absent
            // when no alias is active; the duplicate push would then double-count dfb_in0's
            // semaphore.
#ifdef WELFORD_FP32_ALIAS
            dfb_x_welford.reserve_back(block.full_block_size());
            dfb_x_welford.push_back(block.full_block_size());
#endif
#endif
        }  // wt loop

#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (auto block : generic::blocks(Wt, blk)) {
#ifdef FUSE_GAMMA
                {
                    dfb_gamma.reserve_back(block.full_block_size());
                    UnicastEndpoint local_ep;
                    uint32_t idx = 0;
                    for (auto r : block.local()) {
                        noc.async_read(
                            addrg,
                            dfb_gamma,
                            gamma_row_bytes,
                            {.page_id = block.start() + r},
                            {.offset_bytes = idx * gamma_tile_bytes});
                        noc.async_read_barrier();
                        noc.async_read(
                            local_ep,
                            dfb_gamma,
                            gamma_half_row_bytes,
                            {.noc_x = my_x[noc.get_noc_id()],
                             .noc_y = my_y[noc.get_noc_id()],
                             .addr = dfb_gamma.get_write_ptr() + idx * gamma_tile_bytes + gamma_half_row_bytes},
                            {.offset_bytes = idx * gamma_tile_bytes + gamma_face_bytes});
                        idx++;
                    }
                    noc.async_read_barrier();
                    dfb_gamma.push_back(block.full_block_size());
                }
#endif

#ifdef FUSE_BETA
                {
                    dfb_beta.reserve_back(block.full_block_size());
                    UnicastEndpoint local_ep;
                    uint32_t idx = 0;
                    for (auto r : block.local()) {
                        noc.async_read(
                            addrb,
                            dfb_beta,
                            beta_row_bytes,
                            {.page_id = block.start() + r},
                            {.offset_bytes = idx * beta_tile_bytes});
                        noc.async_read_barrier();
                        noc.async_read(
                            local_ep,
                            dfb_beta,
                            beta_half_row_bytes,
                            {.noc_x = my_x[noc.get_noc_id()],
                             .noc_y = my_y[noc.get_noc_id()],
                             .addr = dfb_beta.get_write_ptr() + idx * beta_tile_bytes + beta_half_row_bytes},
                            {.offset_bytes = idx * beta_tile_bytes + beta_face_bytes});
                        idx++;
                    }
                    noc.async_read_barrier();
                    dfb_beta.push_back(block.full_block_size());
                }
#endif
            }  // wt loop
        }
#endif
        offs += Wt;
    }  // ncht loop
}
