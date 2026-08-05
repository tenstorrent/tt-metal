// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel reads the layernorm inputs, per device statistics, and gamma, beta, epsilon from interleaved dram.
 */

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/debug/assert.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

template <uint32_t t, typename AccessorType>
void async_read_row_to_tile(
    const Noc& noc, const AccessorType& accessor, uint32_t page_id, uint32_t L1_dst_addr, uint32_t datum_bytes) {
    // Byte offsets within a tile scale with the datum size (2B for bf16, 4B for fp32):
    //   row_bytes  = one tile-width row  = TILE_WIDTH (32) datums
    //   face_bytes = one 16x16 tile face = FACE_HW (256) datums (face 1 starts here within the tile)
    //   half_row   = first FACE_WIDTH (16) datums of the row (face boundary inside a row-major stick)
    const uint32_t row_bytes = tt::constants::TILE_WIDTH * datum_bytes;
    const uint32_t face_bytes = tt::constants::FACE_HW * datum_bytes;
    const uint32_t half_row_bytes = tt::constants::FACE_WIDTH * datum_bytes;
    // Read one full row (32 datums) from the start of the page
    noc.async_read(accessor, CoreLocalMem<uint32_t>(L1_dst_addr), row_bytes, {.page_id = page_id}, {});

    if constexpr (t == 0) {  // TILE LAYOUT
        // Read the second tile face from its offset within the same page
        noc.async_read(
            accessor,
            CoreLocalMem<uint32_t>(L1_dst_addr + face_bytes),
            row_bytes,
            {.page_id = page_id, .offset_bytes = face_bytes},
            {});
    } else if constexpr (t == 1) {  // ROW MAJOR LAYOUT
        noc.async_read_barrier();
        // L1→L1 copy: move the second half of the row (datums 16..31) into the second face
        UnicastEndpoint self;
        noc.async_read(
            self,
            CoreLocalMem<uint32_t>(L1_dst_addr + face_bytes),
            row_bytes,
            {.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = L1_dst_addr + half_row_bytes},
            {});
    } else {
        static_assert(t == 0 || t == 1, "Layout must be ROW_MAJOR(t == 1) or TILE_LAYOUT(t == 0)");
    }
}

void kernel_main() {
    const auto NCHt = get_arg(args::NCHt);                // Number of NCH tiles
    const auto tile_offset = get_arg(args::tile_offset);  // Tile offset for this core
    // Tile offset for stats input; the stats input is two tiles wide and contains E(x) and
    // E(x^2) in the left most columns per tile.
    const auto stats_tile_offset = get_arg(args::stats_tile_offset);
    const auto y_offset = get_arg(args::y_offset);

    constexpr auto blk = get_arg(args::blk);
    constexpr auto stats_tiles_cols = get_arg(args::stats_tiles_cols);
    constexpr auto gamma_is_row_major = get_arg(args::gamma_is_row_major);
    constexpr auto beta_is_row_major = get_arg(args::beta_is_row_major);
    constexpr auto dfb_length = get_arg(args::dfb_length);
    constexpr auto Wt = get_arg(args::Wt);  // Width in tiles
    constexpr auto reduce_factor = get_arg(args::reduce_factor);

    const auto src_a = TensorAccessor(tensor::src);
    const auto src_stats = TensorAccessor(tensor::stats_src);

    Noc noc;
    // Input tiles and the per-device statistics, both consumed by the compute kernel.
    DataflowBuffer dfb_inp_buf(dfb::inp);
    DataflowBuffer dfb_stats_buf(dfb::stats);

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = dfb_inp_buf.get_tile_size();
    const uint32_t stats_tile_bytes = dfb_stats_buf.get_tile_size();

#ifdef FUSE_GAMMA
    const auto addrg = TensorAccessor(tensor::gamma_src);
    DataflowBuffer dfb_gamma_buf(dfb::gamma);
    // datum size (bytes) of gamma, derived from its tile size (TILE_HW = 32*32 = 1024 datums/tile).
    // Used to scale the row/face byte offsets when packing a stick into tile layout (bf16=2B, fp32=4B).
    const uint32_t gamma_datum_bytes = dfb_gamma_buf.get_tile_size() / tt::constants::TILE_HW;
#endif
#ifdef FUSE_BETA
    const auto addrb = TensorAccessor(tensor::beta_src);
    DataflowBuffer dfb_beta_buf(dfb::beta);
    const uint32_t beta_datum_bytes = dfb_beta_buf.get_tile_size() / tt::constants::TILE_HW;
#endif

    // Generate constant tiles for layernorm compute
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        dfb::reduce,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        reduce_factor>();
    const auto eps = get_arg(args::eps);
    // generate_bcast_col_scalar is a shared kernel-pool helper that still takes a CircularBuffer by
    // value, so the handle is wrapped here at the call site rather than passed as a DataflowBuffer.
    generate_bcast_col_scalar(CircularBuffer(dfb::eps), eps);

    uint32_t inp_tile_idx = tile_offset;
    uint32_t stats_tile_idx = stats_tile_offset;

    constexpr uint32_t dfb_iterations = Wt / dfb_length;
    constexpr uint32_t dfb_leftovers = Wt % dfb_length;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Read stats tiles
        dfb_stats_buf.reserve_back(stats_tiles_cols);
        uint32_t stats_write_offset = 0;
        for (uint32_t st = 0; st < stats_tiles_cols; ++st) {
            noc.async_read(
                src_stats,
                dfb_stats_buf,
                stats_tile_bytes,
                {.page_id = stats_tile_idx},
                {.offset_bytes = stats_write_offset});
            stats_write_offset += stats_tile_bytes;
            stats_tile_idx++;
        }
        noc.async_read_barrier();
        dfb_stats_buf.push_back(stats_tiles_cols);
        // In the 2D-core-grid path each core handles only a horizontal slice [y_offset, y_offset + Wt)
        // of the gamma/beta tensors. The 1D path passes y_offset == 0 so this is a no-op there.
        uint32_t gamma_tile_count = y_offset;
        uint32_t beta_tile_count = y_offset;
        for (uint32_t i = 0; i < dfb_iterations; i++) {
            for (uint32_t j = 0; j < dfb_length; j++) {
                dfb_inp_buf.reserve_back(1);
                noc.async_read(src_a, dfb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
                inp_tile_idx++;
                noc.async_read_barrier();
                dfb_inp_buf.push_back(1);
            }
            if (ncht == 0 or dfb_iterations != 1) {
#if defined FUSE_GAMMA || defined FUSE_BETA
#ifdef FUSE_GAMMA
                for (uint32_t j = 0; j < dfb_length; j++) {
                    dfb_gamma_buf.reserve_back(1);
                    uint32_t l1_write_addr = dfb_gamma_buf.get_write_ptr();
                    async_read_row_to_tile<gamma_is_row_major>(
                        noc, addrg, gamma_tile_count, l1_write_addr, gamma_datum_bytes);
                    gamma_tile_count++;
                    noc.async_read_barrier();
                    dfb_gamma_buf.push_back(1);
                }
#endif
#ifdef FUSE_BETA
                for (uint32_t j = 0; j < dfb_length; j++) {
                    dfb_beta_buf.reserve_back(1);
                    uint32_t l1_write_addr = dfb_beta_buf.get_write_ptr();
                    async_read_row_to_tile<beta_is_row_major>(
                        noc, addrb, beta_tile_count, l1_write_addr, beta_datum_bytes);
                    beta_tile_count++;
                    noc.async_read_barrier();
                    dfb_beta_buf.push_back(1);
                }
#endif
#endif
            }
        }
        for (uint32_t i = 0; i < dfb_leftovers; i++) {
            dfb_inp_buf.reserve_back(1);
            noc.async_read(src_a, dfb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
            inp_tile_idx++;
            noc.async_read_barrier();
            dfb_inp_buf.push_back(1);
        }
        if (ncht == 0 or dfb_iterations != 1) {
#if defined FUSE_GAMMA || defined FUSE_BETA
#ifdef FUSE_GAMMA
            for (uint32_t i = 0; i < dfb_leftovers; i++) {
                dfb_gamma_buf.reserve_back(1);
                uint32_t l1_write_addr = dfb_gamma_buf.get_write_ptr();
                async_read_row_to_tile<gamma_is_row_major>(
                    noc, addrg, gamma_tile_count, l1_write_addr, gamma_datum_bytes);
                gamma_tile_count++;
                noc.async_read_barrier();
                dfb_gamma_buf.push_back(1);
            }
#endif
#ifdef FUSE_BETA
            for (uint32_t i = 0; i < dfb_leftovers; i++) {
                dfb_beta_buf.reserve_back(1);
                uint32_t l1_write_addr = dfb_beta_buf.get_write_ptr();
                async_read_row_to_tile<beta_is_row_major>(noc, addrb, beta_tile_count, l1_write_addr, beta_datum_bytes);
                beta_tile_count++;
                noc.async_read_barrier();
                dfb_beta_buf.push_back(1);
            }
#endif
#endif
        }
    }  // ncht loop
}
