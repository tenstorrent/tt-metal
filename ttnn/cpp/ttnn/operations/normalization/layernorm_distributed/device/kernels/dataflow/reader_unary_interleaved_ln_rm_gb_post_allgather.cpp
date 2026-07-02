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
    const uint32_t src_addr = get_arg_val<uint32_t>(0);     // Source address in dram
    const uint32_t NCHt = get_arg_val<uint32_t>(1);         // Number of NCH tiles
    const uint32_t tile_offset = get_arg_val<uint32_t>(3);  // Tile offset for this core
    const uint32_t stats_tile_offset =
        get_arg_val<uint32_t>(4);  // Tile offset for stats input; status input is two tiles wide and contains E(x) and
                                   // E(x^2) in the left most columns per tile.

    const uint32_t gamma_addr = get_arg_val<uint32_t>(6);
    const uint32_t beta_addr = get_arg_val<uint32_t>(7);
    const uint32_t stats_addr = get_arg_val<uint32_t>(8);
    const uint32_t y_offset = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_stats = tt::CBIndex::c_1;
    constexpr uint32_t cb_gamma = tt::CBIndex::c_2;
    constexpr uint32_t cb_beta = tt::CBIndex::c_3;
    constexpr uint32_t cb_eps = tt::CBIndex::c_4;
    constexpr uint32_t cb_reduce = tt::CBIndex::c_5;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_inp);
    const uint32_t stats_tile_bytes = get_tile_size(cb_stats);
    // datum size (bytes) of gamma/beta, derived from their tile size (TILE_HW = 32*32 = 1024 datums/tile).
    // Used to scale the row/face byte offsets when packing a stick into tile layout (bf16=2B, fp32=4B).
    const uint32_t gamma_datum_bytes = get_tile_size(cb_gamma) / tt::constants::TILE_HW;
    const uint32_t beta_datum_bytes = get_tile_size(cb_beta) / tt::constants::TILE_HW;

    constexpr uint32_t blk = get_compile_time_arg_val(0);
    constexpr uint32_t stats_tiles_cols = get_compile_time_arg_val(1);
    constexpr uint32_t gamma_stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t beta_stick_size = get_compile_time_arg_val(3);
    constexpr uint32_t gamma_is_row_major = get_compile_time_arg_val(4);
    constexpr uint32_t beta_is_row_major = get_compile_time_arg_val(5);
    constexpr uint32_t cb_length = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);  // Width in tiles
    constexpr uint32_t reduce_factor = get_compile_time_arg_val(8);
    constexpr auto src_args = TensorAccessorArgs<9>();
    constexpr auto stats_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<stats_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto src_a = TensorAccessor(src_args, src_addr);
    const auto src_stats = TensorAccessor(stats_args, stats_addr);

#ifdef FUSE_GAMMA
    const auto addrg = TensorAccessor(gamma_args, gamma_addr, gamma_stick_size);
#endif
#ifdef FUSE_BETA
    const auto addrb = TensorAccessor(beta_args, beta_addr, beta_stick_size);
#endif

    // Generate constant tiles for layernorm compute
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_reduce,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        reduce_factor>();
    const uint32_t eps = get_arg_val<uint32_t>(5);
    generate_bcast_col_scalar(CircularBuffer(cb_eps), eps);

    Noc noc;
    DataflowBuffer cb_inp_buf(cb_inp);
    DataflowBuffer cb_stats_buf(cb_stats);
#ifdef FUSE_GAMMA
    DataflowBuffer cb_gamma_buf(cb_gamma);
#endif
#ifdef FUSE_BETA
    DataflowBuffer cb_beta_buf(cb_beta);
#endif

    uint32_t inp_tile_idx = tile_offset;
    uint32_t stats_tile_idx = stats_tile_offset;

    constexpr uint32_t cb_iterations = Wt / cb_length;
    constexpr uint32_t cb_leftovers = Wt % cb_length;
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        // Read stats tiles
        cb_stats_buf.reserve_back(stats_tiles_cols);
        uint32_t stats_write_offset = 0;
        for (uint32_t st = 0; st < stats_tiles_cols; ++st) {
            noc.async_read(
                src_stats,
                cb_stats_buf,
                stats_tile_bytes,
                {.page_id = stats_tile_idx},
                {.offset_bytes = stats_write_offset});
            stats_write_offset += stats_tile_bytes;
            stats_tile_idx++;
        }
        noc.async_read_barrier();
        cb_stats_buf.push_back(stats_tiles_cols);
        // In the 2D-core-grid path each core handles only a horizontal slice [y_offset, y_offset + Wt)
        // of the gamma/beta tensors. The 1D path passes y_offset == 0 so this is a no-op there.
        uint32_t gamma_tile_count = y_offset;
        uint32_t beta_tile_count = y_offset;
        for (uint32_t i = 0; i < cb_iterations; i++) {
            for (uint32_t j = 0; j < cb_length; j++) {
                cb_inp_buf.reserve_back(1);
                noc.async_read(src_a, cb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
                inp_tile_idx++;
                noc.async_read_barrier();
                cb_inp_buf.push_back(1);
            }
            if (ncht == 0 or cb_iterations != 1) {
#if defined FUSE_GAMMA || defined FUSE_BETA
#ifdef FUSE_GAMMA
                for (uint32_t j = 0; j < cb_length; j++) {
                    cb_gamma_buf.reserve_back(1);
                    uint32_t l1_write_addr = cb_gamma_buf.get_write_ptr();
                    async_read_row_to_tile<gamma_is_row_major>(
                        noc, addrg, gamma_tile_count, l1_write_addr, gamma_datum_bytes);
                    gamma_tile_count++;
                    noc.async_read_barrier();
                    cb_gamma_buf.push_back(1);
                }
#endif
#ifdef FUSE_BETA
                for (uint32_t j = 0; j < cb_length; j++) {
                    cb_beta_buf.reserve_back(1);
                    uint32_t l1_write_addr = cb_beta_buf.get_write_ptr();
                    async_read_row_to_tile<beta_is_row_major>(
                        noc, addrb, beta_tile_count, l1_write_addr, beta_datum_bytes);
                    beta_tile_count++;
                    noc.async_read_barrier();
                    cb_beta_buf.push_back(1);
                }
#endif
#endif
            }
        }
        for (uint32_t i = 0; i < cb_leftovers; i++) {
            cb_inp_buf.reserve_back(1);
            noc.async_read(src_a, cb_inp_buf, src0_tile_bytes, {.page_id = inp_tile_idx}, {.offset_bytes = 0});
            inp_tile_idx++;
            noc.async_read_barrier();
            cb_inp_buf.push_back(1);
        }
        if (ncht == 0 or cb_iterations != 1) {
#if defined FUSE_GAMMA || defined FUSE_BETA
#ifdef FUSE_GAMMA
            for (uint32_t i = 0; i < cb_leftovers; i++) {
                cb_gamma_buf.reserve_back(1);
                uint32_t l1_write_addr = cb_gamma_buf.get_write_ptr();
                async_read_row_to_tile<gamma_is_row_major>(
                    noc, addrg, gamma_tile_count, l1_write_addr, gamma_datum_bytes);
                gamma_tile_count++;
                noc.async_read_barrier();
                cb_gamma_buf.push_back(1);
            }
#endif
#ifdef FUSE_BETA
            for (uint32_t i = 0; i < cb_leftovers; i++) {
                cb_beta_buf.reserve_back(1);
                uint32_t l1_write_addr = cb_beta_buf.get_write_ptr();
                async_read_row_to_tile<beta_is_row_major>(noc, addrb, beta_tile_count, l1_write_addr, beta_datum_bytes);
                beta_tile_count++;
                noc.async_read_barrier();
                cb_beta_buf.push_back(1);
            }
#endif
#endif
        }
    }  // ncht loop
}
