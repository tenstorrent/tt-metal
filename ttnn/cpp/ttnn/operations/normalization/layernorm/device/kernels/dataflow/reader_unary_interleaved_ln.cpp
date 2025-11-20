// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <utility>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
#include "ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include <tt-metalium/constants.hpp>

template <typename T>
void read_row_to_cb(
    const uint32_t cb_id, const T& addr, const uint32_t tile_bytes, const uint32_t offset, const uint32_t blk) {
    cb_reserve_back(cb_id, blk);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    for (uint32_t r = 0; r < blk; r++) {
        noc_async_read_tile(offset + r, addr, l1_write_addr);
        l1_write_addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, blk);
}

namespace detail {
template <uint32_t W, uint32_t Wt, uint32_t block_width_tiles, uint32_t tile_in_block>
void pad_tile_width(uint32_t l1_write_ptr, uint32_t tile_size_bytes) {
    // If W evenly divides into block_width_tiles, then there is no padding to do
    if constexpr (Wt % block_width_tiles == 0 && W / tt::constants::TILE_WIDTH == Wt) {
        return;
    }

    // The last block (the one that needs to be padded)
    constexpr uint32_t last_block = Wt / block_width_tiles;
    constexpr uint32_t last_block_offset_tiles = last_block * block_width_tiles;

    // One tile per block will need to be partially padded
    constexpr uint32_t tile_offset_tiles = last_block_offset_tiles + tile_in_block;

    constexpr uint32_t width_at_tile_start = tile_offset_tiles * tt::constants::TILE_WIDTH;
    constexpr int width_diff = W - width_at_tile_start;
    constexpr auto unpadded_width_fn = []() {
        if constexpr (width_diff >= static_cast<int>(tt::constants::TILE_WIDTH)) {
            // Full tile, no padding needed (unpadded width is 32)
            return static_cast<uint32_t>(tt::constants::TILE_WIDTH);
        } else if constexpr (width_diff >= 0 && width_diff < static_cast<int>(tt::constants::TILE_WIDTH)) {
            // Pad partial tile with unpadded width width_diff
            return static_cast<uint32_t>(width_diff);
        } else {
            // Pad the full tile (unpadded width is 0)
            return static_cast<uint32_t>(0);
        }
    };
    fill_pad_tile<uint32_t, unpadded_width_fn(), tt::constants::TILE_HEIGHT>(l1_write_ptr, 0);
}

template <uint32_t W, uint32_t Wt, uint32_t block_width_tiles, uint32_t... tiles_in_block>
void pad_block_impl(
    uint32_t l1_write_ptr, uint32_t tile_size_bytes, std::integer_sequence<uint32_t, tiles_in_block...>) {
    (pad_tile_width<W, Wt, block_width_tiles, tiles_in_block>(
         l1_write_ptr + tiles_in_block * tile_size_bytes, tile_size_bytes),
     ...);
}

template <uint32_t W, uint32_t Wt, uint32_t block_width_tiles>
void pad_block_width(uint32_t cb_id) {
    auto l1_write_ptr = get_write_ptr(cb_id);
    auto tile_size_bytes = get_tile_size(cb_id);
    constexpr uint32_t first_tile_in_last_block = (Wt / block_width_tiles) * block_width_tiles;
    l1_write_ptr += first_tile_in_last_block * tile_size_bytes;
    pad_block_impl<W, Wt, block_width_tiles>(
        l1_write_ptr, tile_size_bytes, std::make_integer_sequence<uint32_t, block_width_tiles>{});
}
}  // namespace detail
void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t NCHt = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t gamma_addr = get_arg_val<uint32_t>(5);
    uint32_t beta_addr = get_arg_val<uint32_t>(6);
    uint32_t b_addr = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0, cb_id_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_gamma = tt::CBIndex::c_5;
    constexpr uint32_t cb_id_beta = tt::CBIndex::c_6;

    // ublocks size defined in tiles
    const uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);

    constexpr uint32_t blk = get_compile_time_arg_val(0);  // needed for correctness of softmax/LN kernels
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t W = get_compile_time_arg_val(2);
    constexpr bool use_welford = get_compile_time_arg_val(3) == 1;
    constexpr auto src0_args = TensorAccessorArgs<4>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto gamma_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();
    constexpr auto beta_args = TensorAccessorArgs<gamma_args.next_compile_time_args_offset()>();

    const auto src_a = TensorAccessor(src0_args, src_addr, src0_tile_bytes);
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
    if constexpr (!use_welford) {
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        uint32_t scaler = get_arg_val<uint32_t>(3);
        generate_reduce_scaler(cb_in_2, scaler);
    }

    constexpr uint32_t eps_cb_id = 3;
    const uint32_t eps = get_arg_val<uint32_t>(4);
    generate_bcast_col_scalar(eps_cb_id, eps);

    // Pad the last block of input with 0's
    // detail::pad_block_width<W, Wt, blk>(cb_id_in0);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t offs = 0;
    auto read_in0_and_in1 = [&]() {
        for (uint32_t wt = 0; wt < Wt; wt += blk) {
            read_row_to_cb(cb_id_in0, src_a, src0_tile_bytes, offs + wt + tile_offset, blk);
#ifdef FUSE_PRE_ADD
            // TODO(AP): refactor the ifdefs
            read_row_to_cb(cb_id_in1, src_b, src1_tile_bytes, offs + wt + tile_offset, blk);
#endif
        }  // wt loop
    };
    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        read_in0_and_in1();
#if defined FUSE_GAMMA || defined FUSE_BETA
        if (ncht == 0) {
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
#ifdef FUSE_GAMMA
                {
                    read_row_to_cb(cb_id_gamma, addrg, gamma_tile_bytes, wt, blk);
                }
#endif

#ifdef FUSE_BETA
                {
                    read_row_to_cb(cb_id_beta, addrb, beta_tile_bytes, wt, blk);
                }
#endif
            }  // wt loop
        }
#endif
        offs += Wt;
    }  // ncht loop
}
