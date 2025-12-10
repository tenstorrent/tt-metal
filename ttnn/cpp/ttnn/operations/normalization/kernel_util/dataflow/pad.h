// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file pad.h
 * @brief Padding utilities for use in reader/writer kernels.
 */

#pragma once

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/kernel_helper_functions/pad_tile.hpp"
#include <tt-metalium/constants.hpp>

namespace norm::kernel_util::dataflow {
namespace detail {
/**
 * @brief Pad a single row of tiles from width `W` to width `W_pad`
 *
 * @param cb_in Input CB to accumulate
 * @param cb_scalar CB containing the scalar tile to use in reduce
 * @param cb_out Output CB to store the accumulated value
 * @param num_tiles Number of tiles containing the data
 * @param block_size Number of tiles to process at a time
 * @param epilogue Optional functor to call after the accumulation before tile registers
 * are committed and packed
 * @param additional_cbs Optional additional input CBs to accumulate
 * @tparam FLOAT32_REDUCTION Whether to reduce the sum in FP32 precision
 * @tparam pop_input_policy The policy for whether to pop the input CB after processing
 * @tparam wait_at_end_policy The policy for whether to wait at the end of the function
 * @tparam Epilogue The type of the epilogue functor
 * @tparam AdditionalCBs The types of the additional input CBs (must be uint32_t)
 *
 */
template <uint32_t W, uint32_t Wt, uint32_t block_width_tiles, uint32_t tile_in_block>
void pad_tile_width(uint32_t l1_write_ptr, uint32_t tile_size_bytes) {
    // If W evenly divides into block_width_tiles, then there is no padding to do
    if constexpr (Wt % block_width_tiles == 0 && W / tt::constants::TILE_WIDTH == Wt) {
        return;
    }

    // The last block (the one that needs to be padded)
    constexpr uint32_t last_block = Wt / block_width_tiles;
    constexpr uint32_t block_offset_tiles = last_block * block_width_tiles;

    // One tile per block will need to be partially padded
    constexpr uint32_t tile_offset_tiles = block_offset_tiles + tile_in_block;

    constexpr uint32_t width_at_tile_start = tile_offset_tiles * tt::constants::TILE_WIDTH;
    constexpr int width_diff = W - width_at_tile_start;
    constexpr auto unpadded_width_fn = []() {
        if constexpr (width_diff >= static_cast<int>(tt::constants::TILE_WIDTH)) {
            // Full tile, no padding needed (unpadded width is 32)
            return 32;
        } else if constexpr (width_diff >= 0 && width_diff < static_cast<int>(tt::constants::TILE_WIDTH)) {
            // Pad partial tile with unpadded width width_diff
            return width_diff;
        } else {
            // Pad the full tile (unpadded width is 0)
            return 0;
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
}  // namespace detail

template <uint32_t W, uint32_t Wt, uint32_t block_width_tiles>
void pad_block_width(uint32_t cb_id) {
    auto l1_write_ptr = get_write_ptr(cb_id);
    auto tile_size_bytes = get_tile_size(cb_id);
    const uint32_t last_block_start_tiles = (Wt / block_width_tiles) * block_width_tiles;
    l1_write_ptr += last_block_start_tiles * tile_size_bytes;
    pad_block_impl<W, Wt, block_width_tiles>(
        l1_write_ptr, tile_size_bytes, std::make_integer_sequence<uint32_t, block_width_tiles>{});
}
}  // namespace norm::kernel_util::dataflow
