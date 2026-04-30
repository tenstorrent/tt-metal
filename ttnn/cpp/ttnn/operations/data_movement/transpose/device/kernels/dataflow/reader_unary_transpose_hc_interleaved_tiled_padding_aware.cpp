// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t num_writes = get_named_compile_time_arg_val("num_writes");
    constexpr uint32_t padding_val_packed = get_named_compile_time_arg_val("padding_val_packed");
    constexpr uint32_t needs_padding = get_named_compile_time_arg_val("needs_padding") == 1;
    constexpr uint32_t swap_hw = get_named_compile_time_arg_val("swap_hw") == 1;
    constexpr uint32_t H = get_named_compile_time_arg_val("H");
    constexpr uint32_t W = get_named_compile_time_arg_val("W");
    constexpr uint32_t accumulated_outer_dims = get_named_compile_time_arg_val("accumulated_outer_dims");
    constexpr uint32_t TILE_HEIGHT = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t TILE_WIDTH = get_named_compile_time_arg_val("tile_width");
    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t H_p = tt::data_movement::common::round_up<H, TILE_HEIGHT>();
    constexpr uint32_t W_p = tt::data_movement::common::round_up<W, TILE_WIDTH>();

    constexpr uint32_t Wt = W_p / TILE_WIDTH;
    constexpr uint32_t Ht = H_p / TILE_HEIGHT;

    constexpr uint32_t HtWt = Ht * Wt;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const auto s = TensorAccessor(src_args, src_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);
    experimental::CircularBuffer cb_padding(tt::CBIndex::c_1);

// read a ublock of tiles from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        uint32_t linear_tile_index = 0;
        if constexpr (swap_hw) {
            uint32_t rem = i;
            uint32_t ht = rem % Ht;
            rem /= Ht;
            uint32_t wt = rem % Wt;
            rem /= Wt;
            uint32_t offset = rem % accumulated_outer_dims;
            linear_tile_index = offset * HtWt + ht * Wt + wt;
        } else {
            linear_tile_index = i;
        }
        cb.reserve_back(onetile);
        noc.async_read(s, cb, tile_bytes, {.page_id = linear_tile_index}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(onetile);
    }
    if constexpr (needs_padding) {
        // Add padding
        cb_padding.reserve_back(1);
        uint32_t l1_write_addr = cb_padding.get_write_ptr();
        // Fill with padding value
        // if bfloat16 num_writes = FACE_WIDTH / (sizeof(uint32_t))/(element_size)
        tt::data_movement::common::fill_with_val(l1_write_addr, num_writes, padding_val_packed);
        cb_padding.push_back(1);
    }
}
