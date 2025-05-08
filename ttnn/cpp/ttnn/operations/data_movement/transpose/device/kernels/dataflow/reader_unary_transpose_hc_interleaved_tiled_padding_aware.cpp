// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_writes = get_compile_time_arg_val(1);
    constexpr uint32_t padding_val_packed = get_compile_time_arg_val(2);
    constexpr uint32_t needs_padding = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t swap_hw = get_compile_time_arg_val(4) == 1;
    constexpr uint32_t H = get_compile_time_arg_val(5);
    constexpr uint32_t W = get_compile_time_arg_val(6);
    constexpr uint32_t accumulated_outer_dims = get_compile_time_arg_val(7);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(8);
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(9);

    constexpr uint32_t H_p = tt::data_movement::common::round_up<H, TILE_HEIGHT>();
    constexpr uint32_t W_p = tt::data_movement::common::round_up<W, TILE_WIDTH>();

    constexpr uint32_t Wt = W_p / TILE_WIDTH;
    constexpr uint32_t Ht = H_p / TILE_HEIGHT;

    constexpr uint32_t HtWt = Ht * Wt;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

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
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(linear_tile_index, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
    if constexpr (needs_padding) {
        // Add padding
        cb_reserve_back(tt::CBIndex::c_1, 1);
        uint32_t l1_write_addr = get_write_ptr(tt::CBIndex::c_1);
        // Fill with padding value
        // if bfloat16 num_writes = FACE_WIDTH / (sizeof(uint32_t))/(element_size)
        tt::data_movement::common::fill_with_val(l1_write_addr, num_writes, padding_val_packed);
        cb_push_back(tt::CBIndex::c_1, 1);
    }
}
