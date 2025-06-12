// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "../../../../../kernel_helper_functions/pad_tile.hpp"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t blk = get_arg_val<uint32_t>(3);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_datum_padded = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_11;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    constexpr uint32_t cb_id_mask = tt::CBIndex::c_5;
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(4);
    // const uint32_t num_datum_padded = get_arg_val<uint32_t>(5);

    // Adds -inf padding. Note: the value is the uint16 representation of bfloat16's -inf
    constexpr uint16_t mask_val = 0xFF80;
    constexpr uint32_t mask_val_32 = ((uint32_t)mask_val << 16) + mask_val;
    if (mask_padded_data) {
        // generate_bcast_row_mask(cb_id_mask, num_datum_padded, mask_val);
        uint32_t ptr = (get_write_ptr(cb_id_mask));
        // same pointer, but for zeroing out the tile
        volatile tt_l1_ptr uint16_t* zero_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id_mask));
        for (uint32_t i = 0; i < TILE_WIDTH * TILE_HEIGHT; i++) {
            zero_ptr[i] = 0.0f;
        }
        constexpr uint32_t num_datum_unpadded = 32 - num_datum_padded;
        fill_pad_tile<uint16_t, num_datum_unpadded, 32>(ptr, mask_val);
        cb_push_back(cb_id_mask, 1);
    }

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        cb_wait_front(cb_id_out0, blk);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j < blk; j++) {
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, blk);
    }
}
