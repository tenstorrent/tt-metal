// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

// H-bcast mask
FORCE_INLINE void generate_bcast_row_mask(
    const uint32_t cb_id, const uint32_t num_datum_padded, const uint16_t mask_val) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr uint16_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_id));

    if (num_datum_padded > 16) {
        uint32_t num_datum_unpadded_f1 = 32 - num_datum_padded;
        uint32_t idx = 0;
        for (uint32_t j = 0; j < num_datum_unpadded_f1; ++j) {  // first face
            ptr[idx + j] = 0;
        }
        for (uint32_t j = num_datum_unpadded_f1; j < 16; ++j) {  // first face
            ptr[idx + j] = mask_val;
        }

        idx = 1 << 8;
        for (uint32_t j = 0; j < 16; ++j) {  // second face
            ptr[idx + j] = mask_val;
        }
    } else {
        uint32_t num_datum_unpadded_f2 = 16 - num_datum_padded;
        uint32_t idx = 0;
        for (uint32_t j = 0; j < 16; ++j) {  // first face
            ptr[idx + j] = 0;
        }

        idx = 1 << 8;
        for (uint32_t j = 0; j < num_datum_unpadded_f2; ++j) {  // second face
            ptr[idx + j] = 0;
        }
        for (uint32_t j = num_datum_unpadded_f2; j < 16; ++j) {  // second face
            ptr[idx + j] = mask_val;
        }
    }

    cb_push_back(cb_id, 1);
}
inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t blk = get_arg_val<uint32_t>(3);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_out0 = tt::CBIndex::c_11;
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    const DataFormat data_format = get_dataformat(cb_id_out0);

    constexpr uint32_t cb_id_mask = tt::CBIndex::c_5;
    const uint32_t mask_padded_data = get_arg_val<uint32_t>(4);
    const uint32_t num_datum_padded = get_arg_val<uint32_t>(5);

    // Adds -inf padding. Note: the value is the uint16 representation of bfloat16's -inf
    constexpr uint16_t mask_val = 0xFF80;
    if (mask_padded_data) {
        generate_bcast_row_mask(cb_id_mask, num_datum_padded, mask_val);
    }

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < num_tiles; i += blk) {
        cb_wait_front(cb_id_out0, blk);

        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        for (uint32_t j = 0; j < blk; j++) {
            print_full_tile(cb_id_out0, j, true);
            noc_async_write_tile(tile_id, s, l1_read_addr);
            tile_id++;
            l1_read_addr += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_id_out0, blk);
    }
}
