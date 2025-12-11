// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///

#include <stdint.h>
#include "dataflow_api.h"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 1; ++r) {
        SliceRange sr_left = SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right =
            SliceRange{.h0 = (uint8_t)r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const uint32_t page_bytes = get_arg_val<uint32_t>(3);
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    // Debug: print page information for a few iterations
    if (start_id == 0) {
        DPRINT << "Reader kernel debug:" << ENDL();
        DPRINT << "  start_id=" << start_id << " num_tiles=" << num_tiles << ENDL();
        DPRINT << "  page_bytes=" << page_bytes << ENDL();
    }

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);

        const uint64_t src_noc_addr = s.get_noc_addr(i);

        DPRINT << "s.page_size: " << (uint32_t)s.page_size << ENDL();
        noc_async_read(src_noc_addr, l1_write_addr, s.page_size);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* dst_noc2 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
        for (uint16_t value = 0; value < 32; value++) {
            DPRINT << "value at " << (uint16_t)value << " is: " << BF16((uint16_t)dst_noc2[value]) << ENDL();
        }

        cb_push_back(cb_id_in0, onetile);
    }
    DPRINT << "Reader kernel done" << ENDL();
}
