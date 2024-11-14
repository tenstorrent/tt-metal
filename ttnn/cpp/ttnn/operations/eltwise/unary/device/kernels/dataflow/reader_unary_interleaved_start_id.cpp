// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

template <typename T>
FORCE_INLINE void fill_with_val(uint32_t begin_addr, uint32_t n, T val) {
    auto* ptr = reinterpret_cast<volatile tt_l1_ptr T*>(begin_addr);
    for (uint32_t i = 0; i < n; ++i) {
        DPRINT << "fill_with_val: " << i << " " << val << ENDL();
        ptr[i] = val;
    }
}

void kernel_main() {
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr,
        .page_size = tile_bytes,
        .data_format = data_format
    };

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    #ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; -- i) {
    #else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++ i) {
    #endif
        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }

    cb_reserve_back(tt::CB::c_in1, 1);
    uint32_t l1_write_addr = get_write_ptr(tt::CB::c_in1);
    fill_with_val<uint32_t>(l1_write_addr, 8, 123123);
    cb_push_back(tt::CB::c_in1, 1);

}
