// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader kernel: fills input CB with constant-value tiles (bfloat16 1.0).
// No DRAM reads needed -- generates tiles directly in L1.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t in_cb = tt::CBIndex::c_0;

    uint32_t num_total_tiles = get_arg_val<uint32_t>(0);

    // bfloat16(1.0) = 0x3F80, packed two per uint32 = 0x3F803F80
    constexpr uint32_t packed_one = 0x3F803F80;
    // Tile is 32x32 = 1024 bfloat16 elements = 512 uint32 words = 2048 bytes
    constexpr uint32_t tile_words = 512;

    for (uint32_t t = 0; t < num_total_tiles; t++) {
        cb_reserve_back(in_cb, 1);
        uint32_t write_addr = get_write_ptr(in_cb);
        volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(write_addr);

        for (uint32_t i = 0; i < tile_words; i++) {
            ptr[i] = packed_one;
        }

        cb_push_back(in_cb, 1);
    }
}
