// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>

#include "dataflow_api.h"

#define TILE_WIDTH 32

void kernel_main() {
    // Kernel args
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t tile_offset = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start = get_arg_val<uint32_t>(3);
    uint32_t step = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = tt::CB::c_out0;

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    uint32_t num_bytes_per_tile = get_tile_size(cb_out);

    const InterleavedAddrGen<dst_is_dram> s0 = {.bank_base_address = dst_addr, .page_size = num_bytes_per_tile};

    typedef union {
        float f;
        uint32_t u;
    } u;
    u start_u, step_u;

    start_u.u = start;
    step_u.u = step;

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_reserve_back(cb_out, 1);

        uint32_t tile_idx = tile_offset + t;

        uint32_t w_addr = get_write_ptr(cb_out);

        #ifdef OUTPUT_DTYPE_BFLOAT16
        auto ptr = reinterpret_cast<uint16_t *>(w_addr);
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = w + tile_idx * TILE_WIDTH;
            u val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w] = uint16_t(val.u >> 16);
        }
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = (w + 16) + tile_idx * TILE_WIDTH;
            u val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w + 256] = uint16_t(val.u >> 16);
        }
        #endif
        #ifdef OUTPUT_DTYPE_INT32
        auto ptr = reinterpret_cast<uint32_t *>(w_addr);
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = w + tile_idx * TILE_WIDTH;
            int32_t val;
            val = start_u.f + step_u.f * idx;
            ptr[w] = val;
        }
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = (w + 16) + tile_idx * TILE_WIDTH;
            int32_t val;
            val = start_u.f + step_u.f * idx;
            ptr[w + 256] = val;
        }
        #endif
        #ifdef OUTPUT_DTYPE_FLOAT32
        auto ptr = reinterpret_cast<uint32_t *>(w_addr);
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = w + tile_idx * TILE_WIDTH;
            u val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w] = val.u;
        }
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = (w + 16) + tile_idx * TILE_WIDTH;
            u val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w + 256] = val.u;
        }
        #endif

        uint64_t dst_noc_addr = get_noc_addr(tile_idx, s0);
        noc_async_write(w_addr, dst_noc_addr, num_bytes_per_tile);
        noc_async_write_barrier();
    }
}
