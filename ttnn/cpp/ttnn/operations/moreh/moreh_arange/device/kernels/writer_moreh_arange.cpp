// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/core_local_mem.h"
#include "experimental/tensor.h"

#define TILE_WIDTH 32

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t tile_offset = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start = get_arg_val<uint32_t>(3);
    uint32_t step = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    uint32_t num_bytes_per_tile = get_tile_size(cb_out);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(dst_args, dst_addr);

    union value {
        float f;
        uint32_t u;
    };
    value start_u, step_u;

    start_u.u = start;
    step_u.u = step;

    experimental::Noc noc;
    experimental::CircularBuffer cb_out_obj(cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_out_obj.reserve_back(1);

        uint32_t tile_idx = tile_offset + t;

        uint32_t w_addr = cb_out_obj.get_write_ptr();

#ifdef OUTPUT_DTYPE_BFLOAT16
        experimental::CoreLocalMem<uint16_t> ptr(w_addr);
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = w + tile_idx * TILE_WIDTH;
            value val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w] = uint16_t(val.u >> 16);
        }
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = (w + 16) + tile_idx * TILE_WIDTH;
            value val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w + 256] = uint16_t(val.u >> 16);
        }
#endif
#ifdef OUTPUT_DTYPE_INT32
        experimental::CoreLocalMem<uint32_t> ptr(w_addr);
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
        experimental::CoreLocalMem<uint32_t> ptr(w_addr);
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = w + tile_idx * TILE_WIDTH;
            value val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w] = val.u;
        }
        for (uint32_t w = 0; w < 16; w++) {
            int32_t idx = (w + 16) + tile_idx * TILE_WIDTH;
            value val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w + 256] = val.u;
        }
#endif

        noc.async_write(
            experimental::use<experimental::CircularBuffer::AddrSelector::WRITE_PTR>(cb_out_obj),
            s0,
            num_bytes_per_tile,
            {.offset_bytes = 0},
            {.page_id = tile_idx});
        noc.async_write_barrier();
    }
}
