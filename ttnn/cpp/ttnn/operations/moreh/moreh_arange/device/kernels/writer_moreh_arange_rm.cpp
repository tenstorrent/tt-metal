// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#define TILE_WIDTH 32

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t tile_offset = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t start = get_arg_val<uint32_t>(3);
    uint32_t step = get_arg_val<uint32_t>(4);
    uint32_t element_size = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    uint32_t num_bytes_per_tile = TILE_WIDTH * element_size;

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(dst_args, dst_addr);

    union value {
        float f;
        uint32_t u;
    };
    value start_u, step_u;

    start_u.u = start;
    step_u.u = step;

    Noc noc;
    CircularBuffer cb_out_obj(cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_out_obj.reserve_back(1);

        uint32_t tile_idx = tile_offset + t;

        uint32_t w_addr = cb_out_obj.get_write_ptr();

#ifdef OUTPUT_DTYPE_BFLOAT16
        CoreLocalMem<uint16_t> ptr(w_addr);

        for (uint32_t w = 0; w < TILE_WIDTH; w++) {
            int32_t idx = w + tile_idx * TILE_WIDTH;
            value val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w] = uint16_t(val.u >> 16);
        }
#endif
#ifdef OUTPUT_DTYPE_INT32
        CoreLocalMem<uint32_t> ptr(w_addr);

        for (uint32_t w = 0; w < TILE_WIDTH; w++) {
            int32_t idx = w + tile_idx * TILE_WIDTH;
            int32_t val;
            val = start_u.f + step_u.f * idx;
            ptr[w] = val;
        }
#endif
#ifdef OUTPUT_DTYPE_FLOAT32
        CoreLocalMem<uint32_t> ptr(w_addr);

        for (uint32_t w = 0; w < TILE_WIDTH; w++) {
            int32_t idx = w + tile_idx * TILE_WIDTH;
            value val;
            val.f = start_u.f + step_u.f * idx;
            ptr[w] = val.u;
        }
#endif

        uint32_t noc_offfset = tile_idx * TILE_WIDTH * element_size;
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out_obj),
            s0,
            num_bytes_per_tile,
            {.offset_bytes = 0},
            {.page_id = 0, .offset_bytes = noc_offfset});
        noc.async_write_barrier();
    }
}
