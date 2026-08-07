// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

using namespace tt;

void kernel_main() {
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t end_id = start_id + num_tiles;

    const auto output_addrg = TensorAccessor(dst_args, dst_addr);

    const uint32_t page_bytes = get_local_cb_interface(dst_cb_id).fifo_page_size;

    Noc noc;
    CircularBuffer cb_intermed(intermed_cb_id);
    CircularBuffer cb_dst(dst_cb_id);

    cb_dst.reserve_back(1);
    uint32_t dst_cb_write_ptr = cb_dst.get_write_ptr();

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_intermed.wait_front(1);

        uint32_t intermed_cb_read_ptr = cb_intermed.get_read_ptr();
        auto intermed_cb_addr = reinterpret_cast<float*>(intermed_cb_read_ptr);

#ifdef OUTPUT_DTYPE_FLOAT32
        noc.async_write(
            CoreLocalMem<uint32_t>(intermed_cb_read_ptr),
            output_addrg,
            page_bytes,
            {},
            {.page_id = i});
        noc.async_write_barrier();
        cb_intermed.pop_front(1);
#endif

#ifdef OUTPUT_DTYPE_BFLOAT16
        auto dst_cb_addr = reinterpret_cast<uint8_t*>(dst_cb_write_ptr);
        for (uint32_t k = 0; k < constants::TILE_WIDTH; k++) {
            for (uint32_t j = 0; j < constants::TILE_HEIGHT; j++) {
                float rand_float = *intermed_cb_addr;

                uint16_t* uint16_ptr = reinterpret_cast<uint16_t*>(&rand_float) + 1;
                *(uint16_t*)dst_cb_addr = *uint16_ptr;
                dst_cb_addr += 2;
                intermed_cb_addr += 1;
            }
        }
        cb_intermed.pop_front(1);

        noc.async_write(
            CoreLocalMem<uint32_t>(dst_cb_write_ptr),
            output_addrg,
            page_bytes,
            {},
            {.page_id = i});
        noc.async_write_barrier();
#endif
    }

    // dst_cb is reserved once as a conversion-staging region (consumed only by direct NOC
    // writes, never streamed to a consumer); commit the reservation so the CB is left balanced.
    cb_dst.push_back(1);
}
