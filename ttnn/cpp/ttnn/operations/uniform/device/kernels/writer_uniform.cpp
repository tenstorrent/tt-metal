// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/constants.hpp"
#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t intermed_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_cb_id = get_compile_time_arg_val(1);
    constexpr bool output_is_dram = get_compile_time_arg_val(2) == 1;

    union {
        float f;
        uint32_t u;
    } f2u_from, f2u_to;
    f2u_from.u = get_arg_val<uint32_t>(1);
    f2u_to.u = get_arg_val<uint32_t>(2);
    uint32_t start_id = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t end_id = start_id + num_tiles;

    const InterleavedAddrGenFast<output_is_dram> s = {
        .bank_base_address = get_arg_val<uint32_t>(0),
        .page_size = get_tile_size(dst_cb_id),
        .data_format = get_dataformat(dst_cb_id),
    };

    uint32_t max_uint = 4294967295;
    float random_range = f2u_to.f - f2u_from.f;

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(intermed_cb_id, 1);
        uint32_t *intermed_cb_addr = reinterpret_cast<uint32_t *>(get_read_ptr(intermed_cb_id));
        uint8_t *dst_cb_addr = reinterpret_cast<uint8_t *>(get_write_ptr(dst_cb_id));
        for (uint32_t k = 0; k < tt::constants::TILE_WIDTH; k++) {
            for (uint32_t j = 0; j < tt::constants::TILE_HEIGHT; j++) {
                uint32_t rand_uint32 = *intermed_cb_addr;
                float rand_float = (float)rand_uint32 / max_uint;
                // The hardware PRNG is not uniformly distributed. Random values in the range [0, 0.5] are generated
                // with a higher frequency than those in (0.5, 1). To correct this bias and make the distribution more
                // uniform, we scale values less than 0.5 by 2.
                if (rand_float < 0.5f)
                    rand_float *= 2;
                rand_float = rand_float * random_range + f2u_from.f;

#ifdef OUTPUT_DTYPE_FLOAT32
                *(float *)dst_cb_addr = rand_float;
                dst_cb_addr += 4;
#endif
#ifdef OUTPUT_DTYPE_BFLOAT16
                uint16_t *uint16_ptr = reinterpret_cast<uint16_t *>(&rand_float) + 1;
                *(uint16_t *)dst_cb_addr = *uint16_ptr;
                dst_cb_addr += 2;
#endif
                intermed_cb_addr += 1;
            }
        }
        cb_pop_front(intermed_cb_id, 1);

        noc_async_write_tile(i, s, get_read_ptr(dst_cb_id));
        noc_async_write_barrier();
    }
}
