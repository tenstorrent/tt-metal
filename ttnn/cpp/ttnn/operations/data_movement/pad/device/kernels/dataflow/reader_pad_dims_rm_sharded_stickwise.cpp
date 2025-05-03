// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "dataflow_api.h"
#include "tt_metal/hw/inc/debug/dprint_pages.h"

#define u8_l1_ptr volatile tt_l1_ptr uint8_t*
#define u8_vol_ptr volatile uint8_t*
#define u8_ptr uint8_t*

void kernel_main() {
    constexpr uint32_t unpadded_stick_bytes         = get_compile_time_arg_val(0);
    constexpr uint32_t padded_stick_bytes         = get_compile_time_arg_val(1);
    constexpr uint32_t unpadded_shard_height        = get_compile_time_arg_val(2);
    constexpr uint32_t padded_shard_height        = get_compile_time_arg_val(3);
    constexpr uint32_t W_front_pad_bytes = get_compile_time_arg_val(4);

    constexpr uint32_t input_shard_cb = get_compile_time_arg_val(5);
    constexpr uint32_t output_shard_cb = get_compile_time_arg_val(6);
    constexpr uint32_t unpadded_stick_step = get_compile_time_arg_val(7);
    constexpr uint32_t padded_stick_step = get_compile_time_arg_val(8);

    uint32_t input_shard_base_addr = get_write_ptr(input_shard_cb);
    uint32_t output_shard_base_addr = get_write_ptr(output_shard_cb);

    auto input_stick_ptr = reinterpret_cast<u8_l1_ptr>(input_shard_base_addr);
    auto output_stick_ptr = reinterpret_cast<u8_l1_ptr>(output_shard_base_addr);

    // fill the sticks that aren't entirely padding with data from the input tensor
    for (uint32_t h = 0; h < unpadded_shard_height; h++) {
        cb_wait_front(output_shard_cb, 1);  // wait for writer to fill this stick with padding

        // FIXME: this isn't aligned. we need to do a memcpy for now. we can try
        // to do a noc_async_read later on with a trick.
        //
        // currently small noc transfers are slow, but once runtime drops an
        // optimization (upcoming as of 12/12/2024) this might be worth
        // investigating.

        // paulk says that an optimized loop will still be faster.
        // TODO(jkruer): get paul's help optimizing this.

        // read the input stick into the padded output stick starting after the
        // front padding
        for (uint32_t i = 0; i < unpadded_stick_bytes; i++) {
            output_stick_ptr[W_front_pad_bytes + i] = input_stick_ptr[i];
        }

        cb_pop_front(output_shard_cb, 1);

        input_stick_ptr += unpadded_stick_step;
        output_stick_ptr += padded_stick_step;
    }
}
