// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "dataflow_api.h"

#define u8_l1_ptr volatile tt_l1_ptr uint8_t*
#define u8_ptr uint8_t*

#define DEBUG 1

#ifdef DEBUG
#include "debug/dprint.h"
#endif

void kernel_main() {
    DPRINT << "entered reader" << ENDL();
    constexpr uint32_t unpadded_stick_bytes         = get_compile_time_arg_val(0);
    constexpr uint32_t padded_stick_bytes         = get_compile_time_arg_val(1);
    constexpr uint32_t unpadded_shard_height        = get_compile_time_arg_val(2);
    constexpr uint32_t padded_shard_height        = get_compile_time_arg_val(3);
    constexpr uint32_t W_front_pad_bytes            = get_compile_time_arg_val(4);
    constexpr uint32_t W_back_pad_bytes            = get_compile_time_arg_val(5);

    constexpr auto input_shard_cb = tt::CBIndex::c_0;
    constexpr auto output_shard_cb = tt::CBIndex::c_16;

    // cb_reserve_back(cb_output_shard, padded_shard_height); // needed? correct?
    uint32_t input_shard_base_addr = get_write_ptr(input_shard_cb);
    uint32_t output_shard_base_addr = get_write_ptr(output_shard_cb);

    // fill the sticks that aren't entirely padding with data from the input tensor
    for (uint32_t h = 0; h < unpadded_shard_height; h++) {
        DPRINT << "waiting for writer to fill stick " << h << ENDL();
        cb_wait_front(output_shard_cb, 1); // wait for writer to fill this stick with padding
        DPRINT << "writer filled stick " << h << ENDL();
        uint32_t input_stick_addr = input_shard_base_addr + h * unpadded_stick_bytes;
        uint32_t output_stick_addr = output_shard_base_addr + h * padded_stick_bytes;

        // read the input stick into the padded output stick starting after the
        // front padding

        // FIXME: this isn't aligned. we need to do a memcpy for now. we can try
        // to do a noc_async_read later on with a trick.

        // noc_async_read(output_stick_addr + W_front_pad_bytes, input_stick_addr, unpadded_stick_bytes);

        // NOTE: memcpy is safe here because the input/output tensors have disjoint buffers.
        DPRINT << "copying from " << input_stick_addr << " to " << output_stick_addr << ENDL();
        memcpy(
            reinterpret_cast<u8_ptr>(output_stick_addr + W_front_pad_bytes),
            reinterpret_cast<u8_ptr>(input_stick_addr),
            unpadded_stick_bytes);
        DPRINT << "copied" << ENDL();
    }
}
