// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint_pages.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#define u8_l1_ptr volatile tt_l1_ptr uint8_t*
#define u8_vol_ptr volatile uint8_t*
#define u8_ptr uint8_t*

void kernel_main() {
    constexpr auto unpadded_stick_bytes = get_arg(args::unpadded_stick_bytes);
    constexpr auto unpadded_shard_height = get_arg(args::unpadded_shard_height);
    constexpr auto W_front_pad_bytes = get_arg(args::W_front_pad_bytes);

    constexpr auto unpadded_stick_step = get_arg(args::unpadded_stick_step);
    constexpr auto padded_stick_step = get_arg(args::padded_stick_step);

    DataflowBuffer dfb_input_shard(dfb::in_shard);
    DataflowBuffer dfb_output_shard(dfb::out_shard);

    uint32_t input_shard_base_addr = dfb_input_shard.get_write_ptr();
    uint32_t output_shard_base_addr = dfb_output_shard.get_write_ptr();

    auto input_stick_ptr = reinterpret_cast<u8_l1_ptr>(input_shard_base_addr);
    auto output_stick_ptr = reinterpret_cast<u8_l1_ptr>(output_shard_base_addr);

    // fill the sticks that aren't entirely padding with data from the input tensor
    for (uint32_t h = 0; h < unpadded_shard_height; h++) {
        dfb_output_shard.wait_front(1);  // wait for writer to fill this stick with padding

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

        dfb_output_shard.pop_front(1);

        input_stick_ptr += unpadded_stick_step;
        output_stick_ptr += padded_stick_step;
    }
}
