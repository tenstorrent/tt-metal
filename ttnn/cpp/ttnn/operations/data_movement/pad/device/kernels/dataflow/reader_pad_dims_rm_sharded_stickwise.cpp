// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port (in place — used only by the PadRmShardedWidthOnly factory).
// Logic UNCHANGED; only the access mechanism moves to named bindings:
//   - input shard CB  -> dfb::in0  (borrowed input shard; read by base pointer only -> self-loop)
//   - output shard CB -> dfb::out0 (borrowed output shard; reader consumes the padded sticks the
//                                   writer produced, filling them in-place with input data)
//   - positional CTAs -> get_arg(args::...)

#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint_pages.h"
#include "api/dataflow/circular_buffer.h"
#include "experimental/kernel_args.h"
#define u8_l1_ptr volatile tt_l1_ptr uint8_t*
#define u8_vol_ptr volatile uint8_t*
#define u8_ptr uint8_t*

void kernel_main() {
    constexpr uint32_t unpadded_stick_bytes = get_arg(args::unpadded_stick_bytes);
    constexpr uint32_t padded_stick_bytes = get_arg(args::padded_stick_bytes);
    constexpr uint32_t unpadded_shard_height = get_arg(args::unpadded_shard_height);
    constexpr uint32_t padded_shard_height = get_arg(args::padded_shard_height);
    constexpr uint32_t W_front_pad_bytes = get_arg(args::W_front_pad_bytes);

    constexpr uint32_t input_shard_cb = dfb::in0;
    constexpr uint32_t output_shard_cb = dfb::out0;
    constexpr uint32_t unpadded_stick_step = get_arg(args::unpadded_stick_step);
    constexpr uint32_t padded_stick_step = get_arg(args::padded_stick_step);

    DataflowBuffer cb_input_shard(input_shard_cb);
    DataflowBuffer cb_output_shard(output_shard_cb);

    uint32_t input_shard_base_addr = cb_input_shard.get_write_ptr();
    uint32_t output_shard_base_addr = cb_output_shard.get_write_ptr();

    auto input_stick_ptr = reinterpret_cast<u8_l1_ptr>(input_shard_base_addr);
    auto output_stick_ptr = reinterpret_cast<u8_l1_ptr>(output_shard_base_addr);

    // fill the sticks that aren't entirely padding with data from the input tensor
    for (uint32_t h = 0; h < unpadded_shard_height; h++) {
        cb_output_shard.wait_front(1);  // wait for writer to fill this stick with padding

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

        cb_output_shard.pop_front(1);

        input_stick_ptr += unpadded_stick_step;
        output_stick_ptr += padded_stick_step;
    }
}
