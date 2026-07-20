// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 port of the width-only (stickwise) sharded pad reader (private to
// PadRmShardedWidthOnlyProgramFactory). Device-side logic is unchanged; resource access moves to the
// Metal 2.0 named handles (dfb::/args::):
//   - c_0 input shard  -> dfb::cb_input_shard  (borrowed-from-input fake CB: reads the resident input
//                         shard by base pointer; bound as a self-loop).
//   - c_16 output shard -> dfb::cb_output_shard (borrowed-from-output; writer PRODUCER, reader CONSUMER:
//                          the reader patches the real data into each padded stick the writer produced).
#include <stdint.h>
#include <cstring>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#define u8_l1_ptr volatile tt_l1_ptr uint8_t*

void kernel_main() {
    constexpr uint32_t unpadded_stick_bytes = get_arg(args::unpadded_stick_bytes);
    constexpr uint32_t unpadded_shard_height = get_arg(args::unpadded_shard_height);
    constexpr uint32_t W_front_pad_bytes = get_arg(args::W_front_pad_bytes);
    constexpr uint32_t unpadded_stick_step = get_arg(args::unpadded_stick_step);
    constexpr uint32_t padded_stick_step = get_arg(args::padded_stick_step);

    DataflowBuffer cb_input_shard(dfb::cb_input_shard);
    DataflowBuffer cb_output_shard(dfb::cb_output_shard);

    uint32_t input_shard_base_addr = cb_input_shard.get_write_ptr();
    uint32_t output_shard_base_addr = cb_output_shard.get_write_ptr();

    auto input_stick_ptr = reinterpret_cast<u8_l1_ptr>(input_shard_base_addr);
    auto output_stick_ptr = reinterpret_cast<u8_l1_ptr>(output_shard_base_addr);

    // fill the sticks that aren't entirely padding with data from the input tensor
    for (uint32_t h = 0; h < unpadded_shard_height; h++) {
        cb_output_shard.wait_front(1);  // wait for the writer to fill this stick with padding

        // read the input stick into the padded output stick starting after the front padding
        for (uint32_t i = 0; i < unpadded_stick_bytes; i++) {
            output_stick_ptr[W_front_pad_bytes + i] = input_stick_ptr[i];
        }

        cb_output_shard.pop_front(1);

        input_stick_ptr += unpadded_stick_step;
        output_stick_ptr += padded_stick_step;
    }
}
