// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"

// Metal 2.0 (sharded_to_interleaved private copy, ROW_MAJOR / stick layout): writes the shard residing in
// the output DFB (dfb::out) out to the interleaved destination tensor (ta::output). Only-allowed changes
// from the descriptor era: the CB id comes from the DFB binding token (dfb::out), the destination base
// address comes from the TensorAccessor binding (ta::output) instead of positional RTA 0, and the
// remaining run-time values come from the named-arg namespace (args::). The descriptor-era positional
// arg 1 (num_units_per_row) was never read in the body, so it is dropped. The data-movement logic is
// unchanged: where the descriptor-era kernel folded input_width_offset_bytes into the accessor base
// address, the binding-token form (which resolves the base address itself) instead carries that
// within-stick byte offset on the destination page coordinate (.offset_bytes) — the resulting NOC
// address is identical.
void kernel_main() {
    const uint32_t block_height = get_arg(args::block_height);
    const uint32_t block_width_bytes = get_arg(args::block_width_bytes);
    const uint32_t padded_block_width_bytes = get_arg(args::padded_block_width_bytes);
    const uint32_t input_width_offset_bytes = get_arg(args::input_width_offset_bytes);
    const uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t cb_id_out0 = dfb::out;

    const auto s0 = TensorAccessor(ta::output);

    Noc noc;
    CircularBuffer cb_out(cb_id_out0);

    uint32_t stick_id = start_id;
    cb_out.wait_front(block_height);
    uint32_t cb_read_offset = 0;
    for (uint32_t h = 0; h < block_height; ++h) {
        noc.async_write(
            cb_out,
            s0,
            block_width_bytes,
            {.offset_bytes = cb_read_offset},
            {.page_id = stick_id, .offset_bytes = input_width_offset_bytes});
        stick_id++;
        cb_read_offset += padded_block_width_bytes;
    }
    noc.async_write_barrier();
    cb_out.pop_front(block_height);
}
