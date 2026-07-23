// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SCAFFOLDING writer for the gated-delta prefill-then-query op.
// Drains cb_state_out -> state' output (fp32), then cb_o_out -> O output (bf16).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t state_out_addr = get_arg_val<uint32_t>(0);
    const uint32_t o_out_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_state_tiles = get_arg_val<uint32_t>(2);
    const uint32_t num_o_tiles = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_state_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_o_out = tt::CBIndex::c_17;
    const uint32_t state_tile_bytes = get_tile_size(cb_state_out);
    const uint32_t o_tile_bytes = get_tile_size(cb_o_out);

    // Compile-time-arg blocks are appended in program-factory order: state_out, then o_out.
    constexpr auto state_args = TensorAccessorArgs<0>();
    const auto state_gen = TensorAccessor(state_args, state_out_addr, state_tile_bytes);
    constexpr auto o_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    const auto o_gen = TensorAccessor(o_args, o_out_addr, o_tile_bytes);

    Noc noc;
    CircularBuffer cb_state_out_o(cb_state_out);
    CircularBuffer cb_o_out_o(cb_o_out);

    for (uint32_t t = 0; t < num_state_tiles; t++) {
        cb_state_out_o.wait_front(1);
        noc.async_write(cb_state_out_o, state_gen, state_tile_bytes, {.offset_bytes = 0}, {.page_id = t});
        noc.async_write_barrier();
        cb_state_out_o.pop_front(1);
    }

    for (uint32_t t = 0; t < num_o_tiles; t++) {
        cb_o_out_o.wait_front(1);
        noc.async_write(cb_o_out_o, o_gen, o_tile_bytes, {.offset_bytes = 0}, {.page_id = t});
        noc.async_write_barrier();
        cb_o_out_o.pop_front(1);
    }
}
