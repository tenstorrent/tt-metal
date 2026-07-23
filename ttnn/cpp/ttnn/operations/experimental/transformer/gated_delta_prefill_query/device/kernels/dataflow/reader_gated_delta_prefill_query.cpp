// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SCAFFOLDING reader for the gated-delta prefill-then-query op.
// Streams `state` tiles into cb_in: first `num_state_tiles` feed the state
// passthrough, then `num_o_tiles` more feed the placeholder output token.
// The real kernel will also read q/k/v/gate/decay here.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_state_tiles = get_arg_val<uint32_t>(1);
    const uint32_t num_o_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    const uint32_t tile_bytes = get_tile_size(cb_in);

    constexpr auto state_args = TensorAccessorArgs<0>();
    const auto state_gen = TensorAccessor(state_args, state_addr, tile_bytes);

    Noc noc;
    CircularBuffer cb_in_o(cb_in);

    // Pass 1: full state -> state passthrough.
    for (uint32_t t = 0; t < num_state_tiles; t++) {
        cb_in_o.reserve_back(1);
        noc.async_read(state_gen, cb_in_o, tile_bytes, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in_o.push_back(1);
    }

    // Pass 2: placeholder source tiles for the output token.
    for (uint32_t t = 0; t < num_o_tiles; t++) {
        cb_in_o.reserve_back(1);
        noc.async_read(state_gen, cb_in_o, tile_bytes, {.page_id = t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in_o.push_back(1);
    }
}
