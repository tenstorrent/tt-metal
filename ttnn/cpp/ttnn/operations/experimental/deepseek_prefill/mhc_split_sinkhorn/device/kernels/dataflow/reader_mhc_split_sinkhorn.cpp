// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Loads the 8 host-prepared constant tiles once (persistent), then streams the mixes
// token-tiles. See the program factory for the CB layout.
void kernel_main() {
    const uint32_t mixes_addr = get_arg_val<uint32_t>(0);
    const uint32_t consts_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_token_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_tile = get_arg_val<uint32_t>(3);  // this core's first mixes page

    constexpr uint32_t cb_mixes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_consts = get_compile_time_arg_val(1);
    constexpr auto mixes_args = TensorAccessorArgs<2>();
    constexpr auto consts_args = TensorAccessorArgs<mixes_args.next_compile_time_args_offset()>();

    constexpr uint32_t num_const_tiles = 8;

    const uint32_t mixes_page = get_local_cb_interface(cb_mixes).fifo_page_size;
    const uint32_t consts_page = get_local_cb_interface(cb_consts).fifo_page_size;

    const auto s_mixes = TensorAccessor(mixes_args, mixes_addr);
    const auto s_consts = TensorAccessor(consts_args, consts_addr);

    Noc noc;
    CircularBuffer cb_c(cb_consts);
    CircularBuffer cb_m(cb_mixes);

    // Constants: load once, never popped by compute -> resident for every token-tile.
    for (uint32_t i = 0; i < num_const_tiles; ++i) {
        cb_c.reserve_back(1);
        noc.async_read(s_consts, cb_c, consts_page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_c.push_back(1);
    }

    for (uint32_t t = 0; t < num_token_tiles; ++t) {
        cb_m.reserve_back(1);
        noc.async_read(s_mixes, cb_m, mixes_page, {.page_id = start_tile + t}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_m.push_back(1);
    }
}
