// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

// Sharded-input reader: mixes are already resident in this core's L1 (the mixes CB aliases
// the shard), so we only fetch the 8 constant tiles from DRAM and signal the mixes as ready.
void kernel_main() {
    const uint32_t consts_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_token_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_mixes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_consts = get_compile_time_arg_val(1);
    constexpr auto consts_args = TensorAccessorArgs<2>();
    constexpr uint32_t num_const_tiles = 8;

    const uint32_t consts_page = get_local_cb_interface(cb_consts).fifo_page_size;
    const auto s_consts = TensorAccessor(consts_args, consts_addr);

    Noc noc;
    CircularBuffer cb_c(cb_consts);
    CircularBuffer cb_m(cb_mixes);

    for (uint32_t i = 0; i < num_const_tiles; ++i) {
        cb_c.reserve_back(1);
        noc.async_read(s_consts, cb_c, consts_page, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_c.push_back(1);
    }

    cb_m.push_back(num_token_tiles);  // mixes already in L1 via the aliased CB
}
