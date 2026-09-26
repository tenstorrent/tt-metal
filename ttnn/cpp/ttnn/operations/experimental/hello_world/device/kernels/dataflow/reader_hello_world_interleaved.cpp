// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    // Per-core runtime args: (src_addr, num_tiles, start_id)
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const auto s = TensorAccessor(src_args, src_addr);
    CircularBuffer cb_in0(cb_id_in0);

    // Single-tile ublocks: reserve one tile, read one tile, push one tile.
    constexpr uint32_t onetile = 1;
    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_in0.reserve_back(onetile);
        uint32_t l1_write_addr = cb_in0.get_write_ptr();
        noc.async_read(s, CoreLocalMem<uint32_t>(l1_write_addr), get_tile_size(cb_id_in0), {.page_id = i}, {});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
    }
}
