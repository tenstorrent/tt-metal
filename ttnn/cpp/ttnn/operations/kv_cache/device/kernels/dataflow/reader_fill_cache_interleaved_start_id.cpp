// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0;

    CircularBuffer cb_in0(cb_id_in0);

#ifdef INPUT_SHARDED
    cb_in0.reserve_back(num_tiles);
    cb_in0.push_back(num_tiles);
#else
    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    Noc noc;
    const auto s = TensorAccessor(src_args, src_addr);
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

// read a ublock of tiles from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_in0.reserve_back(onetile);
        noc.async_read(s, cb_in0, tile_bytes, {.page_id = i}, {});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
    }
#endif
}
