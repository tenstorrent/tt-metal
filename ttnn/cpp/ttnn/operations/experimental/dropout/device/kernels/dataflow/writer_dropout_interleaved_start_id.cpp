// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    Noc noc;

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    CircularBuffer cb_out(cb_id_out);

#ifdef OUT_SHARDED
    cb_out.wait_front(num_tiles);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(dst_args, dst_addr);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb_out.wait_front(onetile);
        uint32_t l1_read_addr = cb_out.get_read_ptr();
        noc.async_write(CoreLocalMem<uint32_t>(l1_read_addr), s, get_tile_size(cb_id_out), {}, {.page_id = i});
        noc.async_write_barrier();
        cb_out.pop_front(onetile);
    }
#endif
}
