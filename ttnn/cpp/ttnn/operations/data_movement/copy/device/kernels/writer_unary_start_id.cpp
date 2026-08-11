// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    Noc noc;
    DataflowBuffer dfb_out(cb_id_out);

#ifdef OUT_SHARDED
    dfb_out.wait_front(num_tiles);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);

    // TensorAccessor handles both interleaved and sharded outputs.
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto s = TensorAccessor(dst_args, dst_addr);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_tiles;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        dfb_out.wait_front(onetile);
        noc.async_write(dfb_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = i, .offset_bytes = 0});
        noc.async_write_barrier();
        dfb_out.pop_front(onetile);
    }
#endif
}
