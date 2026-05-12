// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t Ht = get_arg_val<uint32_t>(3);
    uint32_t Wt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(out_args, dst_addr);

    uint32_t blk = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb_out_obj(cb_id_out);
    const auto out_tile_bytes = get_tile_size(cb_id_out);

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i++) {
        uint32_t w_idx = curr_tile % Wt;
        uint32_t nc_idx = curr_tile / Wt;
        uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;
        for (uint32_t h = 0; h < Ht; h++) {
            cb_out_obj.wait_front(blk);
            noc.async_write(cb_out_obj, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_idx});
            noc.async_write_barrier();
            cb_out_obj.pop_front(blk);
            tile_idx += Wt;
        }
        curr_tile += 1;
    }
}
