// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto dst_args = TensorAccessorArgs<0>();

    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = 16;

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_out_obj(cb_id_out);
    const auto out_tile_bytes = get_tile_size(cb_id_out);

    uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        dfb_out_obj.wait_front(onetile);

        uint32_t l1_read_addr = dfb_out_obj.get_read_ptr();
        volatile tt_l1_ptr int32_t* out_l1_ptr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(l1_read_addr);

        for (uint32_t w = 0; w < 16; w++) {
            for (uint32_t subh = 1; subh < 4; ++subh) {
                out_l1_ptr[w] += out_l1_ptr[w + (16 * subh)];
                out_l1_ptr[w + 256] += out_l1_ptr[w + 256 + (16 * subh)];
            }
        }

        noc.async_write(dfb_out_obj, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        dfb_out_obj.pop_front(onetile);
    }
}
