// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t tile_offset = get_arg_val<uint32_t>(2);
    uint32_t Wt = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_out = tt::CBIndex::c_16;
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_out);

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(out_args, dst_addr);

    Noc noc;
    DataflowBuffer dfb_out_obj(cb_id_out);

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < N; i++) {
        dfb_out_obj.wait_front(Wt);
        for (uint32_t w = 0; w < Wt; w++) {
            noc.async_write(dfb_out_obj, s, tile_bytes, {.offset_bytes = w * tile_bytes}, {.page_id = tile_id});
            tile_id++;
        }
        noc.async_write_barrier();
        dfb_out_obj.pop_front(Wt);
    }
}
