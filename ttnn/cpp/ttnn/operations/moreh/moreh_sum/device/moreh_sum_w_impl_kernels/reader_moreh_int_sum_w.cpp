// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr auto src_args = TensorAccessorArgs<0>();

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t mask_w = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_mask_w = 1;
#ifdef DO_MASK_W
    DataflowBuffer dfb_mask_w_obj(cb_id_mask_w);
    generate_mask_w<int32_t>(dfb_mask_w_obj, mask_w);
#endif

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in0_obj(cb_id_in0);
    const auto in0_tile_bytes = get_tile_size(cb_id_in0);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        dfb_in0_obj.reserve_back(onetile);
        noc.async_read(s, dfb_in0_obj, in0_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0_obj.push_back(onetile);
    }
}
