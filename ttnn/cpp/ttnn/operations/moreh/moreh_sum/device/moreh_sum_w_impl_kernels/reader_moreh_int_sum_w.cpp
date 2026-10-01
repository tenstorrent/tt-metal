// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t mask_w = get_arg(args::mask_w);

#ifdef DO_MASK_W
    DataflowBuffer dfb_mask_w_obj(dfb::mask_w);
    generate_mask_w<int32_t>(dfb_mask_w_obj, mask_w);
#endif

    constexpr uint32_t onetile = 1;
    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in0_obj(dfb::input);
    const auto in0_tile_bytes = dfb_in0_obj.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        dfb_in0_obj.reserve_back(onetile);
        noc.async_read(s, dfb_in0_obj, in0_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0_obj.push_back(onetile);
    }
}
