// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/generate_mm_scaler.hpp"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);
    uint32_t mask_w = get_arg(args::mask_w);
    constexpr uint32_t scaler = get_arg(args::scaler);

    DataflowBuffer dfb_in2(dfb::scaler);
    generate_mm_scaler(dfb_in2, scaler);

#ifdef DO_MASK_W
    DataflowBuffer dfb_mask_w(dfb::mask_w);
    generate_mask_w(dfb_mask_w, mask_w);
#endif

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::input);
    const auto in0_tile_bytes = dfb_in0.get_tile_size();

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        dfb_in0.reserve_back(onetile);
        noc.async_read(s, dfb_in0, in0_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
    }
}
