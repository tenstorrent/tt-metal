// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    ArgFetcher arg_fetcher;
    const uint32_t src_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t num_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t mask_h = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t mask_w = arg_fetcher.get_next_arg_val<uint32_t>();
    const bool do_mask_h = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    const bool do_mask_w = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);

    constexpr auto src_args = TensorAccessorArgs<0>();

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_mask_h_w = 2;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    DataflowBuffer dfb_scaler(cb_id_scaler);
    fill_cb_with_value(dfb_scaler, scaler.u);

    if (do_mask_h || do_mask_w) {
        DataflowBuffer dfb_mask_h_w(cb_id_mask_h_w);
        generate_mask_h_w(dfb_mask_h_w, mask_h, mask_w);
    }

    const auto s0 = TensorAccessor(src_args, src_addr);

    Noc noc;
    DataflowBuffer dfb_in0(cb_id_in0);
    const auto in0_tile_bytes = get_tile_size(cb_id_in0);

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        dfb_in0.reserve_back(onetile);
        noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
    }
}
