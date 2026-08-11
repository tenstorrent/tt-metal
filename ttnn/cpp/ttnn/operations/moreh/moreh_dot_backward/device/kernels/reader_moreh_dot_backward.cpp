// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t has_input_grad = get_arg(args::has_input_grad);
    uint32_t has_other_grad = get_arg(args::has_other_grad);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t onetile = 1;

    const auto s0 = TensorAccessor(tensor::s0);
    const auto s1 = TensorAccessor(tensor::s1);
    const auto s2 = TensorAccessor(tensor::s2);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);
    const auto in0_tile_bytes = dfb_in0.get_entry_size();
    const auto in1_tile_bytes = dfb_in1.get_entry_size();
    const auto in2_tile_bytes = dfb_in2.get_entry_size();

    dfb_in0.reserve_back(onetile);
    noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    dfb_in0.push_back(onetile);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        if (has_input_grad) {
            dfb_in2.reserve_back(onetile);
            noc.async_read(s2, dfb_in2, in2_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in2.push_back(onetile);
        }

        if (has_other_grad) {
            dfb_in1.reserve_back(onetile);
            noc.async_read(s1, dfb_in1, in1_tile_bytes, {.page_id = i}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in1.push_back(onetile);
        }
    }
}
