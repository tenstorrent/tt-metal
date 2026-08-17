// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    auto N = get_arg(args::N);
    auto tile_offset = get_arg(args::tile_offset);
    auto Wt = get_arg(args::Wt);

    auto mask_w = get_arg(args::mask_w);

    uint32_t l1_write_addr_in;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const auto y_in = TensorAccessor(tensor::y);
    const auto dy_in = TensorAccessor(tensor::dy);

    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    DataflowBuffer dfb_mask_obj(dfb::mask);
    generate_mask_w(dfb_mask_obj, mask_w);

    Noc noc;
    DataflowBuffer dfb_y_obj(dfb::y);
    DataflowBuffer dfb_dy_obj(dfb::dy);
    const auto y_tile_bytes = dfb_y_obj.get_tile_size();
    const auto dy_tile_bytes = dfb_dy_obj.get_tile_size();

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i += onetile) {
        uint32_t curr_offset_i = curr_tile;
        for (uint32_t w = 0; w < Wt; w++) {
#ifndef LOG
            dfb_y_obj.reserve_back(onetile);
            noc.async_read(y_in, dfb_y_obj, y_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_y_obj.push_back(onetile);
#endif

            dfb_dy_obj.reserve_back(onetile);
            noc.async_read(dy_in, dfb_dy_obj, dy_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_dy_obj.push_back(onetile);

            curr_tile++;
        }

        curr_tile = curr_offset_i;
        for (uint32_t w = 0; w < Wt; w++) {
            dfb_y_obj.reserve_back(onetile);
            noc.async_read(y_in, dfb_y_obj, y_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_y_obj.push_back(onetile);

            dfb_dy_obj.reserve_back(onetile);
            noc.async_read(dy_in, dfb_dy_obj, dy_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_dy_obj.push_back(onetile);

            curr_tile++;
        }
    }
}
