// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t col_start_tile_id =
        get_arg(args::col_start_tile_id);  // Start id in column major order. This should be the start of a column
    uint32_t curr_col_in_batch = get_arg(args::curr_col_in_batch);
    uint32_t num_cols = get_arg(args::num_cols);  // number of cols to read
    uint32_t mask_h = get_arg(args::mask_h);

    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t HtWt = get_arg(args::HtWt);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

#ifdef REDUCE_SCALER
    constexpr uint32_t reduce_factor = get_arg(args::reduce_factor);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        dfb::scaler,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_COL,
        reduce_factor>();
#endif

#ifdef DO_MASK_H
    DataflowBuffer dfb_mask_h(dfb::mask_h);
    generate_mask_h(dfb_mask_h, mask_h);
#endif

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::input);
    const auto in0_tile_bytes = dfb_in0.get_tile_size();

    uint32_t w = curr_col_in_batch;

    for (uint32_t i = 0; i < num_cols; i++) {
        uint32_t curr_id = col_start_tile_id;
        for (uint32_t j = 0; j < Ht; j++) {
            dfb_in0.reserve_back(onetile);
            noc.async_read(s, dfb_in0, in0_tile_bytes, {.page_id = curr_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in0.push_back(onetile);
            curr_id += Wt;  // stride in H
        }
        w++;
        if (w == Wt) {
            col_start_tile_id = curr_id - Wt + 1;
            w = 0;
        } else {
            col_start_tile_id++;
        }
    }
}
