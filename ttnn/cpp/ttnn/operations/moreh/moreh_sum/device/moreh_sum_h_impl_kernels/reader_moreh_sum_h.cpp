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
    uint32_t col_start_tile_id =
        get_arg(args::col_start_tile_id);  // Start id in column major order. This should be the start of a column
    uint32_t curr_col_in_batch = get_arg(args::curr_col_in_batch);
    uint32_t num_cols = get_arg(args::num_cols);  // number of cols to read
    // args::mask_h is still supplied by the host but no longer read: the ragged last tile is
    // handled by the partial scaler below rather than by a generated 0/1 mask tile.

    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t HtWt = get_arg(args::HtWt);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

#ifdef REDUCE_SCALER
#ifdef DO_MASK_H
    // Non-tile-aligned H: emit a full scaler (tile 0) plus a partial scaler (tile 1) holding the
    // scaler in only the first partial_h rows. Compute applies tile 1 to the last H tile of each
    // column. partial_h is origin_H % TILE_HEIGHT, so it is always in [1, 31] here.
    constexpr uint32_t partial_h = get_arg(args::partial_h);
    dataflow_kernel_lib::calculate_and_prepare_partial_reduce_scalers<
        dfb::scaler,
        ckernel::PoolType::SUM,
        ckernel::ReduceDim::REDUCE_COL,
        partial_h>();
#else
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();
#endif
#endif

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_in0_obj(dfb::input);
    const auto in0_tile_bytes = dfb_in0_obj.get_tile_size();

    uint32_t w = curr_col_in_batch;

    for (uint32_t i = 0; i < num_cols; i++) {
        uint32_t curr_id = col_start_tile_id;
        for (uint32_t j = 0; j < Ht; j++) {
            dfb_in0_obj.reserve_back(onetile);
            noc.async_read(s, dfb_in0_obj, in0_tile_bytes, {.page_id = curr_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in0_obj.push_back(onetile);
            curr_id += Wt;
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
