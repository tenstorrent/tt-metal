// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t Ht = get_arg(args::Ht);
    // Carries the per-core work-split count (the host's num_cols_per_core_group_N), not a tile width.
    uint32_t Wt = get_arg(args::units_per_core);
    uint32_t NC = get_arg(args::NC);
    constexpr uint32_t origin_H = get_arg(args::origin_H);

    DataflowBuffer dfb_scaler_obj(dfb::scaler);
    constexpr uint32_t TILE_H = 32;
    constexpr bool do_mask_h = (origin_H % TILE_H) != 0;

    compute_kernel_hw_startup(dfb::input, dfb::input, dfb::out);

    constexpr auto partial_scaler = do_mask_h ? compute_kernel_lib::ReducePartialScaler::with_partial()
                                              : compute_kernel_lib::ReducePartialScaler::none();

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::input, dfb::scaler, dfb::out>(
                compute_kernel_lib::ReduceInputBlockShape::col(Ht),
                compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
                compute_kernel_lib::NoAccumulation{},
                compute_kernel_lib::NoOp{},
                partial_scaler);
        }
    }

    constexpr uint32_t num_scaler_tiles = do_mask_h ? 2 : 1;
    dfb_scaler_obj.pop_front(num_scaler_tiles);
}
