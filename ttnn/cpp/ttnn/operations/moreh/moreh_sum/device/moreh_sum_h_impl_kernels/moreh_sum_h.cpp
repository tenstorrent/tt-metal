// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/kernel/compute/moreh_common.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_H = get_compile_time_arg_val(3);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    DataflowBuffer dfb_scaler_obj(cb_scaler);
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t TILE_H = 32;
    constexpr bool do_mask_h = (origin_H % TILE_H) != 0;

    binary_op_init_common(cb_input, cb_input, cb_out);

    // Non-tile-aligned H: the reader emits a full scaler (tile 0) and a partial scaler (tile 1).
    // The reduce helper applies tile 1 to the LAST H tile of each column, so the padding rows are
    // multiplied by zero and contribute nothing. This replaces the previous workaround, which
    // copied the last tile into DST, applied mask_tile against a separate 0/1 mask CB, packed the
    // result into a scratch CB, and folded it in with a second accumulating reduce.
    constexpr auto partial_scaler = do_mask_h ? compute_kernel_lib::ReducePartialScaler::last_tile_at(1)
                                              : compute_kernel_lib::ReducePartialScaler::none();

    // tiles arrive in NCWH order (H-contiguous), so one column's Ht tiles stream in back-to-back
    // and collapse into a single output tile.
    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, cb_input, cb_scaler, cb_out>(
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
