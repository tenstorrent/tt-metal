// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// NC is not a separate loop here: the host splits work into num_rows = N*C*H pages (one RM page per logical row).
// ReduceInputBlockShape::of(1, Wt, 1) uses batches=1 because each iteration consumes one row's tiles (Wt wide).
//
// compute_kernel_hw_startup must run exactly once at kernel entry, before tilize_init / tilize_block (see
// api/compute/compute_kernel_hw_startup.h). Calling it mid-kernel or after tilize is unsafe. Pack dest must be
// configured before the first tilize_block or llk_math_wait_for_dest_available can deadlock (same pattern as
// layernorm_large_tensor.cpp + binary_op_init_common before tilize).
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    const uint32_t rows_per_core = get_compile_time_arg_val(0);
    const uint32_t Wt = get_compile_time_arg_val(1);

    constexpr uint32_t cb_rm = tt::CBIndex::c_24;
    constexpr uint32_t cb_tile_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_tile_out = tt::CBIndex::c_3;

    // Match reduce.cpp: one HW startup before any tilize or reduce_*_init.
    compute_kernel_hw_startup(cb_tile_in, cb_scaler, cb_tile_out);

    for (uint32_t row = 0; row < rows_per_core; row++) {
        tilize_init(cb_rm, Wt, cb_tile_in);

        cb_wait_front(cb_rm, 1);
        cb_reserve_back(cb_tile_in, Wt);
        tilize_block(cb_rm, Wt, cb_tile_in);
        cb_pop_front(cb_rm, 1);
        cb_push_back(cb_tile_in, Wt);

        tilize_uninit(cb_rm, cb_tile_in);

        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
            cb_tile_in,
            cb_scaler,
            cb_tile_out,
            compute_kernel_lib::ReduceInputBlockShape::of(1, Wt, 1),
            compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
            compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
            [](uint32_t dst_idx) {
                constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);
                binop_with_scalar_tile_init();
                mul_unary_tile(dst_idx, post_mul_scaler_bits);
            }
#else
            compute_kernel_lib::NoOp{}
#endif
        );
        (void)row;
    }
}
