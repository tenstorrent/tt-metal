// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// Same reduce helper as kernels/compute/reduce.cpp. Row-major W is fed in W-chunks to cb_rm; we tilize each chunk into
// cb_tile_in, then call reduce with WaitAndPopPerTile over Wt tiles (block shape 1×Wt×1).
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

#ifdef REDUCE_POST_MUL
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#endif

void kernel_main() {
    const uint32_t rows_per_core = get_compile_time_arg_val(0);
    const uint32_t Wt = get_compile_time_arg_val(1);
    const uint32_t wt_tiles_per_chunk = get_compile_time_arg_val(2);

    constexpr uint32_t cb_rm = tt::CBIndex::c_24;
    constexpr uint32_t cb_tile_in = tt::CBIndex::c_0;

    // First op is tilize (RM → tiles); reduce() uses INPUT reconfig for unpack/packer tile formats.
    compute_kernel_hw_startup(cb_rm, cb_tile_in);

    for (uint32_t row = 0; row < rows_per_core; row++) {
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += wt_tiles_per_chunk) {
            const uint32_t wt_in_chunk = (wt_base + wt_tiles_per_chunk < Wt) ? wt_tiles_per_chunk : (Wt - wt_base);

            tilize_init(cb_rm, wt_in_chunk, cb_tile_in);

            cb_wait_front(cb_rm, 1);
            cb_reserve_back(cb_tile_in, wt_in_chunk);
            tilize_block(cb_rm, wt_in_chunk, cb_tile_in);
            cb_pop_front(cb_rm, 1);
            cb_push_back(cb_tile_in, wt_in_chunk);

            tilize_uninit(cb_rm, cb_tile_in);
        }

        compute_kernel_lib::reduce<
            REDUCE_OP,
            REDUCE_DIM,
            compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
            compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
            tt::CBIndex::c_0,
            tt::CBIndex::c_2,
            tt::CBIndex::c_3,
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
    }
}
