// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Thin wrapper around compute_kernel_lib::reduce<>. The host always defines REDUCE_FORMAT
// (the input data format); compute_kernel_lib::reduce<> dispatches to SFPU when
// REDUCE_FORMAT is Int32/Float32 and REDUCE_OP is MAX, otherwise FPU/GMPOOL.
// MIN on Int32/Float32 is dispatched separately via reduce_sfpu_{h,w}_neg.

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        REDUCE_FORMAT,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN and SFPU reduce ignores the
        // scaler CB entirely, so both paths apply the user scalar here per output tile.
        // reduce_post_mul_tile applies typecast-bracketed mul for Int32, plain mul_unary_tile otherwise.
        [](uint32_t dst_idx) {
            constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);
            compute_kernel_lib::detail::reduce_post_mul_tile<REDUCE_FORMAT>(dst_idx, post_mul_scaler_bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );
}
