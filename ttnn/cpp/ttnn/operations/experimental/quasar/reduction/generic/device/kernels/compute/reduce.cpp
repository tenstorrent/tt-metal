// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Thin wrapper around compute_kernel_lib::reduce<>. The input data format is deduced from the input
// CB id inside the helper, so Int32 MAX is routed to the SFPU path automatically; otherwise
// FPU/GMPOOL. MIN on Int32 is dispatched separately via reduce_{h,w}_neg.

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

    compute_kernel_lib::reduce<
        REDUCE_OP,
        REDUCE_DIM,
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT>(
        compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
        compute_kernel_lib::ReduceInputMemoryLayout::contiguous(),
        compute_kernel_lib::NoAccumulation{},
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN and SFPU reduce ignores the
        // scaler CB entirely, so both paths apply the user scalar here per output tile.
        // reduce_post_mul_tile handles Int32 (typecast-bracketed) and float formats uniformly.
        [](uint32_t dst_idx) {
            constexpr uint32_t post_mul_scaler_bits = get_compile_time_arg_val(3);
            constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[tt::CBIndex::c_0]);
            compute_kernel_lib::detail::reduce_post_mul_tile<reduce_format>(dst_idx, post_mul_scaler_bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );

    // The reduce helper waits on the scaler CB but never pops it (the single scaler tile is
    // reused for the whole reduction). Pop it here so the CB is left balanced.
    DataflowBuffer(tt::CBIndex::c_2).pop_front(1);
}
