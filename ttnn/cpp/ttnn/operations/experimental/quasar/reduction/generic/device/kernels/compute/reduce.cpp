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

#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    using Call = ttnn::kernel_lib::BoundReduceCallArgs<
        ttnn::kernel_lib::ReduceCallAtT<5, 0>,
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3>;
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
    compute_kernel_lib::reduce<Call>(
#ifdef REDUCE_POST_MUL
        [](uint32_t dst_idx) {
            constexpr uint32_t bits = get_compile_time_arg_val(3);
            constexpr DataFormat format = static_cast<DataFormat>(unpack_src_format[tt::CBIndex::c_0]);
            compute_kernel_lib::detail::reduce_post_mul_tile<format>(dst_idx, bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );
    DataflowBuffer(tt::CBIndex::c_2).pop_front(get_compile_time_arg_val(16));
}
