// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "experimental/kernel_args.h"

#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    using Call =
        ttnn::kernel_lib::BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, 0>, dfb::in, dfb::scaler, dfb::out>;
    compute_kernel_hw_startup(dfb::in, dfb::scaler, dfb::out);
    compute_kernel_lib::reduce<Call>(
#ifdef REDUCE_POST_MUL
        [](uint32_t dst_idx) {
            constexpr uint32_t bits = get_arg(args::post_mul_scaler_bits);
            constexpr DataFormat format = static_cast<DataFormat>(unpack_src_format[dfb::in]);
            compute_kernel_lib::detail::reduce_post_mul_tile<format>(dst_idx, bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );
    DataflowBuffer(dfb::scaler).pop_front(get_arg(args::auxiliary_tiles));
}
