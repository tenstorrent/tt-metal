// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Thin wrapper around compute_kernel_lib::reduce<>. The input data format is deduced from the input
// buffer inside the helper, so Int32 MAX, MIN and SUM are routed to the SFPU path automatically;
// otherwise FPU/GMPOOL. Accurate fp32 also uses the SFPU; fast-mode float/bf16 MIN is lowered to
// -MAX(-x) via reduce_{h,w}_neg on the host.

#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    using Call =
        ttnn::kernel_lib::BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallAtT<1, 0>, dfb::in0, dfb::scaler, dfb::out>;
    compute_kernel_hw_startup(dfb::in0, dfb::scaler, dfb::out);

    compute_kernel_lib::reduce<Call>(
#ifdef REDUCE_POST_MUL
        // GMPOOL only respects the scaler's exponent for MAX/MIN and SFPU reduce ignores the
        // scaler buffer entirely, so both paths apply the user scalar here per output tile.
        // reduce_post_mul_tile handles Int32 (typecast-bracketed) and float formats uniformly.
        [](uint32_t dst_idx) {
            constexpr auto post_mul_scaler_bits = get_arg(args::post_mul_scaler_bits);
            // The data format has to be a constant expression here (it is a template argument), so it
            // is read from the JIT descriptor array indexed by the DFB handle rather than off a
            // DataflowBuffer object: DataflowBuffer's constructor is not constexpr, so no such object
            // is usable in a constant expression.
            constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[dfb::in0]);
            compute_kernel_lib::detail::reduce_post_mul_tile<reduce_format>(dst_idx, post_mul_scaler_bits);
        }
#else
        compute_kernel_lib::NoOp{}
#endif
    );

    DataflowBuffer(dfb::scaler).pop_front(Call::auxiliary_tile_count);
}
