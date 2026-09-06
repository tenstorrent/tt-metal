// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// UnaryBcast coverage for each hardware broadcast dimension. The element owns
// the per-op short init; compute_kernel_hw_startup owns engine-wide setup.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t n = get_compile_time_arg_val(0);
    constexpr uint32_t dim = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_in, cb_out);

    using namespace compute_kernel_lib;
    if constexpr (dim == 0) {
        unary_bcast<BroadcastDim::None, input(cb_in), output(cb_out)>(IterationShape::tiles(n));
    } else if constexpr (dim == 1) {
        unary_bcast<BroadcastDim::Col, input(cb_in), output(cb_out)>(IterationShape::tiles(n));
    } else if constexpr (dim == 2) {
        unary_bcast<BroadcastDim::Row, input(cb_in), output(cb_out)>(IterationShape::tiles(n));
    } else {
        unary_bcast<BroadcastDim::Scalar, input(cb_in), output(cb_out)>(IterationShape::tiles(n));
    }
}
