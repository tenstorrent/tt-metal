// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

namespace {

// The sequence's varargs: [call_count][call 0][call 1].
static_assert(get_compile_time_arg_val(0) == 2);

// The factory's logical DFB IDs, in order: in_a, in_b, auxiliary, accumulator, output.
template <uint32_t Index>
using CallAt = ttnn::kernel_lib::BoundReduceCallArgs<
    ttnn::kernel_lib::ReduceCallAtT<0, Index>,
    dfb::in_a,
    dfb::in_b,
    dfb::auxiliary,
    dfb::accumulator,
    dfb::out>;

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup(CallAt<0>::input_cb_id, CallAt<0>::output_cb_id);
    compute_kernel_lib::reduce<CallAt<0>>();
    compute_kernel_lib::reduce<CallAt<1>>();
}
