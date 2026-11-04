// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    // The factory's logical DFB IDs, in order: input, auxiliary, output.
    using Call = ttnn::kernel_lib::
        BoundReduceCallArgs<ttnn::kernel_lib::ReduceCallArgs<0>, dfb::input, dfb::auxiliary, dfb::out>;

    compute_kernel_hw_startup(Call::input_cb_id, Call::output_cb_id);
    compute_kernel_lib::reduce<Call>();
}
