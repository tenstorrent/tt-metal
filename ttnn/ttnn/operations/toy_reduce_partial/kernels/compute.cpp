// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    static_assert(get_compile_time_arg_val(0) == 1, "toy_reduce_partial expects one planned call");
    using Call = ttnn::kernel_lib::ReduceCallAtT<1, 0>;
    compute_kernel_hw_startup(Call::input_cb_id, Call::auxiliary_cb_id, Call::output_cb_id);
    compute_kernel_lib::reduce<Call>();
    cb_pop_front(Call::auxiliary_cb_id, Call::auxiliary_tile_count);
}
