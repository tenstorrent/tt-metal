// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Legacy entry point retained until all consumers migrate to Metalium 2.0. Keep this shell free of
// algorithmic logic; both entry points share pool_2d_compute_impl.

#include "ttnn/cpp/ttnn/operations/pool/generic/device/kernels/compute/pool_2d_compute_impl.hpp"

void kernel_main() {
    pool_2d_compute_impl<
        get_compile_time_arg_val(0),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15),
        get_compile_time_arg_val(16),
        get_compile_time_arg_val(38)>(get_arg_val<uint32_t>(0));
}
