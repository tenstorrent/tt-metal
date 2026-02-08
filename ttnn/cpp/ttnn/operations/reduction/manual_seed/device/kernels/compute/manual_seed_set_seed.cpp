// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/rand.h"

void kernel_main() {
    // Get compile time args
    constexpr uint32_t seed = get_compile_time_arg_val(0);

    // Set random generator with seed
    rand_tile_init(seed);
}
