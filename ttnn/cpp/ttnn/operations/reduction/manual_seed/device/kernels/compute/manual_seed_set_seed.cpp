// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/rand.h"

#include <cstdint>

void kernel_main() {
    // Get compile time args
    constexpr uint32_t seed = get_compile_time_arg_val(0);

    // A seed value of -1 (0xFFFFFFFF when interpreted as uint32_t) is a special value
    // that skips rand_tile_init, leaving the PRNG state unchanged.
    // Other negative int32_t values are not supported.
    if (static_cast<int32_t>(seed) != -1) {
        rand_tile_init(seed);
    }
}
