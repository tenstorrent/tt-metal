// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t intermed_cb_id = get_compile_time_arg_val(0);

    const std::uint32_t seed = get_arg_val<std::uint32_t>(0);
    const std::uint32_t start_id = get_arg_val<std::uint32_t>(1);
    const std::uint32_t num_tiles = get_arg_val<std::uint32_t>(2);
    const std::uint32_t end_id = start_id + num_tiles;

    CircularBuffer cb_intermed(intermed_cb_id);

    init_sfpu(intermed_cb_id, intermed_cb_id);

    // rand_tile's internal interval is inclusive. Use the largest FP32 value
    // below 1.0 so that Bernoulli's `random < probability` comparison still
    // produces one for every draw when probability is exactly 1.0.
    constexpr std::uint32_t rand_from = 0;
    constexpr std::uint32_t rand_scale = 0x3F7FFFFFU;

    rand_tile_init(seed, start_id);
    for (std::uint32_t i = start_id; i < end_id; ++i) {
        cb_intermed.reserve_back(1);

        tile_regs_acquire();
        rand_tile(0, rand_from, rand_scale);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, intermed_cb_id, 0);
        tile_regs_release();

        cb_intermed.push_back(1);
    }
}
