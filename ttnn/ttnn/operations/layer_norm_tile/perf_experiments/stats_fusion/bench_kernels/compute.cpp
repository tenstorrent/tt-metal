// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// stats_fusion bench compute: per iteration (one tile-row) center -> variance stage (STATS_VARIANT,
// from stats_variants.hpp) -> [MODE 0] normalize, exactly as layer_norm_tile_compute.cpp issues them.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "../stats_variants.hpp"

void kernel_main() {
    constexpr uint32_t cb_z = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scaler = get_compile_time_arg_val(2);
    constexpr uint32_t cb_work_b = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sumsq = get_compile_time_arg_val(4);
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t iterations = get_compile_time_arg_val(8);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(9);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(10);
    constexpr uint32_t dest_lanes = get_compile_time_arg_val(11);
    constexpr uint32_t mode = get_compile_time_arg_val(12);

    using namespace compute_kernel_lib;
    compute_kernel_hw_startup(cb_z, cb_mean, cb_out);

    DataflowBuffer z(cb_z), mean(cb_mean), work_b(cb_work_b), sumsq(cb_sumsq), inv_std(cb_inv_std), out(cb_out);
    constexpr auto lane_in = [](uint32_t cb) {
        return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block);
    };
    constexpr auto lane_out = [](uint32_t cb) {
        return output(cb, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize);
    };
    const auto lane_shape = IterationShape::grid(1, Wt).block_size(dest_lanes);

    for (uint32_t it = 0; it < iterations; ++it) {
        {
            MaybeDeviceZoneScope("bench_center");
            sub<lane_in(cb_z),
                input(cb_mean, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_out(cb_work_b)>(lane_shape, z, mean, work_b);
        }
        ln_stats::inv_std_block<cb_work_b, cb_sumsq, cb_scaler, cb_inv_std, Wt, inv_w_bits, epsilon_bits>(
            1, work_b, sumsq, inv_std);
        if constexpr (mode == 0) {
            MaybeDeviceZoneScope("bench_normalize");
            mul<lane_in(cb_work_b),
                input(cb_inv_std, BroadcastDim::Col, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Col),
                lane_out(cb_out)>(lane_shape, work_b, inv_std, out);
        }
    }
}
