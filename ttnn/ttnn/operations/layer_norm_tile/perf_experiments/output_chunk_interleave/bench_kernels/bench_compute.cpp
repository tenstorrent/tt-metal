// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated output-stage bench: c and inv_std staged once into the op's work / inv_std CBs, then the
// selected output-stage variant runs `reps` times on the resident data. Reps > 0 re-publish the same pages
// (reserve + push, no data movement), so only rep 0's output is meaningful; timing is data-independent.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "../oci_variants.hpp"

void kernel_main() {
    constexpr uint32_t cb_c_src = get_compile_time_arg_val(0);
    constexpr uint32_t cb_inv_src = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(3);
    constexpr uint32_t cb_wa = get_compile_time_arg_val(4);
    constexpr uint32_t cb_wb = get_compile_time_arg_val(5);
    constexpr uint32_t cb_wc = get_compile_time_arg_val(6);
    constexpr uint32_t cb_inv = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t rows = get_compile_time_arg_val(9);
    constexpr uint32_t tensor_w_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t reps = get_compile_time_arg_val(11);
    constexpr uint32_t variant = get_compile_time_arg_val(12);
    constexpr uint32_t dest_lanes = get_compile_time_arg_val(13);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(14);
    constexpr uint32_t num_tiles = rows * tensor_w_tiles;

    using namespace compute_kernel_lib;
    compute_kernel_hw_startup(cb_c_src, cb_gamma, cb_out);

    DataflowBuffer c_src(cb_c_src), inv_src(cb_inv_src), gamma_tiles(cb_gamma), beta_tiles(cb_beta);
    DataflowBuffer work_a(cb_wa), work_b(cb_wb), work_c(cb_wc), inv_std(cb_inv), output_tiles(cb_out);

    copy<input(cb_c_src, WaitPolicy::PerTile, PopPolicy::PerTile), output(cb_wb)>(
        IterationShape::tiles(num_tiles), c_src, work_b);
    copy<input(cb_inv_src, WaitPolicy::PerTile, PopPolicy::PerTile), output(cb_inv)>(
        IterationShape::tiles(rows), inv_src, inv_std);

    for (uint32_t rep = 0; rep < reps; ++rep) {
        if (rep != 0) {
            work_b.reserve_back(num_tiles);
            work_b.push_back(num_tiles);
            inv_std.reserve_back(rows);
            inv_std.push_back(rows);
        }
        MaybeDeviceZoneScope("compute_output_chunked");
        oci::output_stage<
            variant,
            cb_wa,
            cb_wb,
            cb_wc,
            cb_inv,
            cb_gamma,
            cb_beta,
            cb_out,
            tensor_w_tiles,
            dest_lanes,
            chunk_tiles>(rows, work_a, work_b, work_c, inv_std, gamma_tiles, beta_tiles, output_tiles);
    }
    gamma_tiles.pop_front(tensor_w_tiles);
    beta_tiles.pop_front(tensor_w_tiles);
}
