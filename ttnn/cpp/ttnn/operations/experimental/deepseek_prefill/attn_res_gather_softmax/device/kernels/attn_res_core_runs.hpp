// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A fold core's share of the two passes, derived on the core from its own position.
//
// The split is a function of the shapes alone, so as runtime args it would be the
// same few words shipped to every core at every launch: on Galaxy, ~110 cores times
// three kernels times 32 chips, most of what dispatching a read cost the host. Both
// splits hand cores out row-wise from the origin, which makes a core's run a closed
// form in its linear index; the host checks the closed form against the split it
// computed at program build.
//
// Include after the kernel's API header, which supplies the logical coordinates.

#pragma once

#include <cstdint>

struct AttnResCoreRuns {
    uint32_t num_stat_rows;
    uint32_t stat_start_row;
    uint32_t num_fold_tiles;
    uint32_t fold_start_tile;
};

// The first `group_1` cores take `per_1` units each and the rest `per_2`.
constexpr uint32_t attn_res_run_start(uint32_t i, uint32_t group_1, uint32_t per_1, uint32_t per_2) {
    return i < group_1 ? i * per_1 : group_1 * per_1 + (i - group_1) * per_2;
}

constexpr uint32_t attn_res_run_length(uint32_t i, uint32_t group_1, uint32_t per_1, uint32_t per_2) {
    return i < group_1 ? per_1 : per_2;
}

// The split's compile-time args start at `Base`, in the order the program factory
// appends them: grid width, statistics cores, statistics group-1 cores and the two
// rows-per-core counts, fold group-1 cores and the two tiles-per-core counts.
template <uint32_t Base>
inline AttnResCoreRuns attn_res_core_runs() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(Base);
    constexpr uint32_t num_stat_cores = get_compile_time_arg_val(Base + 1);
    constexpr uint32_t stat_group_1 = get_compile_time_arg_val(Base + 2);
    constexpr uint32_t rows_1 = get_compile_time_arg_val(Base + 3);
    constexpr uint32_t rows_2 = get_compile_time_arg_val(Base + 4);
    constexpr uint32_t fold_group_1 = get_compile_time_arg_val(Base + 5);
    constexpr uint32_t tiles_1 = get_compile_time_arg_val(Base + 6);
    constexpr uint32_t tiles_2 = get_compile_time_arg_val(Base + 7);

    const uint32_t i = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const bool stats = i < num_stat_cores;
    return {
        .num_stat_rows = stats ? attn_res_run_length(i, stat_group_1, rows_1, rows_2) : 0,
        .stat_start_row = stats ? attn_res_run_start(i, stat_group_1, rows_1, rows_2) : 0,
        .num_fold_tiles = attn_res_run_length(i, fold_group_1, tiles_1, tiles_2),
        .fold_start_tile = attn_res_run_start(i, fold_group_1, tiles_1, tiles_2),
    };
}
