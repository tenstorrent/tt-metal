// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Named dense recipes share the same reader, writer and outer-loop contract.
// Numerical choices are fixed by the host policy, not user-visible defines.
#if defined(WATCHER_ENABLED) || defined(SDPA_RECIPE_SIZE_OPTIMIZED)
// Watcher instrumentation (or the paired recipes' extra single-row tail group on
// odd Q chunks) otherwise exceeds the kernel config buffer. Even-chunk release
// builds keep the qualified scheduling.
#pragma GCC push_options
#pragma GCC optimize("Os")
#elif defined(SDPA_RECIPE_ACCURATE)
#pragma GCC push_options
#pragma GCC optimize("O2")
#endif

#include "api/compute/compute_kernel_hw_startup.h"
#ifdef SDPA_RECIPE_LOFI
#include "streaming/lofi_scaling.hpp"
#endif
#include "compute_common.hpp"
#include "streaming/recipe_tail.hpp"

#ifdef SDPA_RECIPE_BASELINE
#include "compute_streaming.hpp"
#else
#include "streaming/recipe_sfpu.hpp"
#include "streaming/recipe_streaming.hpp"
#endif

void kernel_main() {
    constexpr uint32_t k_chunks = get_compile_time_arg_val(0);
    constexpr uint32_t scale = get_compile_time_arg_val(1);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t k_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t d_tiles = get_compile_time_arg_val(4);
    static_assert(d_tiles == 2 || d_tiles == 4 || d_tiles == 8, "Named recipes support D64/D128/D256");
    // PV subblocks are at most four tiles wide and must divide the head dim.
    constexpr uint32_t pv_subblock_w = d_tiles < 4 ? d_tiles : 4;
    static_assert(k_tiles == 8 || k_tiles == 12 || k_tiles == 16, "Named recipes support K256/K384/K512");
    const uint32_t jobs = get_arg_val<uint32_t>(0);
    static_assert(q_tiles >= 4 && q_tiles <= 10, "Named recipes support Q128-Q320");
#ifdef SDPA_RECIPE_BASELINE
    // FAST uses single-row QK/PV subblocks for odd Q chunks; subblock height only
    // changes which rows share a dest pass, not any element's accumulation.
    constexpr uint32_t bf16_subblock_h = q_tiles % 2 == 0 ? 2 : 1;
#else
    // Paired BF16 recipes keep two-row groups and end an odd chunk with a one-row group.
    constexpr uint32_t bf16_subblock_h = 2;
#endif
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    cb_wait_front(0, q_tiles * d_tiles);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    {
        DeviceZoneScopedN("SDPA_RECIPE");
        sdpa_standard_v2<
            q_tiles,
            k_tiles,
#ifdef SDPA_RECIPE_BASELINE
            k_tiles * k_chunks,
#endif
            d_tiles,
            d_tiles,
            scale,
#ifdef SDPA_RECIPE_FP32
            1,
            4,
            1,
            pv_subblock_w,
#else
            bf16_subblock_h,
            4,
            bf16_subblock_h,
            pv_subblock_w,
#endif
#ifdef SDPA_RECIPE_BASELINE
            false,
#endif
            0,
            1,
            2,
            6,
            3,
            14,
            4,
            5,
            16
#ifdef SDPA_RECIPE_BASELINE
            ,
            15
#endif
            >(jobs, k_chunks, 8, 9, 10, 11, 12, 13);
    }
}

#if defined(SDPA_RECIPE_ACCURATE) || defined(WATCHER_ENABLED) || defined(SDPA_RECIPE_SIZE_OPTIMIZED)
#pragma GCC pop_options
#endif
