// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Named dense recipes share the same reader, writer and outer-loop contract.
// Numerical choices are fixed by the host policy, not user-visible defines.
#if defined(WATCHER_ENABLED)
// Watcher instrumentation otherwise exceeds the instruction buffer for the
// compensated loop. Keep the qualified release scheduling unchanged.
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
    const uint32_t jobs = get_arg_val<uint32_t>(0);
    static_assert(q_tiles >= 4 && q_tiles <= 10, "Named recipes support Q128-Q320");
    // Odd Q chunks use single-row QK/PV subblocks for FAST; subblock height only
    // changes which rows share a dest pass, not any element's accumulation.
    constexpr uint32_t bf16_subblock_h = q_tiles % 2 == 0 ? 2 : 1;
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    cb_wait_front(0, q_tiles * 4);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    {
        DeviceZoneScopedN("SDPA_RECIPE");
        sdpa_standard_v2<
            q_tiles,
            16,
#ifdef SDPA_RECIPE_BASELINE
            16 * k_chunks,
#endif
            4,
            4,
            scale,
#ifdef SDPA_RECIPE_FP32
            1,
            4,
            1,
            4,
#else
            bf16_subblock_h,
            4,
            bf16_subblock_h,
            4,
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

#if defined(SDPA_RECIPE_ACCURATE) || defined(WATCHER_ENABLED)
#pragma GCC pop_options
#endif
