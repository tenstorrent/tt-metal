// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated Sinkhorn micro-benchmark (mhc_pre perf tournament E2). One core, one fp32 coefficient-major tile
// (resident sharded L1, UnpackToDestFp32): REPS x { copy_tile -> DEST tile 0, sinkhorn variant } -> pack.
// CT args: [variant, reps]; RT args: [eps_bits, iters].
// Variants live in sinkhorn_variants.hpp (variant 0 = the op's current code, verbatim).

#include <stdint.h>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"

#ifdef SKB_ZONE
#include "tools/profiler/kernel_profiler.hpp"
#endif

constexpr uint32_t n_streams = 4;

#ifdef TRISC_MATH
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sinkhorn_variants.hpp"
#endif

constexpr uint32_t variant = get_compile_time_arg_val(0);
constexpr uint32_t reps = get_compile_time_arg_val(1);
constexpr uint32_t second_half = get_compile_time_arg_val(2);  // 1: run in the second DEST half (a dummy window first)

void kernel_main() {
    constexpr uint32_t cb_in = 0, cb_out = 16;
    const uint32_t eps_bits = get_arg_val<uint32_t>(0);
    const uint32_t iters = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_in, 1);
    cb_push_back(cb_in, 1);
    compute_kernel_hw_startup(cb_in, cb_out);
    cb_wait_front(cb_in, 1);
    cb_reserve_back(cb_out, 1);
    copy_tile_to_dst_init_short(cb_in);
    if constexpr (second_half) {
        tile_regs_acquire();
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }
    tile_regs_acquire();
    for (uint32_t r = 0; r < reps; ++r) {
        copy_tile(cb_in, 0, 0);
        MATH((ckernel::llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
        {
#ifdef SKB_ZONE
            DeviceZoneScopedN("skb");
#endif
            MATH((_llk_math_eltwise_unary_sfpu_params_(skb::run<variant>, 0, VectorMode::None, eps_bits, iters)));
        }
        copy_tile_to_dst_init_short(cb_in);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
