// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Dense and joint SDPA precision recipes (host: sdpa_recipe.cpp): STANDARD, BALANCED, ACCURATE and FAST
// (B-E) run recipe_streaming.hpp. The host selects the recipe through named compile-time arguments
// (streaming/recipe_config.hpp).
//
// Optimization: the default level (-O3), except ACCURATE on Blackhole at -O2 (the host's kernel opt level) and every
// Wormhole build at -O2 (the per-architecture level below): Wormhole images are larger (SFPU launcher, four packers),
// and at -O3 they overflow the kernel config buffer (BALANCED: its chunked prefill build with a sink and a
// device-tensor start, 70768 B). The host disables the SFPI compiler's replay optimization
// (tenstorrent/tt-metal#58433). Watcher builds are size-optimized.
#if defined(WATCHER_ENABLED)
#pragma GCC optimize("Os")
#elif defined(ARCH_WORMHOLE)
#pragma GCC optimize("O2")
#endif

#include "api/compute/compute_kernel_hw_startup.h"
#include "sdpa_block_ops.hpp"
#include "streaming/recipe_tail.hpp"

#include "streaming/recipe_sfpu.hpp"
#include "streaming/recipe_streaming.hpp"

namespace {
constexpr uint32_t k_chunks = get_named_compile_time_arg_val("k_chunks");
constexpr uint32_t scale = get_named_compile_time_arg_val("scale");
constexpr uint32_t q_tiles = get_named_compile_time_arg_val("q_tiles");
constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
// V and output head dim: narrower than Q/K's for MLA.
constexpr uint32_t vd_tiles = get_named_compile_time_arg_val("vd_tiles");
// QK and PV matmul subblock widths: the largest of 4, 2, 1 dividing the K chunk and V head dim.
constexpr uint32_t qk_subblock_w = recipe::qk_w;
constexpr uint32_t pv_subblock_w = recipe::pv_w;
static_assert(q_tiles >= 1 && k_tiles >= 1 && d_tiles >= 1 && vd_tiles >= 1 && vd_tiles <= d_tiles);
static_assert(k_tiles % qk_subblock_w == 0 && vd_tiles % pv_subblock_w == 0);

// Circular buffers (host: recipe_compute_program).
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_identity_scale = 3, cb_col_identity = 4, cb_recip_scratch = 5;
constexpr uint32_t cb_qk = 6, cb_out_a = 8, cb_out_b = 9, cb_max_a = 10, cb_max_b = 11, cb_sum_a = 12,
                   cb_sum_b = 13, cb_exp_max_diff = 14, cb_mask = 15, cb_out = 16;

// B-E: FP32 recipes process one Q tile row per group; paired BF16 recipes two (the host pads odd chunks).
void recipe_run(uint32_t jobs) {
    constexpr uint32_t subblock_h = recipe::fp32 ? 1 : 2;
    sdpa_standard_v2<
        q_tiles,
        k_tiles,
        d_tiles,
        vd_tiles,
        scale,
        subblock_h,
        qk_subblock_w,
        subblock_h,
        pv_subblock_w,
        cb_q,
        cb_k,
        cb_v,
        cb_qk,
        cb_identity_scale,
        cb_exp_max_diff,
        cb_col_identity,
        cb_recip_scratch,
        cb_out>(jobs, k_chunks, cb_out_a, cb_out_b, cb_max_a, cb_max_b, cb_sum_a, cb_sum_b);
}
}  // namespace

void kernel_main() {
    const uint32_t jobs = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q, cb_k, cb_out);
    matmul_init(cb_q, cb_k);
    cb_wait_front(cb_q, q_tiles * d_tiles);
    cb_wait_front(cb_identity_scale, 1);
    cb_wait_front(cb_col_identity, 1);
    DeviceZoneScopedN("SDPA_RECIPE");
    recipe_run(jobs);
}
