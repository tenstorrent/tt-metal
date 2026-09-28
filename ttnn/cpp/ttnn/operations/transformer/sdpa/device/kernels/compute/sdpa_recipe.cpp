// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Dense and joint SDPA precision recipes (host: sdpa_recipe.cpp). FAST (A) runs the legacy streaming loop;
// STANDARD, BALANCED, ACCURATE and LOW_PRECISION (B-E) run recipe_streaming.hpp. The host selects the
// recipe with SDPA_RECIPE_BASELINE / _FP32 / _ACCURATE / _LOFI; the two loops define the same names, so
// exactly one is included.
//
// Optimization: FAST keeps the legacy kernel's default. B-E compile unpack/math at -O2 and pack at -Os, which
// keeps every geometry inside the kernel config buffer. Watcher builds are size-optimized on every thread.
#if defined(WATCHER_ENABLED) || (!defined(SDPA_RECIPE_BASELINE) && defined(TRISC_PACK))
#define SDPA_RECIPE_OPTIMIZE_PUSHED
#pragma GCC push_options
#pragma GCC optimize("Os")
#elif !defined(SDPA_RECIPE_BASELINE)
#define SDPA_RECIPE_OPTIMIZE_PUSHED
#pragma GCC push_options
#pragma GCC optimize("O2")
#endif

#include "api/compute/compute_kernel_hw_startup.h"
#include "compute_common.hpp"
#include "streaming/recipe_tail.hpp"

#ifdef SDPA_RECIPE_BASELINE
#include "compute_streaming.hpp"
#else
#include "streaming/recipe_sfpu.hpp"
#include "streaming/recipe_streaming.hpp"
#endif

namespace {
constexpr uint32_t k_chunks = get_named_compile_time_arg_val("k_chunks");
constexpr uint32_t scale = get_named_compile_time_arg_val("scale");
constexpr uint32_t q_tiles = get_named_compile_time_arg_val("q_tiles");
constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
// QK and PV matmul subblock widths: the largest of 4, 2, 1 dividing the K chunk and head dim.
constexpr uint32_t qk_subblock_w = SDPA_RECIPE_QK_W;
constexpr uint32_t pv_subblock_w = SDPA_RECIPE_PV_W;
static_assert(q_tiles >= 1 && k_tiles >= 1 && d_tiles >= 1);
static_assert(k_tiles % qk_subblock_w == 0 && d_tiles % pv_subblock_w == 0);

// Circular buffers (host: recipe_compute_program).
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_identity_scale = 3, cb_col_identity = 4, cb_recip_scratch = 5;
constexpr uint32_t cb_qk = 6, cb_out_a = 8, cb_out_b = 9, cb_max_a = 10, cb_max_b = 11, cb_sum_a = 12,
                   cb_sum_b = 13, cb_exp_max_diff = 14, cb_mask = 15, cb_out = 16;

#ifdef SDPA_RECIPE_BASELINE
#ifdef SDPA_RECIPE_MASK
constexpr bool has_mask = true;
#else
constexpr bool has_mask = false;
#endif

// FAST: the legacy streaming loop. Odd Q chunks use single-row subblocks; subblock height only changes
// which rows share a dest pass, not any element's accumulation. The mask is L1-accumulated before the max.
void recipe_run(uint32_t jobs) {
    constexpr uint32_t subblock_h = q_tiles % 2 == 0 ? 2 : 1;
    sdpa_standard_v2<
        q_tiles, k_tiles, k_tiles * k_chunks, d_tiles, d_tiles, scale,
        subblock_h, qk_subblock_w, subblock_h, pv_subblock_w,
        /*use_padded_mask=*/false,
        cb_q, cb_k, cb_v, cb_qk, cb_identity_scale, cb_exp_max_diff, cb_col_identity, cb_recip_scratch, cb_out,
        cb_mask,
        /*sliding_window_size=*/0, /*is_causal=*/false, /*use_attention_sink=*/false, INVALID_CB,
        /*use_provided_mask=*/has_mask>(jobs, k_chunks, cb_out_a, cb_out_b, cb_max_a, cb_max_b, cb_sum_a, cb_sum_b);
}
#else
// B-E: FP32 recipes process one Q tile row per group; paired BF16 recipes two (an odd chunk ends with one).
void recipe_run(uint32_t jobs) {
#ifdef SDPA_RECIPE_FP32
    constexpr uint32_t subblock_h = 1;
#else
    constexpr uint32_t subblock_h = 2;
#endif
    sdpa_standard_v2<
        q_tiles, k_tiles, d_tiles, d_tiles, scale,
        subblock_h, qk_subblock_w, subblock_h, pv_subblock_w,
        cb_q, cb_k, cb_v, cb_qk, cb_identity_scale, cb_exp_max_diff, cb_col_identity, cb_recip_scratch,
        cb_out>(jobs, k_chunks, cb_out_a, cb_out_b, cb_max_a, cb_max_b, cb_sum_a, cb_sum_b);
}
#endif
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

#ifdef SDPA_RECIPE_OPTIMIZE_PUSHED
#pragma GCC pop_options
#endif
