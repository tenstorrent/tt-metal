// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/buffer_compat.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp"  // matmul_config::InitMode

namespace compute_kernel_lib {

namespace detail {

/**
 * MatmulSubblockStep: the FMA primitive consumed by `matmul_kloop_pack` via the
 * `step` token its KStep functors receive. Not callable as a free function — a
 * stateless `MatmulSubblockStep` instance is constructed inside
 * `matmul_kloop_pack` and passed by reference into the kloop's per-step
 * functors. Equivalent to a single `ckernel::matmul_block(...)` call with three
 * pieces of helper hygiene:
 *   1. `ASSERT(ct_dim * rt_dim <= DEST_AUTO_LIMIT)` and `ASSERT(kt_dim > 0)`.
 *   2. `SKIP_COMPUTE` guard for microbench harnesses.
 *   3. Buffer abstraction (CircularBuffer or DataflowBuffer; uses `buf_id()`).
 *
 * Precision: same FPU arithmetic as the inner FMA loop of matmul_block, so the
 * "Precision (host-side ComputeConfig guidance)" block on matmul_block applies.
 * In particular, the HiFi4 + fp32_dest_acc_en=True KNOWN-BAD combo on Wormhole
 * B0 (#38306) is just as bad through this primitive.
 *
 * Init dispatch:
 *   InitMode::None  (default) — caller's outer mm_block_init covers state.
 *   InitMode::Short — emit `mm_block_init_short` ahead of the FMA. Use when
 *                     this call follows a different op or a compatible matmul
 *                     in a different shape. Awkward template-arg syntax:
 *                     `step.template operator()<InitMode::Short>(...)`. If a
 *                     real caller surfaces, add a `step.short_init(...)` wrapper.
 *   InitMode::Full  — rejected at compile time (no out_cb_id to bind).
 */
struct MatmulSubblockStep {
    template <matmul_config::InitMode init_mode = matmul_config::InitMode::None, typename Buf>
    ALWI void operator()(
        Buf& in0_buf,
        Buf& in1_buf,
        uint32_t in0_index,
        uint32_t in1_index,
        uint32_t dst_index,
        bool transpose,
        uint32_t ct_dim,
        uint32_t rt_dim,
        uint32_t kt_dim) const;
};

}  // namespace detail

/**
 * Shape for the segmented K-loop pattern shared across all four ring-aware
 * MoE / DeepSeek MLA kernels: an outer loop of `num_blocks` iterations, each
 * waiting / popping `tiles_per_block` tiles on an in1 CB and running an
 * inner stride loop of FMAs at stride `ct_dim`.
 *
 * Build with designated initializers — the optional fields rarely all apply
 * at once:
 *
 *   SegmentedKLoopShape{
 *       .num_blocks = ..., .tiles_per_block = ..., .ct_dim = ...,
 *   }
 *
 * Required:
 *   num_blocks       Outer block count (= cb_wait_front / cb_pop_front cycles).
 *   tiles_per_block  Tiles consumed per cb_wait_front / cb_pop_front. Producer
 *                    pushes this many tiles per block.
 *   ct_dim           Output sub-block width in tiles, also the in1_index stride
 *                    per FMA step (assumes one output sub-block per
 *                    matmul_kloop_pack call, which holds for all four migrated
 *                    kernels).
 *
 * Optional:
 *   rt_dim           Output sub-block height in tiles. Default 1.
 *   kt_dim           K extent per FMA call. Default 1 (one K-tile per step).
 *   last_block_tiles When > 0, helper runs ONE additional partial block after
 *                    the main K-loop: cb_wait_fronts `tiles_per_block` and
 *                    iterates only up to `last_block_tiles` (< tiles_per_block).
 *                    Used by moe_gate_mm where the last block has trailing
 *                    bias tiles read separately via copy_tile.
 *   last_block_no_pop
 *                    When true, helper skips the cb_pop_front for the partial
 *                    last block; in1 stays fronted after the helper returns
 *                    (caller is responsible for the eventual pop). Used for
 *                    post-K-loop copy_tile of the bias (moe_gate_mm non-send).
 */
struct SegmentedKLoopShape {
    uint32_t num_blocks;
    uint32_t tiles_per_block;
    uint32_t ct_dim;
    uint32_t rt_dim = 1;
    uint32_t kt_dim = 1;
    uint32_t last_block_tiles = 0;
    bool last_block_no_pop = false;

    static constexpr SegmentedKLoopShape of(
        uint32_t num_blocks,
        uint32_t tiles_per_block,
        uint32_t ct_dim,
        uint32_t rt_dim = 1,
        uint32_t kt_dim = 1,
        uint32_t last_block_tiles = 0,
        bool last_block_no_pop = false) {
        return {num_blocks, tiles_per_block, ct_dim, rt_dim, kt_dim, last_block_tiles, last_block_no_pop};
    }
};

/**
 * KStepDefault: per-FMA-step functor that fires one regular FMA from a single
 * (in0, in1) pair, auto-incrementing in0_index. The simplest per-step body —
 * mla_wo and moe_gate_mm use this directly.
 *
 * Mutable functor: `in0_index` advances by 1 per call. Construct fresh on
 * each matmul_kloop_pack invocation so state resets correctly.
 */
template <typename Buf>
struct KStepDefault {
    Buf& in0_buf;
    Buf& in1_buf;
    uint32_t in0_index = 0;
    bool transpose = false;

    ALWI void operator()(
        detail::MatmulSubblockStep& step, SegmentedKLoopShape shape, uint32_t /*block_id*/, uint32_t k);
};

/**
 * KStepWithBias: per-FMA-step functor that fires regular FMAs up to bias_at
 * K-tiles, then ONE bias FMA (`ones_buf × in1_buf` at the bias K position),
 * then skips remaining padding K-slots. Used by moe_compute / moe_gpt W0/W1
 * with bias.
 *
 * Mutable functor: in0_index and k_tracker advance per call. Construct fresh
 * per matmul_kloop_pack invocation.
 */
template <typename Buf>
struct KStepWithBias {
    Buf& in0_buf;
    Buf& in1_buf;
    Buf& bias_buf;
    uint32_t in0_index = 0;
    uint32_t k_tracker = 0;
    uint32_t bias_at;
    bool transpose = false;

    ALWI void operator()(
        detail::MatmulSubblockStep& step, SegmentedKLoopShape shape, uint32_t /*block_id*/, uint32_t k);
};

/**
 * Result of a ring-step lookup: the in0 read offset and the tile budget for a
 * given ring slot. Returned by `RingStepFn` callbacks consumed by KStepWithRing.
 */
struct RingStepResult {
    uint32_t in0_index;
    uint32_t tiles_remaining;
};

/**
 * KStepWithRing: per-FMA-step functor that combines KStepWithBias's bias
 * interleaving / padding skip with a ring CB sync. When the active ring slot's
 * tile budget hits 0, pops the ring CB, waits for the next slot, increments
 * the ring step index, and calls `ring_step_fn(step_idx)` to look up the new
 * in0_index and tile budget for that slot.
 *
 * The ring CB's INITIAL token must be fronted by the caller before invoking
 * the matmul_kloop_pack scope; the FINAL ring CB pop (one extra to balance
 * the initial wait) is also caller-side, after matmul_kloop_pack returns.
 *
 * Mutable functor: many fields advance per call. Construct fresh per
 * matmul_kloop_pack invocation.
 *
 * The two ring kernels differ in how in0_index is advanced across slots; the
 * RingStepFn callback hides that:
 *   moe_compute (monotonic):   return {step_idx * tiles_per_step, ring_tiles[step_idx]};
 *   moe_gpt    (6-buf cyclic): return {(step_idx % 6) * tiles_per_step, ring_tiles[step_idx]};
 */
template <bool HasBias, typename Buf, typename RingStepFn>
struct KStepWithRing {
    Buf& in0_buf;
    Buf& in1_buf;
    Buf& bias_buf;
    Buf& ring_cb_buf;
    uint32_t in0_index;
    uint32_t k_tracker = 0;
    uint32_t bias_at;
    uint32_t dm1_step = 0;
    uint32_t dm1_tiles_remaining;
    RingStepFn ring_step_fn;
    bool transpose = false;

    ALWI void operator()(
        detail::MatmulSubblockStep& step, SegmentedKLoopShape shape, uint32_t /*block_id*/, uint32_t k);
};

/**
 * SimplePack: the canonical pack body for callers whose K-loop output is a
 * straight `pack_tile_block(0, out_buf, dst_count)` of the accumulated DST
 * run. Used by mla_wo and any caller not doing custom mid-DST work or
 * out-of-order multi-CB packs. Owns its own `tile_regs_wait` (per the
 * matmul_kloop_pack contract that the pack_body brings the wait).
 *
 * Construct inline at the call site with CTAD:
 *   matmul_kloop_pack(in1_buf, shape, k_step, SimplePack{out_buf, dst_count});
 */
template <typename Buf>
struct SimplePack {
    Buf& out_buf;
    uint32_t dst_count;
    ALWI void operator()() const;
};

namespace detail {

/**
 * matmul_segmented_kloop: drive the segmented K-loop pattern from inside
 * `matmul_kloop_pack`. Internal — kernels reach the K-loop through
 * `matmul_kloop_pack` only. Iterates `num_blocks` outer cb_wait/pop cycles on
 * `in1_cb_buf`; within each block runs an inner stride loop calling
 * `k_step(step, shape, block_id, k)` at every FMA position. Optionally runs a
 * final partial block when `shape.last_block_tiles > 0`.
 */
template <typename Buf, typename KStepFn>
ALWI void matmul_segmented_kloop(MatmulSubblockStep& step, Buf& in1_cb_buf, SegmentedKLoopShape shape, KStepFn& k_step);

}  // namespace detail

/**
 * matmul_kloop_pack: SOLE public entry point for the four ring-aware MoE /
 * DeepSeek MLA kernels. Owns the DST scope (`tile_regs_acquire` →
 * `tile_regs_commit` → caller's pack body → `tile_regs_release`), the
 * segmented K-loop (delegated to `detail::matmul_segmented_kloop`), and an
 * optional post-K compute body that runs after the K-loop and before
 * `tile_regs_commit` for MATH-thread SFPU work that mutates DST after the
 * matmul accumulator is built (sigmoid, bias copy_tile, etc.).
 *
 * Required includes:
 *   #include "api/compute/compute_kernel_hw_startup.h"
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_kloop_helpers.hpp"
 *
 * ── Parameters ────────────────────────────────────────────────────────────
 *
 *   in1_buf       in1 buffer driving the segmented K-loop (cb_wait/pop).
 *   shape         SegmentedKLoopShape — block count, tiles per block, ct/rt/kt
 *                 dim, optional partial last block.
 *   k_step        Per-FMA-step functor (KStepDefault / KStepWithBias /
 *                 KStepWithRing, or any caller-defined struct with the same
 *                 operator() signature). Taken by value but its mutable state
 *                 (in0_index, k_tracker, ring counters) advances internally.
 *   pack_body     Required. Callable with no args; runs after `tile_regs_commit`.
 *                 MUST call `tile_regs_wait()` (or a custom variant — moe_compute
 *                 uses a STALL_CFG semwait) before any pack work. Use
 *                 `SimplePack{out_buf, dst_count}` for the common
 *                 `pack_tile_block` case (mla_wo).
 *   post_k_body   Optional (default `NoOp{}`). Callable with no args; runs
 *                 after the K-loop and BEFORE `tile_regs_commit`. Use for
 *                 MATH-thread SFPU work on the matmul accumulator (sigmoid,
 *                 bias copy_tile, partial-add, etc. — moe_gate_mm non-send).
 *
 * ── Examples ──────────────────────────────────────────────────────────────
 *
 * @example
 *   // mla_wo: simple K-loop, simple pack_tile_block.
 *   KStepDefault k_step{in0_buf, in1_buf, in0_index_base};
 *   matmul_kloop_pack(in1_buf, iter_shape, k_step,
 *                     SimplePack{out_buf, num_n_tiles_per_iter});
 *
 * @example
 *   // moe_compute W0/W1: K-loop with bias FMA + custom pack with PACK-thread SFPU.
 *   KStepWithBias k_step{
 *       .in0_buf = in_buf, .in1_buf = w_buf, .bias_buf = ones_buf,
 *       .in0_index = in0_index, .bias_at = num_w0_w1_tiles_h};
 *   matmul_kloop_pack(w_buf, w0_w1_kloop_shape, k_step,
 *       [&] {
 *           PACK(TTI_SEMWAIT(...));
 *           detail::pack_compute_activation<activation>();
 *           pack_tile<true>(0, cb_s2c_in2, tile_id);
 *           pack_tile<true>(2, cb_s2c_in2, tile_id + 1);
 *       });
 *
 * @example
 *   // moe_gate_mm non-send: K-loop + post-K SFPU + multi-CB pack.
 *   KStepDefault k_step{in0_buf, in1_buf};
 *   matmul_kloop_pack(in1_buf,
 *       SegmentedKLoopShape{ ... .last_block_no_pop = true },
 *       k_step,
 *       NonSendPack{},                                        // pack_body
 *       NonSendPostK{.bias_tile_index = bias_tile_index});    // post_k_body
 */
template <typename Buf, typename KStepFn, typename PackBody, typename PostKBody = NoOp>
ALWI void matmul_kloop_pack(
    Buf& in1_buf, SegmentedKLoopShape shape, KStepFn k_step, PackBody pack_body, PostKBody post_k_body = NoOp{});

}  // namespace compute_kernel_lib

#include "matmul_kloop_helpers.inl"
