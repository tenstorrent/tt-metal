// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Ensure ALWI is available for forward declarations even when this header
// is the first include in a kernel file (before common_globals.h).
#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

/**
 * @file matmul_helpers_compute.hpp
 * @brief Runtime-config matmul compute helpers for SDPA and dynamic patterns
 *
 * Complements the compile-time matmul_block_helpers.hpp/matmul_tile_helpers.hpp
 * with runtime-configurable matmul operations using MatmulConfig and MatmulMode.
 *
 * Use matmul_block_helpers.hpp for standard blocked matmul kernels (compile-time
 * CB params, full K-loop automation). Use this file for:
 *   - SDPA kernels that need absolute-offset packing
 *   - Kernels that need runtime CB selection
 *   - Reduce-W patterns via matmul
 *   - Building blocks for custom matmul patterns (detail:: namespace)
 *
 * ## MatmulMode
 *
 * Template parameter controlling the underlying LLK function:
 * - TILE: calls matmul_tiles (per-tile, 5 args) — for simple/streaming kernels
 * - BLOCK: calls matmul_block (per-block, 9 args) — for production blocked kernels
 *
 * ## Usage Examples
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp"
 *   using namespace compute_kernel_lib;
 *
 *   // Example 1: SDPA matmul with fused recip (PostComputeFn)
 *   auto cfg = MatmulConfig::block(sum_cb, identity_cb, scratch_cb, 1, 1, 1);
 *   matmul_init_short<BLOCK>(cfg);
 *   struct RecipFn { ALWI void operator()(uint32_t) const { recip_tile_init(); ... } };
 *   matmul_single_and_pack<BLOCK>(cfg, 0, 0, scratch_cb, RecipFn{});
 *
 *   // Example 2: SDPA blocked matmul with causal mask (PostComputeFn)
 *   auto cfg = MatmulConfig::block(q_cb, k_cb, qk_cb, sbw, sbh, block_w, true);
 *   struct MaskFn { uint32_t m, z; ALWI void operator()(uint32_t n) const { ... } };
 *   matmul_blocks_absolute<BLOCK>(cfg, M, N, K, nsub0, nsub1, MaskFn{m,z});
 *
 *   // Example 3: Reduce-W via matmul (replaces reduce_w.cpp)
 *   auto cfg = MatmulConfig::tile(cb_in0, cb_in1, cb_out);
 *   matmul_init<TILE>(cfg);
 *   matmul_reduce_w_and_pack<TILE>(cfg, Wt, 0, cb_out);
 *
 *   // Example 4: Full automated tile-mode matmul
 *   auto cfg = MatmulConfig::tile(cb_in0, cb_in1, cb_out);
 *   matmul_init<TILE>(cfg);
 *   matmul<TILE>(cfg, MatmulBlockShape::of(batch, Mt, Nt, Kt, 1, 1, 1, 1, 1));
 */

namespace compute_kernel_lib {

// =============================================================================
// MatmulMode — selects tile vs block LLK functions
// =============================================================================

enum class MatmulMode { TILE, BLOCK };

// Convenience aliases for template arguments
inline constexpr MatmulMode TILE = MatmulMode::TILE;
inline constexpr MatmulMode BLOCK = MatmulMode::BLOCK;

// =============================================================================
// MatmulConfig — runtime configuration for matmul operations
// =============================================================================

/**
 * @brief Configuration for matmul operations
 *
 * Holds CB IDs, block dimensions, transpose flag, and optional partials CB.
 * Block dimensions (ct_dim, rt_dim, kt_dim) are only used in BLOCK mode.
 *
 * Create via factory methods for clarity:
 *   MatmulConfig::tile(in0, in1, out)              // tile mode defaults
 *   MatmulConfig::block(in0, in1, out, ct, rt, kt) // block mode with dims
 */
struct MatmulConfig {
    uint32_t in0_cb_id;
    uint32_t in1_cb_id;
    uint32_t out_cb_id;

    uint32_t ct_dim = 1;  // subblock_w
    uint32_t rt_dim = 1;  // subblock_h
    uint32_t kt_dim = 1;  // in0_block_w

    bool transpose = false;
    uint32_t partials_cb_id = 0;

    static constexpr MatmulConfig tile(uint32_t in0, uint32_t in1, uint32_t out, bool trans = false) {
        return {in0, in1, out, 1, 1, 1, trans, 0};
    }

    static constexpr MatmulConfig block(
        uint32_t in0,
        uint32_t in1,
        uint32_t out,
        uint32_t ct,
        uint32_t rt,
        uint32_t kt,
        bool trans = false,
        uint32_t partials = 0) {
        return {in0, in1, out, ct, rt, kt, trans, partials};
    }
};

// =============================================================================
// MatmulBlockShape — dimensions for full automated matmul
// =============================================================================

struct MatmulBlockShape {
    uint32_t batch;
    uint32_t num_blocks_h;
    uint32_t num_blocks_w;
    uint32_t num_blocks_inner;
    uint32_t in0_num_subblocks;
    uint32_t in1_num_subblocks;
    uint32_t in0_block_num_tiles;
    uint32_t in1_block_num_tiles;
    uint32_t in1_block_w;

    static constexpr MatmulBlockShape of(
        uint32_t b,
        uint32_t bh,
        uint32_t bw,
        uint32_t bi,
        uint32_t in0_nsub,
        uint32_t in1_nsub,
        uint32_t in0_btiles,
        uint32_t in1_btiles,
        uint32_t in1_bw) {
        return {b, bh, bw, bi, in0_nsub, in1_nsub, in0_btiles, in1_btiles, in1_bw};
    }
};

// =============================================================================
// Functor types for helper callbacks
// =============================================================================
//
// ── PostComputeFn ───────────────────────────────────────────────────────────
//
// Fires AFTER matmul accumulation, BEFORE tile_regs_commit.
// Tiles sit in DST[0..num_tiles-1] and can be modified in place.
//
// SAFE: SFPU tile ops, elementwise, init calls, CB reads (if tiles pre-produced)
// FORBIDDEN: tile_regs_acquire/commit/wait/release, pack_tile, matmul calls
//
// ── PostPackFn ──────────────────────────────────────────────────────────────
//
// Fires AFTER the pack loop, BEFORE tile_regs_release.
// Used for side effects while DST is held (e.g., hardware semaphore posting).
// Same FORBIDDEN rules as PostComputeFn.

struct NoPostPack {
    ALWI void operator()() const {}
};

struct NoPostCompute {
    ALWI void operator()(uint32_t /* num_tiles */) const {}
};

// =============================================================================
// Layer 0: Initialization
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_init(const MatmulConfig& cfg);

template <MatmulMode mode>
ALWI void matmul_init_short(const MatmulConfig& cfg);

template <MatmulMode mode>
ALWI void matmul_init_short_with_dt(const MatmulConfig& cfg, uint32_t old_in1_cb_id);

template <MatmulMode mode>
ALWI void matmul_init_short_with_both_dt(const MatmulConfig& cfg, uint32_t old_in0_cb_id, uint32_t old_in1_cb_id);

// =============================================================================
// detail:: — Internal building blocks (no DST management)
// =============================================================================

namespace detail {

template <MatmulMode mode>
ALWI void matmul_single(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx);

template <MatmulMode mode>
ALWI void matmul_accumulate(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t dst_start,
    uint32_t count,
    uint32_t in0_stride,
    uint32_t in1_stride,
    uint32_t dst_stride);

template <MatmulMode mode>
ALWI void matmul_accumulate_subblock(
    const MatmulConfig& cfg,
    uint32_t in0_subblock_offset,
    uint32_t in1_subblock_offset,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t inner_dim,
    uint32_t in1_stride);

#ifdef ARCH_BLACKHOLE
template <MatmulMode mode>
ALWI void matmul_accumulate_no_mop(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t dst_start,
    uint32_t count,
    uint32_t in0_stride,
    uint32_t in1_stride,
    uint32_t dst_stride);
#endif

ALWI void matmul_pack_to_cb(uint32_t dest_cb_id, uint32_t num_tiles);

ALWI void matmul_pack_to_partials(const MatmulConfig& cfg, uint32_t num_tiles);

template <MatmulMode mode>
ALWI void matmul_reload_partials(const MatmulConfig& cfg, uint32_t num_tiles);

template <MatmulMode mode>
ALWI void matmul_reduce_w(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx);

template <MatmulMode mode>
ALWI void matmul_reduce_w_with_init(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx);

}  // namespace detail

// =============================================================================
// Layer 4: Compound patterns
// =============================================================================

template <MatmulMode mode, typename PostComputeFn = NoPostCompute>
ALWI void matmul_accumulate_and_pack(
    const MatmulConfig& cfg,
    uint32_t in0_index_start,
    uint32_t in1_index_start,
    uint32_t inner_dim,
    uint32_t in1_stride,
    uint32_t dest_cb_id,
    uint32_t num_tiles,
    bool reload = false,
    PostComputeFn post_compute = {});

template <MatmulMode mode>
ALWI void matmul_compute_one_tile(const MatmulConfig& cfg, uint32_t inner_dim);

template <MatmulMode mode>
ALWI void matmul_compute_inner_block(
    const MatmulConfig& cfg,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in1_block_num_tiles,
    uint32_t in1_block_w,
    bool enable_reload,
    bool last_out);

// =============================================================================
// Layer 5: Specialized patterns (DST-managed)
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_reduce_w_and_pack(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx, uint32_t out_cb);

template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles);

// =============================================================================
// Layer 6: Full automated matmul
// =============================================================================

template <MatmulMode mode>
ALWI void matmul(const MatmulConfig& cfg, const MatmulBlockShape& shape);

// =============================================================================
// Layer 7: Single-tile matmul with DST+CB encapsulation
// =============================================================================

template <MatmulMode mode, typename PostComputeFn = NoPostCompute>
ALWI void matmul_single_and_pack(
    const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t out_cb, PostComputeFn post_compute = {});

// =============================================================================
// SDPA Helpers: Absolute-offset packing patterns
// =============================================================================

template <MatmulMode mode, bool blocked_pack = false, typename PostPackFn = NoPostPack>
ALWI void matmul_and_pack_absolute(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t inner_dim,
    uint32_t in1_stride,
    uint32_t out_num_cols,
    uint32_t row_offset,
    uint32_t col_offset,
    PostPackFn post_pack = {});

template <MatmulMode mode, typename PostComputeFn = NoPostCompute>
ALWI void matmul_blocks_absolute(
    const MatmulConfig& cfg,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    PostComputeFn post_compute = {});

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.inl"
