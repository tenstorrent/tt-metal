// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

/**
 * @file matmul_helpers_compute.hpp
 * @brief Runtime-config matmul compute helpers for SDPA and dynamic patterns
 *
 * Complements the compile-time matmul_block_helpers.hpp with runtime-configurable
 * matmul operations using MatmulConfig and MatmulMode.
 *
 * Use matmul_block_helpers.hpp for standard blocked matmul kernels (compile-time
 * CB params, full K-loop automation). Use this file for:
 *   - SDPA kernels that need absolute-offset packing
 *   - Kernels that need runtime CB selection
 *   - Building blocks for custom matmul patterns (detail:: namespace)
 *
 * ## MatmulMode
 *
 * Template parameter controlling the underlying LLK function:
 * - TILE: calls matmul_tiles (per-tile, 5 args)
 * - BLOCK: calls matmul_block (per-block, 9 args)
 */

namespace compute_kernel_lib {

// =============================================================================
// MatmulMode — selects tile vs block LLK functions
// =============================================================================

enum class MatmulMode { TILE, BLOCK };

inline constexpr MatmulMode TILE = MatmulMode::TILE;
inline constexpr MatmulMode BLOCK = MatmulMode::BLOCK;

// =============================================================================
// MatmulConfig — runtime configuration for matmul operations
// =============================================================================

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
// Functor defaults
// =============================================================================

struct NoPostPack {
    ALWI void operator()() const {}
};

struct NoPostCompute {
    ALWI void operator()(uint32_t /* num_tiles */) const {}
};

// =============================================================================
// Initialization
// =============================================================================

template <MatmulMode mode>
ALWI void matmul_init_short(const MatmulConfig& cfg);

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

}  // namespace detail

// =============================================================================
// DST-managed helpers
// =============================================================================

/**
 * @brief Single matmul + PostComputeFn + pack to output CB.
 *
 * Full lifecycle: init_short + reconfig + wait(in0,1) + wait(in1,1) +
 * reserve(out_cb,1) → acquire → matmul → post_compute(1) → commit →
 * wait → pack(0, out_cb) → release → push(out_cb,1) → pop(in0,1)
 *
 * Caller must: ensure in1 tiles are produced (helper waits but does NOT pop in1).
 * The helper pops 1 tile from in0 after packing.
 */
template <MatmulMode mode, typename PostComputeFn = NoPostCompute>
ALWI void matmul_single_and_pack(
    const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t out_cb, PostComputeFn post_compute = {});

/**
 * @brief SDPA reduce subblock inplace.
 *
 * Full lifecycle: init_short + reconfig + input CB waits +
 * per-subblock (acquire → matmul(0,0,0) → commit → pop(out,n) →
 *               wait → pack(n) → release → push(n))
 *
 * Caller must: ensure in1 (identity) tiles are produced, ensure out_cb has total_in0_tiles.
 * The helper waits on both, pops/pushes out_cb per subblock, does NOT pop in1.
 */
template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(
    const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles, uint32_t total_in0_tiles);

// =============================================================================
// SDPA Helpers: Absolute-offset packing patterns
// =============================================================================

/**
 * @brief Matmul one subblock and pack at absolute offsets in output CB.
 *
 * Sequence: acquire → accumulate(inner_dim) → commit → wait →
 *           pack_tile<true> at row-major offsets → post_pack() → release
 *
 * On Blackhole, uses matmul_block_no_mop automatically.
 * Caller must: reserve output CB upfront, push after all subblocks, wait on inputs.
 */
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

/**
 * @brief Full blocked matmul with absolute-offset packing and CB management.
 *
 * Manages: init_short, reconfig, DST lifecycle, CB lifecycle, double subblock loop.
 * Caller must: produce in0/in1 tiles, pop in0 after return.
 */
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
