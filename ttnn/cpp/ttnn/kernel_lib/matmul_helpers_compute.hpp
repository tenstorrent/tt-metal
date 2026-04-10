// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file matmul_helpers_compute.hpp
 * @brief Unified matmul helper library for compute kernels
 *
 * Provides free functions at three abstraction levels for matmul operations:
 *
 * **Low-Level (Init/Exec):**
 *   matmul_init, matmul_init_short, matmul_init_short_with_dt, matmul_tile
 *
 * **Mid-Level (Accumulate/Pack):**
 *   matmul_accumulate, matmul_accumulate_subblock, matmul_acquire_dst,
 *   matmul_pack_output, matmul_pack_partials, matmul_reload_partials,
 *   matmul_accumulate_and_pack
 *
 * **High-Level (Full Operations):**
 *   matmul - full blocked matmul with all loop/CB/DST management
 *
 * **Specialized Patterns:**
 *   matmul_reduce_w, matmul_attention, matmul_reduce_subblock_inplace,
 *   matmul_moe_with_bias, matmul_moe_w2_dm1_cycling, matmul_moe_w2_dm1_linear
 *
 * All functions are template-parameterized on MatmulMode (TILE or BLOCK).
 * TILE mode wraps matmul_tiles(); BLOCK mode wraps matmul_block() with
 * subblock dimensions (ct_dim, rt_dim, kt_dim).
 *
 * PREREQUISITE: Call matmul_init<mode>(cfg) before any other matmul function.
 *
 * ── Examples ────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.hpp"
 *   using namespace compute_kernel_lib;
 *
 *   // 1. Full-auto blocked matmul (simplest usage)
 *   auto cfg = MatmulConfig::tile(cb_in0, cb_in1, cb_out);
 *   matmul_init<MatmulMode::TILE>(cfg);
 *   matmul<MatmulMode::TILE>(cfg, MatmulBlockShape::of(batch, Mt, Nt, Kt,
 *       in0_num_sub, in1_num_sub, in0_block_tiles, in1_block_tiles, in1_block_w));
 *
 *   // 2. Block mode with explicit subblock control
 *   auto cfg = MatmulConfig::block(cb_in0, cb_in1, cb_out, ct, rt, kt);
 *   matmul_init<MatmulMode::BLOCK>(cfg);
 *   matmul_acquire_dst();
 *   matmul_accumulate<MatmulMode::BLOCK>(cfg, in0_start, in1_start, 0, kt, 1, in1_bw, 0);
 *   matmul_pack_output(cb_out, num_tiles);
 *
 *   // 3. Single tile matmul
 *   matmul_tile<MatmulMode::TILE>(cfg, in0_idx, in1_idx, dst_idx);
 *
 *   // 4. Reduce-W via matmul
 *   matmul_reduce_w(cfg, Wt, 0);
 *
 *   // 5. Attention pattern
 *   matmul_attention(cfg, inner_dim, progressive_in0);
 */

#include "api/compute/matmul.h"

namespace compute_kernel_lib {

// =============================================================================
// Enums
// =============================================================================

/**
 * @brief Compile-time selection of matmul execution mode
 *
 * - TILE: Uses matmul_tiles() LLK — single tile at a time, no subblock dims
 * - BLOCK: Uses matmul_block() LLK — subblock dimensions (ct_dim, rt_dim, kt_dim)
 */
enum class MatmulMode { TILE, BLOCK };

// =============================================================================
// Configuration Types
// =============================================================================

/**
 * @brief Configuration for matmul operations
 *
 * Stores circular buffer IDs, subblock dimensions, and optional settings.
 * Use factory methods for common configurations:
 *   MatmulConfig::tile(in0, in1, out)
 *   MatmulConfig::block(in0, in1, out, ct, rt, kt)
 *
 * Or designated initializers for full control:
 *   MatmulConfig{.in0_cb_id=cb0, .in1_cb_id=cb1, .out_cb_id=cb2, .ct_dim=4, ...}
 */
struct MatmulConfig {
    uint32_t in0_cb_id;           // CB for A matrix (left operand)
    uint32_t in1_cb_id;           // CB for B matrix (right operand)
    uint32_t out_cb_id;           // CB for output
    uint32_t ct_dim = 1;          // Output subblock width in tiles (subblock_w)
    uint32_t rt_dim = 1;          // Output subblock height in tiles (subblock_h)
    uint32_t kt_dim = 1;          // Inner dimension block size in tiles (in0_block_w)
    bool transpose = false;       // Transpose B tiles
    uint32_t partials_cb_id = 0;  // CB for spill/reload partials (0 = disabled)

    static constexpr MatmulConfig tile(uint32_t in0, uint32_t in1, uint32_t out) {
        return {in0, in1, out, 1, 1, 1, false, 0};
    }

    static constexpr MatmulConfig block(
        uint32_t in0, uint32_t in1, uint32_t out, uint32_t ct, uint32_t rt, uint32_t kt) {
        return {in0, in1, out, ct, rt, kt, false, 0};
    }

    constexpr MatmulConfig with_transpose(bool t) const {
        return {in0_cb_id, in1_cb_id, out_cb_id, ct_dim, rt_dim, kt_dim, t, partials_cb_id};
    }

    constexpr MatmulConfig with_partials(uint32_t cb) const {
        return {in0_cb_id, in1_cb_id, out_cb_id, ct_dim, rt_dim, kt_dim, transpose, cb};
    }
};

/**
 * @brief Loop dimensions for full-auto matmul (matmul<mode>(cfg, shape))
 *
 * Encapsulates the full set of blocking parameters. Use ::of() factory.
 */
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
        uint32_t batch,
        uint32_t nbh,
        uint32_t nbw,
        uint32_t nbi,
        uint32_t in0_nsub,
        uint32_t in1_nsub,
        uint32_t in0_bnt,
        uint32_t in1_bnt,
        uint32_t in1_bw) {
        return {batch, nbh, nbw, nbi, in0_nsub, in1_nsub, in0_bnt, in1_bnt, in1_bw};
    }
};

/**
 * @brief MoE DM1 buffer tracking state
 *
 * Tracks progress through DM1 buffer tables for MoE W2 kernels.
 */
struct MoeDm1State {
    uint32_t step;
    uint32_t tiles_remaining;
    uint32_t buf;     // current buffer index (for cycling)
    uint32_t offset;  // tile offset for current buffer
    uint32_t index;   // current in2 tile index
};

// =============================================================================
// Low-Level: Initialization
// =============================================================================

/**
 * @brief Full matmul initialization with packer config
 *
 * Must be called once before any matmul operation. Configures unpacker, math,
 * and packer pipelines for the given mode and CB configuration.
 */
template <MatmulMode mode>
ALWI void matmul_init(const MatmulConfig& cfg);

/**
 * @brief Quick matmul re-initialization (unpacker/math only)
 *
 * Use after switching away from matmul to another operation (e.g., copy_tile,
 * reduce) and needing to restore matmul configuration without full packer setup.
 */
template <MatmulMode mode>
ALWI void matmul_init_short(const MatmulConfig& cfg);

/**
 * @brief Re-init with SRCA data format reconfiguration
 *
 * Use when the previous operation used a different CB for SRCA (e.g., after
 * copy_tile_to_dst from a partials CB). Reconfigures data format from
 * old_in1_cb_id back to cfg.in1_cb_id.
 */
template <MatmulMode mode>
ALWI void matmul_init_short_with_dt(const MatmulConfig& cfg, uint32_t old_in1_cb_id);

/**
 * @brief Re-init with both SRCA and SRCB data format reconfiguration (block mode only)
 */
ALWI void matmul_init_short_with_both_dt(const MatmulConfig& cfg, uint32_t old_in0_cb_id, uint32_t old_in1_cb_id);

// =============================================================================
// Low-Level: Single Tile Execution
// =============================================================================

/**
 * @brief Execute a single matmul tile operation
 *
 * Dispatches to matmul_tiles() (TILE mode) or matmul_block() (BLOCK mode).
 * Caller is responsible for DST register management and CB synchronization.
 */
template <MatmulMode mode>
ALWI void matmul_tile(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx);

#ifdef ARCH_BLACKHOLE
/**
 * @brief No-MOP single matmul tile (Blackhole, block mode only)
 *
 * Uses matmul_block_no_mop for better performance on Blackhole SDPA kernels.
 */
ALWI void matmul_tile_no_mop(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx);
#endif

// =============================================================================
// Mid-Level: Accumulation Patterns
// =============================================================================

/**
 * @brief Strided matmul accumulation loop
 *
 * Iterates count times, advancing each index by its stride:
 *   for k in [0, count): matmul(in0_start+k*in0_stride, in1_start+k*in1_stride, dst_start+k*dst_stride)
 *
 * Common stride patterns:
 *   Inner-dim reduction:  (a, b, dst, K, 1, stride, 0) — dst fixed
 *   Broadcast-in0:        (0, b, 0, N, 0, 1, 1)       — in0 fixed
 */
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

/**
 * @brief 2D subblock matmul accumulation
 *
 * Computes an (out_h x out_w) subblock of output tiles. Each output tile at
 * (h, w) accumulates over inner_dim with tile indexing:
 *   in0 = in0_offset + h * inner_dim + k
 *   in1 = in1_offset + k * in1_stride + w
 *   dst = sequential (0, 1, 2, ...)
 */
template <MatmulMode mode>
ALWI void matmul_accumulate_subblock(
    const MatmulConfig& cfg,
    uint32_t in0_offset,
    uint32_t in1_offset,
    uint32_t out_h,
    uint32_t out_w,
    uint32_t inner_dim,
    uint32_t in1_stride);

#ifdef ARCH_BLACKHOLE
/**
 * @brief No-MOP strided accumulation (Blackhole, block mode only)
 */
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

// =============================================================================
// Mid-Level: DST Register and Pack Management
// =============================================================================

/**
 * @brief Acquire DST registers for a new subblock computation
 *
 * Wraps tile_regs_acquire(). Must be paired with matmul_pack_output() or
 * matmul_pack_partials() which release the registers after packing.
 */
ALWI void matmul_acquire_dst();

/**
 * @brief Commit, pack, and push results to an output CB
 *
 * Sequence: tile_regs_commit → cb_reserve_back → tile_regs_wait →
 *           pack_tile(0..n-1) → tile_regs_release → cb_push_back
 */
ALWI void matmul_pack_output(uint32_t dest_cb_id, uint32_t num_tiles);

/**
 * @brief Commit, pack, and push results to the partials CB
 *
 * Same as matmul_pack_output but targets cfg.partials_cb_id.
 */
ALWI void matmul_pack_partials(const MatmulConfig& cfg, uint32_t num_tiles);

/**
 * @brief Reload partial accumulations from the partials CB into DST
 *
 * Loads previously spilled partials back into DST registers, then
 * reconfigures data format back to matmul mode (the copy_tile operation
 * changes the SRCA format).
 */
template <MatmulMode mode>
ALWI void matmul_reload_partials(const MatmulConfig& cfg, uint32_t num_tiles);

/**
 * @brief Combined: acquire + optional reload + accumulate + pack_output
 *
 * Replaces the common pattern:
 *   begin_subblock(); if(reload) reload_partials(n); accumulate(...); end_to_output(cb, n);
 */
template <MatmulMode mode>
ALWI void matmul_accumulate_and_pack(
    const MatmulConfig& cfg,
    uint32_t in0_start,
    uint32_t in1_start,
    uint32_t inner_dim,
    uint32_t in1_stride,
    uint32_t dest_cb_id,
    uint32_t num_tiles,
    bool reload = false);

// =============================================================================
// High-Level: Full Operations
// =============================================================================

/**
 * @brief Full blocked matmul with all loop, CB, and DST management
 *
 * Handles the complete batch × h_blocks × w_blocks × inner_blocks loop nest,
 * including spill/reload when num_blocks_inner > 1 and partials_cb_id != 0.
 *
 * TILE mode: per-tile CB wait/pop for each K iteration
 * BLOCK mode: per-block CB wait/pop with subblock loops and optional spill/reload
 */
template <MatmulMode mode>
ALWI void matmul(const MatmulConfig& cfg, const MatmulBlockShape& shape);

/**
 * @brief Compute one inner block across subblocks with optional spill/reload (block mode)
 *
 * Handles the double subblock loop (in0_num_subblocks × in1_num_subblocks),
 * CB wait/pop for the input block, and routing output to either out_cb (last_out)
 * or partials_cb (spill).
 */
ALWI void matmul_inner_block(
    const MatmulConfig& cfg,
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in1_block_num_tiles,
    uint32_t in1_block_w,
    bool enable_reload,
    bool last_out);

/**
 * @brief Compute one output tile by accumulating over inner_dim (tile mode)
 *
 * Per-tile CB wait/pop on both inputs. Handles DST acquire/commit/pack.
 * Sequence: acquire → for(K){wait/matmul/pop} → commit → reserve → pack → release → push
 */
ALWI void matmul_compute_tile(const MatmulConfig& cfg, uint32_t inner_dim);

// =============================================================================
// Specialized: Reduce-W via Matmul
// =============================================================================

/**
 * @brief Width reduction via matmul: per-tile CB wait/pop with matmul accumulation
 *
 * Absorbs: for(w){cb_wait(in0,1); matmul(0,0,dst); cb_pop(in0,1);}
 *
 * @tparam reinit_per_tile If true, calls matmul_init_short() per tile (for kernels
 *         that reconfigure data formats between iterations, e.g., moreh_mean_w).
 */
template <bool reinit_per_tile = false>
ALWI void matmul_reduce_w(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx);

// =============================================================================
// Specialized: Attention Patterns
// =============================================================================

/**
 * @brief Transformer attention accumulate: progressive in0 reveal + per-tile in1
 *
 * Each iteration: [optionally wait(in0, kt+1)] → wait(in1, 1) → matmul(kt, 0, 0) → pop(in1, 1)
 *
 * @param progressive_in0 If true, progressively waits for in0 tiles (first call pattern)
 */
ALWI void matmul_attention(const MatmulConfig& cfg, uint32_t inner_dim, bool progressive_in0);

/**
 * @brief SDPA reduce subblock inplace
 *
 * Per-subblock: acquire → matmul(0,0,0) → commit → pop(out,n) → pack(n) → release → push(n)
 */
template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles);

// =============================================================================
// Specialized: MoE Patterns
// =============================================================================

/**
 * @brief MoE blocked accumulate with bias at limit
 *
 * Iterates over weight blocks, accumulating with stride. When k_tracker reaches
 * limit, applies bias via a separate matmul config. Returns updated in0_index.
 */
template <MatmulMode mode>
ALWI uint32_t matmul_moe_with_bias(
    const MatmulConfig& cfg,
    const MatmulConfig& bias_cfg,
    uint32_t in0_start,
    uint32_t num_blocks,
    uint32_t tiles_per_block,
    uint32_t tile_stride,
    uint32_t limit);

/**
 * @brief MoE W2 with DM1 buffer cycling and bias
 *
 * Weight blocks with DM1 buffer tracking, N-buffer cycling, and bias at limit.
 */
template <MatmulMode mode>
ALWI void matmul_moe_w2_dm1_cycling(
    const MatmulConfig& cfg,
    const MatmulConfig& bias_cfg,
    MoeDm1State& dm1,
    uint32_t num_blocks,
    uint32_t tiles_per_block,
    uint32_t tile_stride,
    uint32_t limit,
    uint32_t dm1_rdy_cb,
    uint32_t tiles_per_step,
    uint32_t num_buffers,
    const uint32_t* dm1_table);

/**
 * @brief MoE W2 with DM1 linear advance (no bias)
 *
 * Weight blocks with DM1 buffer tracking, linear offset advance,
 * and early exit on last block.
 */
template <MatmulMode mode>
ALWI void matmul_moe_w2_dm1_linear(
    const MatmulConfig& cfg,
    MoeDm1State& dm1,
    uint32_t num_blocks,
    uint32_t tiles_per_block,
    uint32_t tile_stride,
    uint32_t dm1_rdy_cb,
    uint32_t tiles_per_step,
    const uint32_t* dm1_table,
    uint32_t last_block_early_exit_k);

}  // namespace compute_kernel_lib

// Include implementation
#include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.inl"
