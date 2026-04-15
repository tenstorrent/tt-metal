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
 * @brief Unified matmul compute helpers for tile and block mode operations
 *
 * Replaces the MatmulOp<IsBlockMode> class with free functions following the
 * compute_kernel_lib conventions (namespace, .hpp/.inl split, config structs,
 * validation, DEST management).
 *
 * ## Architecture
 *
 * Functions are organized in layers of increasing abstraction:
 *
 * | Layer | Functions                                       | What it does                              |
 * |-------|-------------------------------------------------|-------------------------------------------|
 * | 0     | matmul_init, matmul_init_short, ..._with_dt     | Hardware init / reinit                    |
 * | 1     | matmul_single                                   | Single matmul_tiles or matmul_block call  |
 * | 2     | matmul_accumulate, ..._subblock, ..._no_mop     | Strided accumulation loops                |
 * | 3     | matmul_pack_to_cb, ..._to_partials, reload      | DEST commit + pack + CB management        |
 * | 4     | matmul_accumulate_and_pack, ..._inner_block     | Compound acquire+accumulate+pack patterns |
 * | 5     | matmul_reduce_w, ..._reduce_subblock_inplace    | Specialized op-specific patterns          |
 * | 6     | matmul                                          | Full blocked matmul with automation       |
 * | 7     | matmul_single_and_pack                          | Single-tile matmul with DST+CB encap.     |
 * | SDPA  | matmul_and_pack_absolute, matmul_blocks_absolute| Absolute-offset packing for SDPA          |
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
 *   // Example 1: Simple tile-mode matmul (replaces bmm.cpp TileMatmulOp::run)
 *   auto cfg = MatmulConfig::tile(cb_in0, cb_in1, cb_out);
 *   matmul_init<TILE>(cfg);
 *   matmul<TILE>(cfg, MatmulBlockShape::of(batch, Mt, Nt, Kt, 1, 1, 1, 1, 1));
 *
 *   // Example 2: Block-mode with spill/reload (replaces bmm_large_block_zm_fused)
 *   auto cfg = MatmulConfig::block(cb_in0, cb_in1, cb_partials,
 *                                   out_subblock_w, out_subblock_h, in0_block_w,
 *                                   transpose, cb_partials);
 *   matmul_init<BLOCK>(cfg);
 *   tile_regs_acquire();
 *   if (enable_reload) matmul_reload_partials<BLOCK>(cfg, num_tiles);
 *   matmul_accumulate<BLOCK>(cfg, in0_off, in1_off, 0, in0_block_w, 1, in1_block_w, 0);
 *   matmul_pack_to_cb(out_cb, num_tiles);
 *
 *   // Example 3: Conv pattern — accumulate and pack in one call
 *   auto cfg = MatmulConfig::block(in0, in1, out, ct, rt, kt);
 *   matmul_init_short<BLOCK>(cfg);
 *   matmul_accumulate_and_pack<BLOCK>(cfg, in0_off, in1_off, in0_block_w, stride, out_cb, tiles);
 *
 *   // Example 4: SDPA matmul with fused recip (PostComputeFn)
 *   auto cfg = MatmulConfig::block(sum_cb, identity_cb, scratch_cb, 1, 1, 1);
 *   matmul_init_short<BLOCK>(cfg);
 *   struct RecipFn { ALWI void operator()(uint32_t) const { recip_tile_init(); ... } };
 *   matmul_single_and_pack<BLOCK>(cfg, 0, 0, scratch_cb, RecipFn{});
 *
 *   // Example 5: SDPA blocked matmul with causal mask (PostComputeFn)
 *   auto cfg = MatmulConfig::block(q_cb, k_cb, qk_cb, sbw, sbh, block_w, true);
 *   struct MaskFn { uint32_t m, z; ALWI void operator()(uint32_t n) const { ... } };
 *   matmul_blocks_absolute<BLOCK>(cfg, M, N, K, nsub0, nsub1, sbh, sbw, MaskFn{m,z});
 *
 *   // Example 6: Reduce-W via matmul (replaces reduce_w.cpp)
 *   auto cfg = MatmulConfig::tile(cb_in0, cb_in1, cb_out);
 *   matmul_init<TILE>(cfg);
 *   tile_regs_acquire();
 *   matmul_reduce_w<TILE>(cfg, Wt, 0);
 *   // ... pack result ...
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
// MatmulConfig — replaces ckernel::MatmulOpConfig
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
    uint32_t in0_cb_id;  // CB for the A matrix (left operand)
    uint32_t in1_cb_id;  // CB for the B matrix (right operand)
    uint32_t out_cb_id;  // CB for output (or intermediate partials for spill)

    uint32_t ct_dim = 1;  // Output subblock column dimension in tiles (subblock_w)
    uint32_t rt_dim = 1;  // Output subblock row dimension in tiles (subblock_h)
    uint32_t kt_dim = 1;  // Inner dimension block size in tiles (in0_block_w)

    bool transpose = false;       // Transpose B tiles (width-height swap)
    uint32_t partials_cb_id = 0;  // CB for partial accumulations (0 = disabled)

    // --- Factory methods ---

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

/**
 * @brief Block shape specification for the full matmul() function
 *
 * Describes the complete iteration space for a blocked matrix multiply.
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
// Several helpers accept callback functors that fire at specific points in the
// DST lifecycle. These let callers fuse operations into the matmul pipeline
// without breaking DST or CB encapsulation.
//
// ── PostComputeFn ───────────────────────────────────────────────────────────
//
// Fires AFTER matmul accumulation, BEFORE tile_regs_commit. The matmul result
// tiles sit in DST[0..num_tiles-1] and can be modified in place.
//
//   tile_regs_acquire()               // helper owns this
//   matmul_accumulate(...)            // tiles now in DST
//   post_compute(num_tiles)           // <── YOUR CODE RUNS HERE
//   tile_regs_commit()                // helper owns this
//   ...pack...
//   tile_regs_release()               // helper owns this
//
// SAFE operations inside PostComputeFn:
//   - SFPU tile ops:  relu_tile(i), gelu_tile(i), recip_tile(i), exp_tile(i)
//   - Elementwise:    add_tiles(src_a_cb, src_b_cb, a_idx, b_idx, dst_idx)
//   - Init calls:     recip_tile_init(), add_tiles_init(), etc.
//   - CB reads:       cb_wait_front(some_cb, n) — ONLY if tiles are guaranteed
//                     to already be produced. If they are not yet produced,
//                     the kernel WILL hang with DST locked. The helper cannot
//                     prevent or detect this.
//
// FORBIDDEN inside PostComputeFn (will deadlock or corrupt DST state):
//   - tile_regs_acquire / tile_regs_commit / tile_regs_wait / tile_regs_release
//     The helper already holds the DST lock. Nested locking WILL deadlock.
//   - pack_tile / cb_push_back / cb_reserve_back on the matmul output CB.
//     The helper manages packing after PostComputeFn returns.
//   - matmul_single / matmul_accumulate or any matmul LLK call.
//     DST already contains the matmul result; a second matmul would corrupt it.
//
// PATTERN — define a struct, capture what you need, keep it simple:
//
//   struct MyPostCompute {
//       uint32_t bias_cb;
//       ALWI void operator()(uint32_t num_tiles) const {
//           cb_wait_front(bias_cb, num_tiles);      // bias must be pre-produced
//           add_tiles_init(bias_cb, bias_cb, true);
//           for (uint32_t i = 0; i < num_tiles; i++) {
//               add_tiles(bias_cb, bias_cb, 0, i, i);
//           }
//       }
//   };
//   matmul_blocks_absolute<BLOCK>(cfg, M, N, K, ..., MyPostCompute{bias_cb});
//
// ── PostPackFn ──────────────────────────────────────────────────────────────
//
// Fires AFTER the pack loop, BEFORE tile_regs_release. Used for side effects
// that must happen while DST is still held (e.g., hardware semaphore posting).
// Same FORBIDDEN rules as PostComputeFn: no DST locking, no packing, no matmul.
//
// ── Default functors ────────────────────────────────────────────────────────
//
// NoPostCompute and NoPostPack are no-ops that the compiler optimizes away
// entirely. Pass them (or omit the argument) when no fused operation is needed.

// Default no-op post-pack functor.
struct NoPostPack {
    ALWI void operator()() const {}
};

// Default no-op post-compute functor.
struct NoPostCompute {
    ALWI void operator()(uint32_t /* num_tiles */) const {}
};

// =============================================================================
// Layer 0: Initialization
// =============================================================================

/**
 * @brief Full hardware initialization for matmul
 *
 * Configures unpacker, math engine, and packer for matmul mode.
 * Must be called once before any matmul operations.
 *
 * @tparam mode TILE or BLOCK — selects mm_init vs mm_block_init
 * @param cfg Matmul configuration
 */
template <MatmulMode mode>
ALWI void matmul_init(const MatmulConfig& cfg);

/**
 * @brief Short reinit for matmul (unpacker + math only)
 *
 * Use when switching back to matmul from another operation (e.g., after
 * copy_tile, transpose, bias add). Cheaper than full init.
 */
template <MatmulMode mode>
ALWI void matmul_init_short(const MatmulConfig& cfg);

/**
 * @brief Short reinit with data format reconfiguration for srcA
 *
 * Reconfigures unpacker/math data format from old_in1_cb_id to the
 * config's in1_cb_id, then does a short init. Used after copy_tile
 * from a partials CB (which has a different data format than in1).
 */
template <MatmulMode mode>
ALWI void matmul_init_short_with_dt(const MatmulConfig& cfg, uint32_t old_in1_cb_id);

/**
 * @brief Short reinit with data format reconfiguration for both srcA and srcB
 *
 * Block mode only. Used after operations that change both input data formats
 * (e.g., transformer_group_attn_matmul after tilize).
 */
template <MatmulMode mode>
ALWI void matmul_init_short_with_both_dt(const MatmulConfig& cfg, uint32_t old_in0_cb_id, uint32_t old_in1_cb_id);

// =============================================================================
// detail:: — Internal building blocks (not for direct use in kernels)
// =============================================================================
//
// These functions do NOT manage DST (tile_regs_acquire/commit/wait/release).
// They are building blocks used internally by the DST-managed helpers below.
//
// Kernel code should use the DST-managed helpers (matmul_accumulate_and_pack,
// matmul_single_and_pack, matmul_and_pack_absolute, matmul_blocks_absolute,
// matmul_compute_one_tile, matmul_compute_inner_block, matmul, etc.) which
// encapsulate the full DST lifecycle and prevent acquire/commit/release bugs.
//
// If you are writing kernel code and find yourself calling detail:: functions
// with manual tile_regs_acquire/release, consider whether a DST-managed helper
// can handle your pattern instead. If not, use detail:: with extreme care.

namespace detail {

/// Single matmul LLK dispatch. DEST must be acquired. No DST/CB management.
template <MatmulMode mode>
ALWI void matmul_single(const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t dst_idx);

/// Strided accumulation loop. DEST must be acquired. No DST/CB management.
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

/// Tile-mode subblock accumulate. DEST must be acquired. No DST/CB management.
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
/// BH no-MOP accumulation (block mode only). DEST must be acquired.
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

/// Commit + reserve + wait + pack + release + push. DEST must be acquired.
ALWI void matmul_pack_to_cb(uint32_t dest_cb_id, uint32_t num_tiles);

/// Pack to partials CB. DEST must be acquired.
ALWI void matmul_pack_to_partials(const MatmulConfig& cfg, uint32_t num_tiles);

/// Reload partials into DEST. Must be called after acquire, before accumulation.
template <MatmulMode mode>
ALWI void matmul_reload_partials(const MatmulConfig& cfg, uint32_t num_tiles);

/// Reduce-W accumulation loop. DEST must be acquired.
template <MatmulMode mode>
ALWI void matmul_reduce_w(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx);

/// Reduce-W with per-tile init. DEST must be acquired.
template <MatmulMode mode>
ALWI void matmul_reduce_w_with_init(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx);

}  // namespace detail

// =============================================================================
// Layer 4: Compound patterns
// =============================================================================

/**
 * @brief Acquire + optional reload + accumulate + post_compute + pack to output
 *
 * Sequence: acquire → [reload_partials] → accumulate → post_compute(num_tiles) →
 *           commit → reserve(dest_cb) → wait → pack → release → push(dest_cb)
 *
 * Manages the full DST lifecycle and output CB reserve/pack/push.
 * Do NOT call tile_regs_acquire/release around this — the helper owns DST.
 *
 * @tparam mode           TILE or BLOCK matmul mode.
 * @tparam PostComputeFn  Functor called after accumulation, before commit.
 *                        See functor rules above. (default: no-op)
 *
 * @param cfg             Matmul configuration.
 * @param in0_index_start Starting in0 tile index (inputs must be waited by caller).
 * @param in1_index_start Starting in1 tile index (inputs must be waited by caller).
 * @param inner_dim       Number of accumulation iterations.
 * @param in1_stride      Stride between in1 tile indices per iteration.
 * @param dest_cb_id      Output CB to pack result into.
 * @param num_tiles       Number of output tiles (must fit in DST).
 * @param reload          If true, reload partials from cfg.partials_cb_id first.
 * @param post_compute    PostComputeFn instance (default: no-op).
 */
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

/**
 * @brief Compute one output tile with per-tile CB management
 *
 * For each of inner_dim iterations: wait(in0,1) + wait(in1,1) + matmul + pop both.
 * Then packs the single output tile.
 *
 * Used by moreh_matmul and tile-mode bmm.cpp for simple streaming patterns.
 */
template <MatmulMode mode>
ALWI void matmul_compute_one_tile(const MatmulConfig& cfg, uint32_t inner_dim);

/**
 * @brief Compute one inner block with subblock iteration and spill/reload
 *
 * Handles the double subblock loop (in0_num_subblocks x in1_num_subblocks),
 * CB wait/pop for the input block, and spill/reload between inner blocks.
 *
 * Block mode only.
 */
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

/**
 * @brief Reduce-W via matmul with full DST lifecycle and output CB management.
 *
 * Sequence: acquire → [per-tile: wait(in0,1) → matmul → pop(in0,1)] →
 *           commit → reserve(out_cb,1) → wait → pack → release → push(out_cb,1)
 *
 * Caller must wait on in1 (scaler/identity) before calling. The helper manages
 * in0 CB wait/pop per tile, DST lifecycle, and output CB reserve/pack/push.
 * Do NOT call tile_regs_acquire/release around this — the helper owns DST.
 */
template <MatmulMode mode>
ALWI void matmul_reduce_w_and_pack(const MatmulConfig& cfg, uint32_t count, uint32_t dst_idx, uint32_t out_cb);

/**
 * @brief SDPA reduce subblock inplace
 *
 * Per-subblock: acquire → matmul(0,0,0) → commit → pop(out,n) → pack(n) → release → push(n)
 *
 * Used by SDPA compute_common matmul_reduce pattern.
 */
template <MatmulMode mode>
ALWI void matmul_reduce_subblock_inplace(const MatmulConfig& cfg, uint32_t num_subblocks, uint32_t subblock_tiles);

// =============================================================================
// Layer 6: Full automated matmul
// =============================================================================

/**
 * @brief Complete blocked matmul with full automation
 *
 * In TILE mode: for each (batch, h, w): compute_one_tile(num_blocks_inner)
 * In BLOCK mode: for each (batch, bh, bw, bi): compute_inner_block(...)
 *
 * Handles spill/reload between inner blocks automatically.
 */
template <MatmulMode mode>
ALWI void matmul(const MatmulConfig& cfg, const MatmulBlockShape& shape);

// =============================================================================
// Layer 7: Single-tile matmul with DST+CB encapsulation
// =============================================================================

/**
 * @brief Single matmul operation with full DST lifecycle and output CB management.
 *
 * Sequence: reserve(out_cb,1) → acquire → matmul → post_compute(1) → commit →
 *           wait → pack(0, out_cb) → release → push(out_cb,1)
 *
 * Caller responsibilities:
 *   - Call matmul_init or matmul_init_short before first use
 *   - cb_wait_front on input CBs before calling this function
 *   - cb_pop_front on input CBs after this function returns
 *   - Do NOT call tile_regs_acquire/release around this — the helper owns DST
 *
 * @tparam mode           TILE or BLOCK matmul mode.
 * @tparam PostComputeFn  Functor called after matmul, before commit. Receives 1.
 *                        See functor rules above. Safe: SFPU ops, add_tiles.
 *                        Forbidden: DST locking, pack_tile, matmul calls.
 *
 * @param cfg             Matmul configuration (must match prior init call).
 * @param in0_idx         Tile index in in0 CB (input tiles must be waited by caller).
 * @param in1_idx         Tile index in in1 CB (input tiles must be waited by caller).
 * @param out_cb          Output CB — helper reserves 1 tile, packs, and pushes.
 * @param post_compute    PostComputeFn instance (default: no-op).
 */
template <MatmulMode mode, typename PostComputeFn = NoPostCompute>
ALWI void matmul_single_and_pack(
    const MatmulConfig& cfg, uint32_t in0_idx, uint32_t in1_idx, uint32_t out_cb, PostComputeFn post_compute = {});

// =============================================================================
// SDPA Helpers: Absolute-offset packing patterns
// =============================================================================

/**
 * @brief Matmul one subblock and pack at absolute offsets in output CB.
 *
 * Sequence: acquire → accumulate(inner_dim) → commit → wait →
 *           pack_tile<true> at row-major offsets → post_pack() → release
 *
 * Manages DST lifecycle and absolute-offset pack loop.
 * On Blackhole, uses matmul_block_no_mop automatically.
 *
 * Subblock dimensions are taken from cfg: cfg.rt_dim = subblock_h, cfg.ct_dim = subblock_w.
 * Output CB is cfg.out_cb_id.
 *
 * Caller responsibilities:
 *   - cb_reserve_back(cfg.out_cb_id, total_region) BEFORE the first call
 *   - cb_push_back(cfg.out_cb_id, ...) AFTER all subblocks are packed
 *   - cb_wait_front on input CBs before calling
 *   - cb_pop_front on input CBs after all subblocks for this block are done
 *   - matmul init (mm_block_init_short or similar) before first call and
 *     after any intervening operation that changes the unpack data format
 *   - Do NOT call tile_regs_acquire/release around this — the helper owns DST
 *
 * @tparam mode           TILE or BLOCK matmul mode.
 * @tparam blocked_pack   BH-only: if true, use blocked pack (one pack_tile per
 *                        row, advancing dst by subblock_w). Requires caller to
 *                        configure MOP via llk_pack_mop_config beforehand.
 * @tparam PostPackFn     Functor called after pack, before release. See functor
 *                        rules above. Forbidden: DST locking, pack_tile, matmul.
 *
 * @param cfg             Matmul configuration. rt_dim = subblock_h, ct_dim = subblock_w,
 *                        out_cb_id = output CB (must be pre-reserved by caller).
 * @param in0_start       Starting tile index in in0 CB.
 * @param in1_start       Starting tile index in in1 CB.
 * @param inner_dim       Number of accumulation iterations along inner dimension.
 * @param in1_stride      Stride between in1 tile indices per iteration.
 * @param out_num_cols    Total columns in the output region (for offset calc).
 * @param row_offset      Row offset for this subblock in the output region.
 * @param col_offset      Column offset for this subblock in the output region.
 * @param post_pack       PostPackFn instance (default: no-op).
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
 * This is the highest-encapsulation SDPA helper. It manages EVERYTHING:
 *   - matmul_init_short + reconfig_data_format at start
 *   - Full DST lifecycle per subblock (acquire/commit/wait/release)
 *   - Full CB lifecycle (in0 progressive wait, in1 wait+pop, out reserve+push)
 *   - Double subblock loop (M-subblocks x N-subblocks)
 *   - Absolute-offset packing to row-major positions in output CB
 *
 * CB protocol (managed internally):
 *   - in1: wait(K*N) upfront, pop(K*N) at end
 *   - in0: progressive wait per M-subblock, NOT popped (caller manages lifetime)
 *   - out: reserve(M*N) upfront, push(subblock_h*N) per M-subblock
 *
 * Caller responsibilities:
 *   - Ensure in0 and in1 tiles are produced before calling
 *   - cb_pop_front(in0_cb, ...) after this function returns (if needed)
 *   - Do NOT call tile_regs_acquire/release, cb_reserve_back(out_cb), or
 *     cb_push_back(out_cb) around this — the helper manages all of these
 *
 * @tparam mode           TILE or BLOCK matmul mode.
 * @tparam PostComputeFn  Functor called per subblock after accumulation,
 *                        before commit. Receives out_subblock_num_tiles.
 *                        See functor rules above. Safe: SFPU ops, add_tiles.
 *                        Forbidden: DST locking, pack_tile, matmul calls.
 *
 * @param cfg             Matmul configuration. cfg.kt_dim = inner block width
 *                        (number of K tiles per accumulation pass).
 * @param M               Total output rows in tiles.
 * @param N               Total output columns in tiles.
 * @param K               Full inner dimension in tiles (for in1 CB sizing).
 * @param in0_num_subblocks  Number of subblocks along M dimension.
 * @param in1_num_subblocks  Number of subblocks along N dimension.
 * @param post_compute    PostComputeFn instance (default: no-op).
 *
 * Subblock dimensions are taken from cfg: cfg.rt_dim = subblock_h, cfg.ct_dim = subblock_w.
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

// =============================================================================
// Bias Addition Helper
// =============================================================================

/**
 * @brief Row-broadcast bias addition on matmul output.
 *
 * Reads matmul output sub-blocks from partials_cb, adds bias with row broadcast,
 * optionally applies a post-bias operation (e.g., SFPU activation), and packs
 * the result to out_cb. Composes with matmul helpers by reading from the same
 * interm_cb that the matmul packed to.
 *
 * Manages full DST lifecycle and CB management per sub-block.
 * Do NOT call tile_regs_acquire/release around this — the helper owns DST.
 *
 * Caller responsibilities:
 *   - Call add_bcast_rows_init_short() before first use (or use reconfig_data_format)
 *   - Ensure partials and bias tiles are produced before calling
 *   - cb_pop_front(bias_cb, ...) after this function returns if needed
 *     (the helper does NOT pop bias_cb — caller manages bias tile lifetime)
 *
 * @tparam partials_cb    CB containing matmul output (= interm_cb from matmul_block).
 * @tparam bias_cb        CB containing bias tiles. One tile per output column, row-broadcast.
 * @tparam out_cb         Output CB for biased result.
 * @tparam PostBiasFn     Functor called per sub-block after bias addition, before commit.
 *                        See PostComputeFn rules. (default: NoPostCompute)
 *
 * @param in0_num_subblocks  Number of sub-blocks along M dimension.
 * @param in1_num_subblocks  Number of sub-blocks along N dimension.
 * @param out_subblock_h     Output sub-block height in tiles.
 * @param out_subblock_w     Output sub-block width in tiles.
 * @param bias_width_tiles   Number of bias tiles to wait for (= in1_num_subblocks * out_subblock_w).
 * @param post_bias          PostBiasFn instance (default: no-op).
 */
template <uint32_t partials_cb, uint32_t bias_cb, uint32_t out_cb, typename PostBiasFn = NoPostCompute>
ALWI void add_bias_bcast_rows(
    uint32_t in0_num_subblocks,
    uint32_t in1_num_subblocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t bias_width_tiles,
    PostBiasFn post_bias = {});

}  // namespace compute_kernel_lib

// Include implementation
#include "ttnn/cpp/ttnn/kernel_lib/matmul_helpers_compute.inl"
