// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"

/**
 * @file binary_op_helpers.hpp
 * @brief Unified binary operation helper for compute kernels
 *
 * Provides a single binary_op() function (and add/sub/mul/square aliases) that handles
 * element-wise binary operations with automatic broadcast, DEST register management,
 * circular buffer synchronization, and data format reconfiguration.
 *
 * PREREQUISITE: Call binary_op_init_common(icb_a, icb_b, ocb) at the start of your
 * kernel before using these functions.
 *
 * ## Broadcast Dimension Reference
 *
 * BroadcastDim specifies the SHAPE of operand B and how it broadcasts to match A:
 *
 * | BroadcastDim | B Shape   | B Tiles | What Gets Broadcast                    |
 * |--------------|-----------|---------|----------------------------------------|
 * | NONE         | [Ht, Wt]  | Ht * Wt | All elements used as-is                |
 * | ROW          | [1, Wt]   | Wt      | Single row replicated down (rows)      |
 * | COL          | [Ht, 1]   | Ht      | Single column replicated right (cols)  |
 * | SCALAR       | [1, 1]    | 1       | Single tile replicated everywhere      |
 *
 * ## Relationship to Reduce Operations
 *
 * After reduction, use the corresponding broadcast to apply the result:
 *
 * | Reduce Operation | Output Shape | Use Broadcast | Example                     |
 * |------------------|--------------|---------------|-----------------------------|
 * | REDUCE_ROW       | [Ht, 1]      | COL           | Subtract row-wise mean      |
 * | REDUCE_COL       | [1, Wt]      | ROW           | Subtract column-wise mean   |
 * | REDUCE_SCALAR    | [1, 1]       | SCALAR        | Subtract global mean        |
 *
 * Note: REDUCE_ROW produces COL-shaped output (this is correct but counterintuitive).
 * "REDUCE_ROW" means "reduce along row direction" = sum across width = column output.
 *
 * ── Examples ────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
 *   using namespace compute_kernel_lib;
 *
 *   binary_op_init_common(cb_a, cb_b, cb_out);
 *
 *   // 1. Basic element-wise add (most common — streaming, one tile at a time)
 *   add(cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 2. Subtract with broadcast — B is a column vector [Ht, 1]
 *   //    (e.g., subtract per-row mean after REDUCE_ROW)
 *   sub<BroadcastDim::COL>(cb_in, cb_mean, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 3. Multiply with scalar broadcast — B is a single tile [1, 1]
 *   mul<BroadcastDim::SCALAR>(cb_in, cb_scale, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 4. Square (self-multiply) — single input CB
 *   square(cb_in, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 5. WaitUpfrontNoPop — tiles persist in CB for reuse by later operations
 *   //    (ideal for softmax: subtract max, then reuse input for exp)
 *   sub<BroadcastDim::COL,
 *       BinaryInputPolicy::WaitUpfrontNoPop>(
 *       cb_in, cb_max, cb_centered, BinaryInputBlockShape::of(Ht, Wt));
 *   // cb_in tiles still available for next operation
 *
 *   // 6. NoWaitNoPop — caller manages wait/pop externally (sharded or pre-waited)
 *   cb_wait_front(cb_a, total_tiles);
 *   cb_wait_front(cb_b, total_tiles);
 *   add<BroadcastDim::NONE,
 *       BinaryInputPolicy::NoWaitNoPop,
 *       BinaryInputPolicy::NoWaitNoPop>(
 *       cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *   cb_pop_front(cb_a, total_tiles);
 *   cb_pop_front(cb_b, total_tiles);
 *
 *   // 7. Independent A/B policies — A streams, B persists (reused across rows)
 *   mul<BroadcastDim::ROW,
 *       BinaryInputPolicy::WaitAndPopPerTile,
 *       BinaryInputPolicy::WaitUpfrontNoPop>(
 *       cb_in, cb_weights, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 8. Skip data format reconfiguration (first op or formats already match)
 *   add<BroadcastDim::NONE,
 *       BinaryInputPolicy::WaitAndPopPerTile,
 *       BinaryInputPolicy::WaitAndPopPerTile,
 *       BinaryOutputPolicy::PerTile,
 *       BinaryDataFormatReconfig::NONE>(
 *       cb_a, cb_b, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // 9. Post-operation chain — apply rsqrt after multiply
 *   mul(cb_in, cb_in, cb_out, BinaryInputBlockShape::of(Ht, Wt),
 *       sfpu_chain(Rsqrt<Legacy::Off, Approx::Exact, Dst::D0>{}));
 *   // PostOp must be either NoOp (default) or an sfpu_chain(...). Single ops
 *   // must be wrapped in a chain; raw lambdas are not accepted.
 *
 *   // 10. Low-level binary_op with explicit op type
 *   binary_op<BinaryOpType::ADD, BroadcastDim::SCALAR>(
 *       cb_in, cb_bias, cb_out, BinaryInputBlockShape::of(Ht, Wt));
 */

namespace compute_kernel_lib {

// =============================================================================
// Enums
// =============================================================================

enum class BinaryOpType { ADD, SUB, MUL, SQUARE };
enum class BroadcastDim { NONE, ROW, COL, SCALAR };

/**
 * @brief Data format reconfiguration mode for binary operations
 *
 * Controls whether unpacker (srca/srcb) and/or packer (output) are reconfigured.
 * All reconfigurations are unconditional (no old-operand compare); if the caller
 * needs the "skip if already matching" optimization, set NONE and invoke the raw
 * `reconfig_data_format[_srca|_srcb](old, new)` LLK externally before this helper.
 *
 * - NONE: Skip all reconfiguration (first op in chain or formats already match)
 * - INPUT: Reconfigure both srca and srcb (`reconfig_data_format(icb_a, icb_b)`)
 * - OUTPUT: Reconfigure packer only (`pack_reconfig_data_format(ocb)`)
 * - INPUT_AND_OUTPUT: Reconfigure srca+srcb and packer (default, safest)
 * - SRCA_ONLY: Reconfigure srca only (`reconfig_data_format_srca(icb_a)`)
 * - SRCB_ONLY: Reconfigure srcb only (`reconfig_data_format_srcb(icb_b)`)
 * - SRCA_ONLY_AND_OUTPUT: srca + packer
 * - SRCB_ONLY_AND_OUTPUT: srcb + packer
 */
enum class BinaryDataFormatReconfig {
    NONE = 0,
    INPUT = 1,
    OUTPUT = 2,
    INPUT_AND_OUTPUT = 3,
    SRCA_ONLY = 4,
    SRCB_ONLY = 5,
    SRCA_ONLY_AND_OUTPUT = 6,
    SRCB_ONLY_AND_OUTPUT = 7,
};

/**
 * @brief Input synchronization and consumption policy for binary operations
 *
 * Controls when to wait for input tiles and whether to pop them after processing.
 * Can be set independently for input A and input B.
 *
 * - WaitAndPopPerTile: Wait/process/pop one tile at a time (streaming, safe for any CB size)
 * - WaitAndPopPerChunk: Wait for chunk (DEST_LIMIT tiles), process all, pop chunk
 * - WaitUpfrontNoPop: Wait for all tiles upfront, don't pop (persistent, for tile reuse)
 * - WaitUpfrontPopAtEnd: Wait for all tiles upfront, pop all at end (consume after processing)
 * - NoWaitNoPop: Caller manages wait/pop externally (preloaded, tiles already in CB)
 * - NoWaitPopAtEnd: Caller manages wait, pop all at end (preloaded, consume after processing)
 *
 * WARNING - NoWait Policies (NoWaitNoPop, NoWaitPopAtEnd):
 * These policies can cause data hazards if used incorrectly. ONLY use when:
 *   1. Paired with explicit cb_wait_front() before the operation, OR
 *   2. As the FIRST operation in a chain, OR
 *   3. With sharded tensors where data is pre-loaded in CB
 * When in doubt, use WaitAndPopPerTile or WaitUpfrontNoPop.
 */
enum class BinaryInputPolicy {
    WaitAndPopPerTile,    // Wait/process/pop one tile at a time (streaming)
    WaitAndPopPerChunk,   // Wait for chunk, process all, pop chunk
    WaitUpfrontNoPop,     // Wait for all tiles upfront, don't pop (persistent)
    WaitUpfrontPopAtEnd,  // Wait for all tiles upfront, pop at end (consume)
    NoWaitNoPop,          // Caller manages wait/pop (preloaded)
    NoWaitPopAtEnd,       // Caller manages wait, pop at end (preloaded, consume)
    CumulativeWaitNoPop   // GAP-2: helper's internal chunk loop does cumulative waits
                          // (cb_wait_front grows by extras.wait_step per chunk). No pop —
                          // caller pops externally after all consumers. shape.cols = full
                          // block size; wait_step = per-chunk granularity (must be <=
                          // DEST_AUTO_LIMIT). Absorbs the hand-rolled outer loop.
};

/**
 * @brief Output policy for binary operations
 *
 * Controls when to reserve and push output tiles:
 * - PerTile: Reserve/push one tile at a time (streaming)
 * - PerChunk: Reserve/push chunk of tiles at a time (DEST_LIMIT tiles)
 * - Bulk: Reserve all upfront, push all at end
 */
enum class BinaryOutputPolicy {
    PerTile,   // Reserve/push one tile at a time
    PerChunk,  // Reserve/push chunk of tiles at a time
    Bulk       // Reserve all upfront, push all at end
};

// =============================================================================
// Data Types
// =============================================================================

/**
 * @brief Input tile grid dimensions (rows x cols in tiles)
 *
 * Use factory methods: ::of(r, c), ::single(), ::row(c), ::col(r)
 */
struct BinaryInputBlockShape {
    uint32_t rows;
    uint32_t cols;

    static constexpr BinaryInputBlockShape of(uint32_t r, uint32_t c) { return {r, c}; }
    static constexpr BinaryInputBlockShape single() { return {1, 1}; }
    static constexpr BinaryInputBlockShape row(uint32_t c) { return {1, c}; }
    static constexpr BinaryInputBlockShape col(uint32_t r) { return {r, 1}; }
};

/**
 * @brief Per-operand runtime extras: tile-index base + cumulative wait step.
 *
 * `base` — added to the per-iter tile index for this operand (enables absolute
 * indexing like `mul_tiles(cb, cb, wt+wtr, wt+wtr, wtr)` without a hand-written
 * tile loop). Default 0 (sequential from CB front).
 *
 * `wait_step` — only used under BinaryInputPolicy::CumulativeWaitNoPop. The
 * helper's internal chunk loop processes tiles in `wait_step`-sized groups;
 * at the start of each group it issues `cb_wait_front(cb, group_end)` where
 * group_end grows cumulatively. This matches the layernorm pattern
 * `cb_wait_front(cb_inp, wt + blk)` without a hand-rolled outer loop — caller
 * passes shape.cols = full block size and wait_step = blk, and the helper
 * absorbs the outer loop (one init/reconfig per binary_op call). Must be
 * <= DEST_AUTO_LIMIT. Ignored for any non-cumulative policy.
 */
struct BinaryInputExtras {
    uint32_t base = 0;
    uint32_t wait_step = 0;
};

struct BinaryAccumulate {
    uint32_t cb_accumulator, dst_index = 0;
};

// =============================================================================
// Low-Level LLK Wrappers
// =============================================================================

/**
 * @brief Initialize binary operation hardware for given op type and broadcast mode
 *
 * Configures math and unpack pipelines. Called automatically by binary_op() when
 * init=true (default). Call manually only when init=false and you need explicit
 * control over initialization timing.
 */
template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_init(uint32_t icb_a, uint32_t icb_b);

/**
 * @brief Execute a single binary tile operation
 *
 * Unpacks tiles from input CBs and performs the binary math operation into DEST.
 * Used internally by binary_op(). Call directly only when building custom tile loops.
 *
 * @param icb_a Input CB for operand A
 * @param icb_b Input CB for operand B
 * @param itile_a Tile index within icb_a
 * @param itile_b Tile index within icb_b
 * @param idst DEST register index for the result
 */
template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_exec(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst);

// =============================================================================
// Main API
// =============================================================================

/**
 * @brief Unified binary operation with automatic broadcast, DEST management, and CB sync
 *
 * Performs A op B for all tiles in the input block shape. Handles:
 * - DEST register acquire/commit/wait/release (chunked by DEST_AUTO_LIMIT)
 * - Circular buffer wait/pop/reserve/push per configured policies
 * - Broadcast tile indexing (ROW, COL, SCALAR, or NONE)
 * - Data format reconfiguration
 * - Optional post-op callback on each output tile in DEST
 *
 * @tparam op_type       Binary operation: ADD, SUB, MUL, or SQUARE
 * @tparam bcast_dim     Broadcast mode for operand B (default: NONE)
 * @tparam input_a_policy  Input A synchronization policy (default: WaitAndPopPerTile)
 * @tparam input_b_policy  Input B synchronization policy (default: same as input_a_policy)
 * @tparam output_policy   Output synchronization policy (default: PerTile)
 * @tparam reconfig      Data format reconfiguration mode (default: INPUT_AND_OUTPUT)
 * @tparam init          Whether to call binary_init (default: true)
 *
 * @param icb_a  Input circular buffer for operand A
 * @param icb_b  Input circular buffer for operand B (same as icb_a for SQUARE)
 * @param ocb    Output circular buffer
 * @param shape  Tile grid dimensions — use BinaryInputBlockShape::of(Ht, Wt)
 * @param post_op  Post-operation callback receiving dst_idx (default: NoOp)
 * @param accum  Accumulation config (default: NoAccumulation)
 */
template <
    BinaryOpType op_type,
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_a_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy input_b_policy = input_a_policy,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp,
    typename AccumT = NoAccumulation>
ALWI void binary_op(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op = {},
    AccumT accum = {},
    BinaryInputExtras a_extras = {},
    BinaryInputExtras b_extras = {});

// =============================================================================
// Convenience Aliases
// =============================================================================

/** @brief Element-wise addition: A + B. See binary_op() for full documentation. */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_a_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy input_b_policy = input_a_policy,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp,
    typename AccumT = NoAccumulation>
ALWI void add(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op = {},
    AccumT accum = {},
    BinaryInputExtras a_extras = {},
    BinaryInputExtras b_extras = {});

/** @brief Element-wise subtraction: A - B. See binary_op() for full documentation. */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_a_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy input_b_policy = input_a_policy,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp,
    typename AccumT = NoAccumulation>
ALWI void sub(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op = {},
    AccumT accum = {},
    BinaryInputExtras a_extras = {},
    BinaryInputExtras b_extras = {});

/** @brief Element-wise multiplication: A * B. See binary_op() for full documentation. */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_a_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy input_b_policy = input_a_policy,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp,
    typename AccumT = NoAccumulation>
ALWI void mul(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op = {},
    AccumT accum = {},
    BinaryInputExtras a_extras = {},
    BinaryInputExtras b_extras = {});

/** @brief Element-wise square: A * A. Uses a single input CB. See binary_op() for full documentation. */
template <
    BinaryInputPolicy input_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp,
    typename AccumT = NoAccumulation>
ALWI void square(
    uint32_t icb,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op = {},
    AccumT accum = {},
    BinaryInputExtras extras = {});

// =============================================================================
// Dest-Reuse PostOp
// =============================================================================

/**
 * @brief Data-format reconfig mode for DestReuseOp.
 *
 * DestReuseOp reads DEST (which needs no data-format reconfig) and CB (which
 * may need one if CB's format differs from binary_op's icb_a/icb_b format).
 * This enum toggles the CB-side reconfig only.
 *
 * - None:  no reconfig; CB's data format must already match binary_op's setup
 *          (common case: cb_den / cb_scale has the same format as icb_b).
 * - Input: reconfig the SRC register that receives the CB tile. DestReuseOp
 *          picks srca or srcb based on ReuseType — srcb when DEST_TO_SRCA,
 *          srca when DEST_TO_SRCB. (Contrast with Load::LoadReconfig::Input,
 *          which always reconfigs srca because copy_tile is single-stream.)
 */
enum class DestReuseReconfig {
    None,
    Input,
};

/**
 * @brief PostOp: fused in-place binary op using DEST register as one operand.
 *
 * Uses binary_dest_reuse_tiles. Operand routing is controlled by ReuseType:
 *   DEST_TO_SRCA: DEST[Slot + dst_idx] -> SRCA, CB[cb_tile_idx] -> SRCB, result = op(DEST, CB)
 *   DEST_TO_SRCB: CB[cb_tile_idx] -> SRCA, DEST[Slot + dst_idx] -> SRCB, result = op(CB, DEST)
 * Result is written back to DEST[Slot + dst_idx].
 * ReuseType matters for non-commutative ops (ELWSUB); irrelevant for ELWMUL / ELWADD.
 *
 * CB lifecycle is controlled by Policy:
 *   WaitAndPop:  cb_wait_front(CB, 1) + cb_pop_front(CB, 1) each invocation.
 *                Requires cb_tile_idx == 0 (ASSERTed) — streaming pop-per-call is
 *                incompatible with indexing into a batch.
 *   WaitNoPop:   cb_wait_front(CB, cb_tile_idx + 1) only (persistent tile, reused).
 *                cb_tile_idx > 0 allowed: waits for enough tiles to cover the index.
 *   NoWaitNoPop: neither (caller owns CB lifecycle externally).
 *                cb_tile_idx > 0 allowed: caller is responsible for the pre-wait.
 *
 * Data-format reconfig for the CB side is controlled by Reconfig (see DestReuseReconfig).
 *
 * Typical uses:
 *   // batch_norm stage 2: (x - mean) *= rsqrt(var + eps).
 *   // cb_rsqrt waited upfront in the outer scope, reused across iterations.
 *   sub(cb_input, cb_mean, cb_out, shape, sfpu_chain(DestReuseMul<cb_rsqrt>{}));
 *
 *   // Non-commutative: compute (cb_mean - x) by swapping operand routing.
 *   sub(cb_x, cb_dummy, cb_out, shape,
 *       sfpu_chain(DestReuseOp<cb_mean, EltwiseBinaryType::ELWSUB,
 *                              EltwiseBinaryReuseDestType::DEST_TO_SRCB>{}));
 *
 *   // Combine with an SFPU post-transform — chain runs in order on the same DEST.
 *   sub(cb_input, cb_mean, cb_out, shape,
 *       sfpu_chain(DestReuseMul<cb_rsqrt>{}, Relu<Dst::D0>{}));
 *
 *   // Read tile 3 of a pre-waited CB instead of tile 0.
 *   auto drm = DestReuseMul<cb_scale, Dst::D0, LoadPolicy::NoWaitNoPop>{};
 *   drm.cb_tile_idx = 3;
 *   mul(cb_a, cb_b, cb_out, shape, sfpu_chain(drm));
 *
 * DestReuseOp is a chain element (extends LoadTag). Pass it through sfpu_chain()
 * to binary_op's PostOp. clashes_with_fpu is inherited from LoadTag, so binary_op
 * re-calls binary_init before each tile's binary_exec to restore the FPU state
 * (unpack MOP, math MOP, ADDR_MOD) after binary_dest_reuse_tiles_init clobbers it.
 *
 * @tparam CB        CB index for the second operand.
 * @tparam OpType    Binary op to apply (default: ELWMUL).
 * @tparam ReuseType Which SRC gets DEST (default: DEST_TO_SRCA).
 * @tparam Slot      DEST slot holding the first operand (default: D0).
 * @tparam Policy    CB wait/pop lifecycle (default: WaitNoPop — safe for persistent tile).
 * @tparam Reconfig  CB-side data-format reconfig (default: None).
 */
template <
    uint32_t CB,
    EltwiseBinaryType OpType = EltwiseBinaryType::ELWMUL,
    EltwiseBinaryReuseDestType ReuseType = EltwiseBinaryReuseDestType::DEST_TO_SRCA,
    Dst Slot = Dst::D0,
    LoadPolicy Policy = LoadPolicy::WaitNoPop,
    DestReuseReconfig Reconfig = DestReuseReconfig::None>
struct DestReuseOp {
    // NOT a LoadTag: DestReuseOp does its own binary_dest_reuse_tiles_init
    // inside exec(). If we inherited LoadTag, apply_post_chain_batched would
    // spuriously call copy_tile_to_dst_init_short(FirstLoadCB) before us,
    // which corrupts state when the chain has no actual Load to init for.
    // clashes_with_fpu stays true (explicit) so binary_op still re-runs
    // binary_init per tile to restore the AB MOP after we clobber it.
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t cb = CB;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static constexpr LoadPolicy policy = Policy;
    static constexpr bool do_wait = load_does_wait(Policy);
    static constexpr bool do_pop = load_does_pop(Policy);
    static constexpr bool is_upfront = load_is_upfront(Policy);
    static_assert(static_cast<uint32_t>(Slot) < 8, "DEST slot exceeds maximum capacity (8)");

    ALWI void init() const;
    ALWI void exec(uint32_t offset = 0) const;
    ALWI void apply(uint32_t offset = 0) const {
        init();
        exec(offset);
    }

    // Reset internal tile counter back to 0. Called by sfpu_pipeline after all
    // tiles are processed so the chain can be reused across blocks.
    ALWI void reset_tile_idx() const {
        if constexpr (is_upfront) {
            cb_tile_idx_ = 0;
        }
    }

    // Upfront CB lifecycle: pipeline bulk-waits N tiles before the tile loop
    // and bulk-pops N tiles after. Only fires for WaitUpfrontPopAtEnd policy.
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (is_upfront) {
            cb_wait_front(CB, n);
        }
    }
    ALWI void pop_upfront(uint32_t n) const {
        if constexpr (is_upfront) {
            cb_pop_front(CB, n);
        }
    }

private:
    // Internal tile counter. Meaningful only for WaitUpfrontPopAtEnd: exec()
    // self-advances it each call, pipeline zeroes it at the end of a call.
    // For every other policy it stays 0 — callers cannot override.
    mutable uint32_t cb_tile_idx_ = 0;
};

/** @brief Alias: DestReuseOp specialised to ELWMUL + DEST_TO_SRCA. */
template <
    uint32_t CB,
    Dst Slot = Dst::D0,
    LoadPolicy Policy = LoadPolicy::WaitNoPop,
    DestReuseReconfig Reconfig = DestReuseReconfig::None>
using DestReuseMul =
    DestReuseOp<CB, EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA, Slot, Policy, Reconfig>;

}  // namespace compute_kernel_lib

#include "binary_op_helpers.inl"
