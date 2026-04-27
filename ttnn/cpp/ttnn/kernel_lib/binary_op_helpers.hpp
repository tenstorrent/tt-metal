// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

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
 *   // 9. Post-operation callback — apply rsqrt after multiply
 *   mul(cb_in, cb_in, cb_out, BinaryInputBlockShape::of(Ht, Wt),
 *       [](uint32_t dst_idx) {
 *           rsqrt_tile_init();
 *           rsqrt_tile(dst_idx);
 *       });
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
 * Controls whether unpacker (input) and/or packer (output) are reconfigured:
 * - NONE: Skip all reconfiguration (binary op is first op or formats match)
 * - INPUT: Reconfigure unpacker only (input format changed)
 * - OUTPUT: Reconfigure packer only (output format changed)
 * - INPUT_AND_OUTPUT: Reconfigure both unpacker and packer (default, safest option)
 */
enum class BinaryDataFormatReconfig { NONE = 0, INPUT = 1, OUTPUT = 2, INPUT_AND_OUTPUT = 3 };

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
    NoWaitPopAtEnd        // Caller manages wait, pop at end (preloaded, consume)
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
    uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {}, AccumT accum = {});

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
    uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {}, AccumT accum = {});

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
    uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {}, AccumT accum = {});

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
    uint32_t icb_a, uint32_t icb_b, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {}, AccumT accum = {});

/** @brief Element-wise square: A * A. Uses a single input CB. See binary_op() for full documentation. */
template <
    BinaryInputPolicy input_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp,
    typename AccumT = NoAccumulation>
ALWI void square(uint32_t icb, uint32_t ocb, BinaryInputBlockShape shape, PostOp post_op = {}, AccumT accum = {});

// =============================================================================
// In-Place Binary Operations
// =============================================================================

/**
 * @brief In-place binary operation: cb_a = cb_a op B
 *
 * Modifies tiles in cb_a by applying (cb_a op icb_b) and writing the result
 * back to cb_a. Processes one tile at a time using the pop-before-pack cycle:
 *
 *     wait_front(cb_a, 1)  →  unpack A[front] + B[tile_b]  →  pop_front(cb_a, 1)
 *     →  compute in DEST  →  reserve_back(cb_a, 1)  →  pack  →  push_back(cb_a, 1)
 *
 * Each popped slot is immediately reused by the pack, so the CB tile count
 * stays constant throughout. After Ht*Wt iterations the modified tiles occupy
 * the same slots in the original order.
 *
 * ## Prerequisites
 *
 *   1. Call compute_kernel_hw_startup() once at kernel start (the only full init).
 *   2. All Ht*Wt tiles of A must already be present in cb_a before calling.
 *      In a multi-phase kernel, fill cb_a in a prior phase (e.g. copy_tiles).
 *   3. cb_a must be sized for at least Ht*Wt tiles.
 *
 * ## CB Ownership Rule
 *
 *   cb_a must be exclusively owned by the compute kernel — no reader (NCRISC)
 *   or writer (BRISC) may push/pop on cb_a while this function runs.
 *   Violating this causes silent data corruption: cb_push_back updates the
 *   write pointer, and two threads updating the same pointer is undefined
 *   behavior (see cb_api.h).
 *
 *   In practice this means cb_a should be an intermediate CB that only the
 *   compute kernel reads from and writes to. Use a separate CB for
 *   reader→compute streaming and another for compute→writer streaming.
 *
 * ## Data Format Reconfiguration
 *
 *   Use reconfig=INPUT_AND_OUTPUT (default) when transitioning from a
 *   different operation (e.g. copy_tiles → add_in_place). This calls
 *   reconfig_data_format(cb_a, icb_b) and pack_reconfig_data_format(cb_a)
 *   to switch the unpacker/packer to the correct CB formats. Essential for
 *   mixed-precision (e.g. cb_a=bfloat16, icb_b=bfloat8_b).
 *
 *   Use reconfig=NONE only when the hardware is already configured for cb_a
 *   and icb_b (e.g. binary_op_in_place is the first and only operation).
 *
 * ## Restrictions
 *
 *   - Input A policy is always per-tile (no template parameter). Chunk/bulk
 *     modes would require popping multiple tiles before packing, breaking the
 *     one-slot-at-a-time invariant.
 *   - No accumulation support (AccumT) — accumulation needs a persistent DEST
 *     slot across tiles, which conflicts with the per-tile DEST[0] reuse.
 *   - SQUARE: supported via square_in_place(). Uses a single CB (cb_a as both
 *     inputs), so broadcast and input_b_policy are ignored.
 *
 * ── Examples ────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
 *   #include "ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp"
 *   using namespace compute_kernel_lib;
 *
 *   // One full init per kernel
 *   compute_kernel_hw_startup(cb_input, cb_work);
 *
 *   // Phase 1: Fill cb_work from reader's CB
 *   copy_tiles(cb_input, cb_work, Ht * Wt);
 *
 *   // Phase 2: In-place add (reconfig switches formats from copy → binary)
 *   add_in_place(cb_work, cb_b, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // Phase 3: Drain to output
 *   copy_tile_to_dst_init_short(cb_work);
 *   copy_tiles<CopyInputPolicy::WaitAndPop, CopyDataFormatReconfig::OUTPUT>(
 *       cb_work, cb_out, Ht * Wt);
 *
 *   // --- Broadcast examples ---
 *
 *   // Subtract per-row mean (after REDUCE_ROW, mean is [Ht, 1] tiles)
 *   sub_in_place<BroadcastDim::COL>(cb_work, cb_mean, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // Scale by a single value (scalar broadcast)
 *   mul_in_place<BroadcastDim::SCALAR>(cb_work, cb_scale, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // B tiles pre-loaded by caller — skip B wait/pop
 *   mul_in_place<BroadcastDim::ROW, BinaryInputPolicy::NoWaitNoPop>(
 *       cb_work, cb_weights, BinaryInputBlockShape::of(Ht, Wt));
 *
 *   // Post-op callback: in-place add then apply rsqrt
 *   add_in_place(cb_work, cb_var, BinaryInputBlockShape::of(Ht, Wt),
 *       [](uint32_t dst_idx) {
 *           rsqrt_tile_init();
 *           rsqrt_tile(dst_idx);
 *       });
 *
 *   // In-place square: cb_work = cb_work * cb_work (single CB, no B operand)
 *   square_in_place(cb_work, BinaryInputBlockShape::of(Ht, Wt));
 *
 * @tparam op_type       Binary operation: ADD, SUB, or MUL (not SQUARE)
 * @tparam bcast_dim     Broadcast mode for operand B (default: NONE)
 * @tparam input_b_policy  Input B synchronization policy (default: WaitAndPopPerTile)
 * @tparam reconfig      Data format reconfiguration mode (default: INPUT_AND_OUTPUT)
 * @tparam init          Whether to call binary_init (default: true)
 *
 * @param cb_a   Circular buffer for operand A (read and written in-place)
 * @param icb_b  Input circular buffer for operand B
 * @param shape  Tile grid dimensions — use BinaryInputBlockShape::of(Ht, Wt)
 * @param post_op  Post-operation callback receiving dst_idx (default: NoOp)
 */
template <
    BinaryOpType op_type,
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_b_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp>
ALWI void binary_op_in_place(uint32_t cb_a, uint32_t icb_b, BinaryInputBlockShape shape, PostOp post_op = {});

/** @brief In-place addition: cb_a = cb_a + B. See binary_op_in_place() for full documentation. */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_b_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp>
ALWI void add_in_place(uint32_t cb_a, uint32_t icb_b, BinaryInputBlockShape shape, PostOp post_op = {});

/** @brief In-place subtraction: cb_a = cb_a - B. See binary_op_in_place() for full documentation. */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_b_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp>
ALWI void sub_in_place(uint32_t cb_a, uint32_t icb_b, BinaryInputBlockShape shape, PostOp post_op = {});

/** @brief In-place multiplication: cb_a = cb_a * B. See binary_op_in_place() for full documentation. */
template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_b_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp>
ALWI void mul_in_place(uint32_t cb_a, uint32_t icb_b, BinaryInputBlockShape shape, PostOp post_op = {});

/** @brief In-place square: cb_a = cb_a * cb_a. Single CB, no B operand. See binary_op_in_place(). */
template <
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename PostOp = NoOp>
ALWI void square_in_place(uint32_t cb_a, BinaryInputBlockShape shape, PostOp post_op = {});

}  // namespace compute_kernel_lib

#include "binary_op_helpers.inl"
