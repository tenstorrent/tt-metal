// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

/**
 * @file binary_op_helpers.hpp
 * @brief Unified binary operations with direct LLK calls - massively simplified from 1399 to ~500 lines
 *
 * ## Broadcast Dimension Reference
 *
 * BroadcastDim specifies the SHAPE of operand B and how it broadcasts to match A:
 *
 * | BroadcastDim | B Shape   | B Tiles | B Replicates | What Gets Broadcast                    |
 * |--------------|-----------|---------|--------------|----------------------------------------|
 * | NONE         | [Ht, Wt]  | Ht * Wt | (none)       | All elements used as-is                |
 * | ROW          | [1, Wt]   | Wt      | Down (↓)     | Top row of each tile                   |
 * | COL          | [Ht, 1]   | Ht      | Right (→)    | Leftmost column of each tile           |
 * | SCALAR       | [1, 1]    | 1       | Both (↓→)    | Top-left element of the single tile    |
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
 */

namespace compute_kernel_lib {

// =============================================================================
// Enums for Binary Operations
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
 * Controls when to wait for input tiles and whether to pop them after processing:
 * - WaitAndPopPerTile: Wait/process/pop one tile at a time (streaming, safe for any CB size)
 * - WaitAndPopPerChunk: Wait for chunk (DEST_LIMIT tiles), process all, pop chunk
 * - WaitUpfrontNoPop: Wait for all tiles upfront, don't pop (persistent, for tile reuse)
 * - WaitUpfrontPopAtEnd: Wait for all tiles upfront, pop all at end (consume after processing)
 * - NoWaitNoPop: Caller manages wait/pop externally (preloaded, tiles already in CB)
 * - NoWaitPopAtEnd: Caller manages wait, pop all at end (preloaded, consume after processing)
 *
 * WARNING - NoWait Policies (NoWaitNoPop, NoWaitPopAtEnd):
 * These policies are DANGEROUS when used incorrectly and can cause data hazards:
 * - DO NOT use directly after other operations without prior cb_wait_front() calls
 * - ONLY use when:
 *   1. Paired with explicit cb_wait_front() before the operation, OR
 *   2. As the FIRST operation in a chain (no prior data movement or compute operations), OR
 *   3. With sharded tensors where data is pre-loaded in CB
 * - Failure to follow these rules can result in reading stale/invalid data from CB
 * - When in doubt, use WaitAndPopPerTile or WaitUpfrontNoPop for safety
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
// BinaryDataFormatReconfig Helper Functions
// =============================================================================

constexpr bool reconfig_input(BinaryDataFormatReconfig mode) {
    return mode == BinaryDataFormatReconfig::INPUT || mode == BinaryDataFormatReconfig::INPUT_AND_OUTPUT;
}

constexpr bool reconfig_output(BinaryDataFormatReconfig mode) {
    return mode == BinaryDataFormatReconfig::OUTPUT || mode == BinaryDataFormatReconfig::INPUT_AND_OUTPUT;
}

// =============================================================================
// BinaryInputPolicy Helper Functions
// =============================================================================

constexpr bool waits_per_tile(BinaryInputPolicy p) { return p == BinaryInputPolicy::WaitAndPopPerTile; }
constexpr bool waits_per_chunk(BinaryInputPolicy p) { return p == BinaryInputPolicy::WaitAndPopPerChunk; }
constexpr bool waits_upfront(BinaryInputPolicy p) {
    return p == BinaryInputPolicy::WaitUpfrontNoPop || p == BinaryInputPolicy::WaitUpfrontPopAtEnd;
}
constexpr bool waits_caller_managed(BinaryInputPolicy p) {
    return p == BinaryInputPolicy::NoWaitNoPop || p == BinaryInputPolicy::NoWaitPopAtEnd;
}

constexpr bool pops_per_tile(BinaryInputPolicy p) { return p == BinaryInputPolicy::WaitAndPopPerTile; }
constexpr bool pops_per_chunk(BinaryInputPolicy p) { return p == BinaryInputPolicy::WaitAndPopPerChunk; }
constexpr bool pops_at_end(BinaryInputPolicy p) {
    return p == BinaryInputPolicy::WaitUpfrontPopAtEnd || p == BinaryInputPolicy::NoWaitPopAtEnd;
}
constexpr bool pops_never(BinaryInputPolicy p) {
    return p == BinaryInputPolicy::WaitUpfrontNoPop || p == BinaryInputPolicy::NoWaitNoPop;
}
constexpr bool pops_caller_managed(BinaryInputPolicy p) { return p == BinaryInputPolicy::NoWaitNoPop; }

// =============================================================================
// BinaryOutputPolicy Helper Functions
// =============================================================================

constexpr bool output_per_tile(BinaryOutputPolicy p) { return p == BinaryOutputPolicy::PerTile; }
constexpr bool output_per_chunk(BinaryOutputPolicy p) { return p == BinaryOutputPolicy::PerChunk; }
constexpr bool output_bulk(BinaryOutputPolicy p) { return p == BinaryOutputPolicy::Bulk; }

// =============================================================================
// Data Types
// =============================================================================

struct BinaryInputBlockShape {
    uint32_t rows;
    uint32_t cols;

    static constexpr BinaryInputBlockShape of(uint32_t r, uint32_t c) { return {r, c}; }
    static constexpr BinaryInputBlockShape single() { return {1, 1}; }
    static constexpr BinaryInputBlockShape row(uint32_t c) { return {1, c}; }
    static constexpr BinaryInputBlockShape col(uint32_t r) { return {r, 1}; }
};

struct BinaryTileLayout {
    uint32_t row_major_a = 1, row_major_b = 1;
};

// Use types from common_types.hpp (NoAccumulation, NoOp already defined there)
struct BinaryAccumulate {
    uint32_t cb_accumulator, dst_index = 0;
};

template <BinaryOpType op_type>
constexpr EltwiseBinaryType map_to_eltwise_type() {
    return op_type == BinaryOpType::ADD ? ELWADD : op_type == BinaryOpType::SUB ? ELWSUB : ELWMUL;
}

template <BroadcastDim bcast_dim>
constexpr BroadcastType map_to_broadcast_type() {
    return bcast_dim == BroadcastDim::NONE  ? BroadcastType::NONE
           : bcast_dim == BroadcastDim::ROW ? BroadcastType::ROW
           : bcast_dim == BroadcastDim::COL ? BroadcastType::COL
                                            : BroadcastType::SCALAR;
}

template <BroadcastDim bcast_dim>
ALWI constexpr uint32_t get_b_tile_count(uint32_t Ht, uint32_t Wt) {
    return bcast_dim == BroadcastDim::SCALAR ? 1
           : bcast_dim == BroadcastDim::ROW  ? Wt
           : bcast_dim == BroadcastDim::COL  ? Ht
                                             : Ht * Wt;
}

template <typename AccumT>
ALWI constexpr bool is_accumulator_enabled() {
    return !std::is_same_v<AccumT, NoAccumulation>;
}

template <typename AccumT>
ALWI constexpr uint32_t get_binary_dst_index(AccumT accum) {
    if constexpr (is_accumulator_enabled<AccumT>()) {
        return accum.dst_index;
    } else {
        return 0;
    }
}

// =============================================================================
// Unified LLK Calls - Single Init and Exec for All Broadcast Modes
// =============================================================================

template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_init(uint32_t icb_a, uint32_t icb_b) {
    constexpr EltwiseBinaryType elt_type = map_to_eltwise_type<op_type>();
    constexpr BroadcastType bcast_type = map_to_broadcast_type<bcast_dim>();

    MATH((llk_math_eltwise_binary_init_with_operands<elt_type, bcast_type, MATH_FIDELITY>(icb_a, icb_b)));
    UNPACK((llk_unpack_AB_init<bcast_type>(icb_a, icb_b)));
}

template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_exec(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    constexpr EltwiseBinaryType elt_type = map_to_eltwise_type<op_type>();
    constexpr BroadcastType bcast_type = map_to_broadcast_type<bcast_dim>();

    UNPACK((llk_unpack_AB<bcast_type>(icb_a, icb_b, itile_a, itile_b)));
    MATH(
        (llk_math_eltwise_binary<elt_type, bcast_type, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(
            icb_a, icb_b, idst, true)));
}

// =============================================================================
// Unified Binary Operation - All Broadcast Dimensions in One Function
// =============================================================================

template <
    BinaryOpType op_type,
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_a_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy input_b_policy = input_a_policy,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void binary_op(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dest_limit = DEST_AUTO_LIMIT;
    constexpr bool is_square = (op_type == BinaryOpType::SQUARE);

    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t total_tiles_a = Ht * Wt;
    const uint32_t b_tile_count = get_b_tile_count<bcast_dim>(Ht, Wt);

    // Data format reconfiguration
    if constexpr (reconfig_input(reconfig)) {
        reconfig_data_format(icb_a, icb_b);
    }
    if constexpr (reconfig_output(reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // Initialization
    if constexpr (init) {
        binary_init<op_type, bcast_dim>(icb_a, icb_b);
    }

    // Upfront waits
    if constexpr (waits_upfront(input_a_policy)) {
        cb_wait_front(icb_a, total_tiles_a);
    }

    // B policy: ROW and SCALAR always wait upfront
    if constexpr (!is_square) {
        if constexpr (bcast_dim == BroadcastDim::ROW || bcast_dim == BroadcastDim::SCALAR) {
            if constexpr (!waits_caller_managed(input_b_policy)) {
                cb_wait_front(icb_b, b_tile_count);
            }
        } else if constexpr (waits_upfront(input_b_policy)) {
            cb_wait_front(icb_b, b_tile_count);
        }
    }

    // Upfront output reserve
    if constexpr (output_bulk(output_policy)) {
        cb_reserve_back(ocb, total_tiles_a);
    }

    const uint32_t base_dst = get_binary_dst_index(accum);
    const uint32_t effective_dest_limit = dest_limit - base_dst;
    uint32_t tiles_processed = 0;

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        for (uint32_t wt_base = 0; wt_base < Wt; wt_base += effective_dest_limit) {
            const uint32_t chunk_size = (wt_base + effective_dest_limit <= Wt) ? effective_dest_limit : (Wt - wt_base);

            // Per-chunk waits
            if constexpr (waits_per_chunk(input_a_policy)) {
                cb_wait_front(icb_a, chunk_size);
            }
            if constexpr (!is_square && waits_per_chunk(input_b_policy)) {
                if constexpr (bcast_dim == BroadcastDim::NONE) {
                    cb_wait_front(icb_b, chunk_size);
                } else if constexpr (bcast_dim == BroadcastDim::COL) {
                    cb_wait_front(icb_b, onetile);
                }
            }

            if constexpr (output_per_chunk(output_policy)) {
                cb_reserve_back(ocb, chunk_size);
            }

            tile_regs_acquire();

            // Accumulator reload if needed
            if constexpr (is_accumulator_enabled<AccumT>()) {
                cb_wait_front(accum.cb_accumulator, 1);
                copy_tile(accum.cb_accumulator, 0, accum.dst_index);
                cb_pop_front(accum.cb_accumulator, 1);
                binary_init<op_type, bcast_dim>(icb_a, icb_b);
            }

            for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                // Per-tile waits
                if constexpr (waits_per_tile(input_a_policy)) {
                    cb_wait_front(icb_a, onetile);
                }
                if constexpr (!is_square && waits_per_tile(input_b_policy)) {
                    if constexpr (bcast_dim == BroadcastDim::NONE || bcast_dim == BroadcastDim::COL) {
                        cb_wait_front(icb_b, onetile);
                    }
                }

                // Tile indices
                uint32_t tile_a, tile_b, dst_idx;

                if constexpr (waits_per_tile(input_a_policy)) {
                    tile_a = 0;
                    dst_idx = base_dst;
                } else if constexpr (waits_per_chunk(input_a_policy)) {
                    tile_a = wt;
                    dst_idx = base_dst + wt;
                } else {
                    tile_a = ht * Wt + wt_base + wt;
                    dst_idx = base_dst + wt;
                }

                if constexpr (is_square) {
                    tile_b = tile_a;
                } else if constexpr (bcast_dim == BroadcastDim::SCALAR) {
                    tile_b = 0;
                } else if constexpr (bcast_dim == BroadcastDim::ROW) {
                    tile_b = wt_base + wt;
                } else if constexpr (bcast_dim == BroadcastDim::COL) {
                    tile_b = waits_per_tile(input_b_policy) ? 0 : ht;
                } else {  // NONE
                    if constexpr (waits_per_tile(input_b_policy)) {
                        tile_b = 0;
                    } else if constexpr (waits_per_chunk(input_b_policy)) {
                        tile_b = wt;
                    } else {
                        tile_b = ht * Wt + wt_base + wt;
                    }
                }

                // Execute (unified LLK call)
                binary_exec<op_type, bcast_dim>(icb_a, icb_b, tile_a, tile_b, dst_idx);

                // Post-operation callback (e.g., rsqrt, recip)
                post_op(dst_idx);

                // Per-tile streaming
                if constexpr (waits_per_tile(input_a_policy)) {
                    tile_regs_commit();
                    tile_regs_wait();

                    if constexpr (output_per_tile(output_policy)) {
                        cb_reserve_back(ocb, onetile);
                        pack_tile(base_dst, ocb);
                        cb_push_back(ocb, onetile);
                    } else if constexpr (output_per_chunk(output_policy)) {
                        pack_tile(base_dst, ocb, wt);
                    } else {
                        pack_tile(base_dst, ocb, tiles_processed);
                    }

                    if constexpr (pops_per_tile(input_a_policy)) {
                        cb_pop_front(icb_a, onetile);
                    }
                    // Note: input_b pop for NONE broadcast only. COL broadcast pops once per row (see end of ht loop).
                    if constexpr (!is_square && pops_per_tile(input_b_policy)) {
                        if constexpr (bcast_dim == BroadcastDim::NONE) {
                            cb_pop_front(icb_b, onetile);
                        }
                    }

                    tile_regs_acquire();
                    tiles_processed++;
                }
            }

            // Per-chunk commit/pack/pop
            if constexpr (!waits_per_tile(input_a_policy)) {
                tile_regs_commit();
                tile_regs_wait();

                if constexpr (output_per_chunk(output_policy)) {
                    for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                        pack_tile(base_dst + wt, ocb, wt);
                    }
                    cb_push_back(ocb, chunk_size);
                } else if constexpr (output_bulk(output_policy)) {
                    for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                        pack_tile(base_dst + wt, ocb, tiles_processed + wt);
                    }
                } else {
                    for (uint32_t wt = 0; wt < chunk_size; ++wt) {
                        cb_reserve_back(ocb, onetile);
                        pack_tile(base_dst + wt, ocb);
                        cb_push_back(ocb, onetile);
                    }
                }

                tiles_processed += chunk_size;
            }

            if constexpr (pops_per_chunk(input_a_policy)) {
                cb_pop_front(icb_a, chunk_size);
            }
            if constexpr (!is_square && pops_per_chunk(input_b_policy)) {
                if constexpr (bcast_dim == BroadcastDim::NONE) {
                    cb_pop_front(icb_b, chunk_size);
                } else if constexpr (bcast_dim == BroadcastDim::COL) {
                    cb_pop_front(icb_b, onetile);
                }
            }

            tile_regs_release();
        }

        // COL broadcast: pop input_b once per row (ht iteration).
        // This is decoupled from input_a's policy - controlled solely by input_b's policy.
        if constexpr (!is_square && bcast_dim == BroadcastDim::COL && pops_per_tile(input_b_policy)) {
            cb_pop_front(icb_b, onetile);
        }
    }

    // Bulk output push
    if constexpr (output_bulk(output_policy)) {
        cb_push_back(ocb, total_tiles_a);
    }

    // At-end pops
    if constexpr (pops_at_end(input_a_policy)) {
        cb_pop_front(icb_a, total_tiles_a);
    }
    if constexpr (!is_square && pops_at_end(input_b_policy)) {
        cb_pop_front(icb_b, b_tile_count);
    }

    // B pop for ROW/SCALAR (unless caller-managed, never, or already popped at end)
    if constexpr (!is_square) {
        if constexpr (bcast_dim == BroadcastDim::ROW || bcast_dim == BroadcastDim::SCALAR) {
            if constexpr (
                !pops_caller_managed(input_b_policy) && !pops_never(input_b_policy) && !pops_at_end(input_b_policy)) {
                cb_pop_front(icb_b, b_tile_count);
            }
        }
    }
}

// =============================================================================
// Convenience Aliases - Full Control APIs
// =============================================================================

template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_a_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy input_b_policy = input_a_policy,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void add(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    binary_op<
        BinaryOpType::ADD,
        bcast_dim,
        input_a_policy,
        input_b_policy,
        output_policy,
        reconfig,
        init,
        AccumT,
        PostOp>(icb_a, icb_b, ocb, shape, layout, accum, post_op);
}

template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_a_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy input_b_policy = input_a_policy,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void sub(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    binary_op<
        BinaryOpType::SUB,
        bcast_dim,
        input_a_policy,
        input_b_policy,
        output_policy,
        reconfig,
        init,
        AccumT,
        PostOp>(icb_a, icb_b, ocb, shape, layout, accum, post_op);
}

template <
    BroadcastDim bcast_dim = BroadcastDim::NONE,
    BinaryInputPolicy input_a_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy input_b_policy = input_a_policy,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void mul(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    binary_op<
        BinaryOpType::MUL,
        bcast_dim,
        input_a_policy,
        input_b_policy,
        output_policy,
        reconfig,
        init,
        AccumT,
        PostOp>(icb_a, icb_b, ocb, shape, layout, accum, post_op);
}

template <
    BinaryInputPolicy input_policy = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryOutputPolicy output_policy = BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig reconfig = BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    bool init = true,
    typename AccumT = NoAccumulation,
    typename PostOp = NoOp>
ALWI void square(
    uint32_t icb,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    BinaryTileLayout layout = {},
    AccumT accum = {},
    PostOp post_op = {}) {
    binary_op<
        BinaryOpType::SQUARE,
        BroadcastDim::NONE,
        input_policy,
        input_policy,
        output_policy,
        reconfig,
        init,
        AccumT,
        PostOp>(icb, icb, ocb, shape, layout, accum, post_op);
}

}  // namespace compute_kernel_lib
