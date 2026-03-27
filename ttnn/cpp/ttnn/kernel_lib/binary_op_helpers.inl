// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file binary_op_helpers.inl
 * @brief Implementation of binary operation helper functions
 *
 * This file contains the implementation details for binary_op() and its convenience aliases.
 * It should only be included by binary_op_helpers.hpp.
 */

#include <type_traits>

#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

// =============================================================================
// Internal Helpers (not part of public API)
// =============================================================================

constexpr bool reconfig_input(BinaryDataFormatReconfig mode) {
    return mode == BinaryDataFormatReconfig::INPUT || mode == BinaryDataFormatReconfig::INPUT_AND_OUTPUT;
}

constexpr bool reconfig_output(BinaryDataFormatReconfig mode) {
    return mode == BinaryDataFormatReconfig::OUTPUT || mode == BinaryDataFormatReconfig::INPUT_AND_OUTPUT;
}

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

constexpr bool output_per_tile(BinaryOutputPolicy p) { return p == BinaryOutputPolicy::PerTile; }
constexpr bool output_per_chunk(BinaryOutputPolicy p) { return p == BinaryOutputPolicy::PerChunk; }
constexpr bool output_bulk(BinaryOutputPolicy p) { return p == BinaryOutputPolicy::Bulk; }

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

    // MUL and SQUARE use configured MATH_FIDELITY; ADD/SUB always use LoFi (matches eltwise_binary.h)
    if constexpr (op_type == BinaryOpType::MUL || op_type == BinaryOpType::SQUARE) {
        MATH((llk_math_eltwise_binary_init_with_operands<elt_type, bcast_type, MATH_FIDELITY>(icb_a, icb_b)));
    } else {
        MATH((llk_math_eltwise_binary_init_with_operands<elt_type, bcast_type, MathFidelity::LoFi>(icb_a, icb_b)));
    }
    UNPACK((llk_unpack_AB_init<bcast_type>(icb_a, icb_b)));
}

template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_exec(uint32_t icb_a, uint32_t icb_b, uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    constexpr EltwiseBinaryType elt_type = map_to_eltwise_type<op_type>();
    constexpr BroadcastType bcast_type = map_to_broadcast_type<bcast_dim>();

    UNPACK((llk_unpack_AB<bcast_type>(icb_a, icb_b, itile_a, itile_b)));
    if constexpr (op_type == BinaryOpType::MUL || op_type == BinaryOpType::SQUARE) {
        MATH((llk_math_eltwise_binary<elt_type, bcast_type, DST_ACCUM_MODE, MATH_FIDELITY,
                                      EltwiseBinaryReuseDestType::NONE>(icb_a, icb_b, idst, true)));
    } else {
        MATH((llk_math_eltwise_binary<elt_type, bcast_type, DST_ACCUM_MODE, MathFidelity::LoFi,
                                      EltwiseBinaryReuseDestType::NONE>(icb_a, icb_b, idst, true)));
    }
}

// =============================================================================
// Unified Binary Operation - All Broadcast Dimensions in One Function
// =============================================================================

template <
    BinaryOpType op_type,
    BroadcastDim bcast_dim,
    BinaryInputPolicy input_a_policy,
    BinaryInputPolicy input_b_policy,
    BinaryOutputPolicy output_policy,
    BinaryDataFormatReconfig reconfig,
    bool init,
    typename PostOp,
    typename AccumT>
ALWI void binary_op(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op,
    AccumT accum) {
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

            // Per-tile path: each tile gets its own acquire/commit/wait/release cycle
            // Per-chunk path: one acquire/commit/wait/release cycle for the whole chunk
            if constexpr (!waits_per_tile(input_a_policy)) {
                tile_regs_acquire();
            }

            // Accumulator reload if needed
            if constexpr (is_accumulator_enabled<AccumT>()) {
                if constexpr (!waits_per_tile(input_a_policy)) {
                    cb_wait_front(accum.cb_accumulator, 1);
                    copy_tile(accum.cb_accumulator, 0, accum.dst_index);
                    cb_pop_front(accum.cb_accumulator, 1);
                    binary_init<op_type, bcast_dim>(icb_a, icb_b);
                }
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

                // Per-tile: acquire DEST for this tile
                if constexpr (waits_per_tile(input_a_policy)) {
                    tile_regs_acquire();
                }

                // Execute (unified LLK call)
                binary_exec<op_type, bcast_dim>(icb_a, icb_b, tile_a, tile_b, dst_idx);

                // Post-operation callback (e.g., rsqrt, recip)
                post_op(dst_idx);

                // Per-tile input_b pop — decoupled from input_a's policy.
                // For NONE broadcast, pop B after each tile regardless of A's streaming mode.
                // COL broadcast pops once per row (see end of ht loop), not per tile.
                if constexpr (!is_square && pops_per_tile(input_b_policy)) {
                    if constexpr (bcast_dim == BroadcastDim::NONE) {
                        cb_pop_front(icb_b, onetile);
                    }
                }

                // Per-tile streaming: commit, wait, pack, release — complete handshake per tile
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

                    tile_regs_release();
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

            // Per-chunk: release after all tiles packed
            if constexpr (!waits_per_tile(input_a_policy)) {
                tile_regs_release();
            }
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
    BroadcastDim bcast_dim,
    BinaryInputPolicy input_a_policy,
    BinaryInputPolicy input_b_policy,
    BinaryOutputPolicy output_policy,
    BinaryDataFormatReconfig reconfig,
    bool init,
    typename PostOp,
    typename AccumT>
ALWI void add(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op,
    AccumT accum) {
    binary_op<
        BinaryOpType::ADD,
        bcast_dim,
        input_a_policy,
        input_b_policy,
        output_policy,
        reconfig,
        init,
        PostOp,
        AccumT>(icb_a, icb_b, ocb, shape, post_op, accum);
}

template <
    BroadcastDim bcast_dim,
    BinaryInputPolicy input_a_policy,
    BinaryInputPolicy input_b_policy,
    BinaryOutputPolicy output_policy,
    BinaryDataFormatReconfig reconfig,
    bool init,
    typename PostOp,
    typename AccumT>
ALWI void sub(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op,
    AccumT accum) {
    binary_op<
        BinaryOpType::SUB,
        bcast_dim,
        input_a_policy,
        input_b_policy,
        output_policy,
        reconfig,
        init,
        PostOp,
        AccumT>(icb_a, icb_b, ocb, shape, post_op, accum);
}

template <
    BroadcastDim bcast_dim,
    BinaryInputPolicy input_a_policy,
    BinaryInputPolicy input_b_policy,
    BinaryOutputPolicy output_policy,
    BinaryDataFormatReconfig reconfig,
    bool init,
    typename PostOp,
    typename AccumT>
ALWI void mul(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op,
    AccumT accum) {
    binary_op<
        BinaryOpType::MUL,
        bcast_dim,
        input_a_policy,
        input_b_policy,
        output_policy,
        reconfig,
        init,
        PostOp,
        AccumT>(icb_a, icb_b, ocb, shape, post_op, accum);
}

template <
    BinaryInputPolicy input_policy,
    BinaryOutputPolicy output_policy,
    BinaryDataFormatReconfig reconfig,
    bool init,
    typename PostOp,
    typename AccumT>
ALWI void square(
    uint32_t icb,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op,
    AccumT accum) {
    binary_op<
        BinaryOpType::SQUARE,
        BroadcastDim::NONE,
        input_policy,
        input_policy,
        output_policy,
        reconfig,
        init,
        PostOp,
        AccumT>(icb, icb, ocb, shape, post_op, accum);
}

}  // namespace compute_kernel_lib
