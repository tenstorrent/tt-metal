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
#include <utility>

#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

// =============================================================================
// Internal Helpers (not part of public API)
// =============================================================================

constexpr bool reconfig_srca(BinaryDataFormatReconfig mode) {
    return mode == BinaryDataFormatReconfig::INPUT || mode == BinaryDataFormatReconfig::INPUT_AND_OUTPUT ||
           mode == BinaryDataFormatReconfig::SRCA_ONLY || mode == BinaryDataFormatReconfig::SRCA_ONLY_AND_OUTPUT;
}

constexpr bool reconfig_srcb(BinaryDataFormatReconfig mode) {
    return mode == BinaryDataFormatReconfig::INPUT || mode == BinaryDataFormatReconfig::INPUT_AND_OUTPUT ||
           mode == BinaryDataFormatReconfig::SRCB_ONLY || mode == BinaryDataFormatReconfig::SRCB_ONLY_AND_OUTPUT;
}

constexpr bool reconfig_output(BinaryDataFormatReconfig mode) {
    return mode == BinaryDataFormatReconfig::OUTPUT || mode == BinaryDataFormatReconfig::INPUT_AND_OUTPUT ||
           mode == BinaryDataFormatReconfig::SRCA_ONLY_AND_OUTPUT ||
           mode == BinaryDataFormatReconfig::SRCB_ONLY_AND_OUTPUT;
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

namespace detail {

// is_sfpu_chain_v<T>: true iff T is a specialization of SfpuChain.
template <typename T>
struct is_sfpu_chain : std::false_type {};
template <typename... Ops>
struct is_sfpu_chain<SfpuChain<Ops...>> : std::true_type {};
template <typename T>
inline constexpr bool is_sfpu_chain_v =
    is_sfpu_chain<std::remove_cv_t<std::remove_reference_t<T>>>::value;

// has_dst_idx_v<T>: SFINAE probe. UnaryOp CRTP and Load expose dst_idx;
// BinaryOp, TernaryOp, and CompactLoad do not.
template <typename, typename = void>
struct has_dst_idx : std::false_type {};
template <typename T>
struct has_dst_idx<T, std::void_t<decltype(T::dst_idx)>> : std::true_type {};
template <typename T>
inline constexpr bool has_dst_idx_v = has_dst_idx<T>::value;

// dst_idx_value_v<T>: value-fetch with fallback. Needed because
// `has_dst_idx_v<T> && (T::dst_idx == 0u)` in a constexpr expression does NOT
// short-circuit for SFINAE purposes: the subexpression T::dst_idx is evaluated
// even when has_dst_idx_v<T> is false, yielding a hard error for types that
// lack dst_idx (e.g. CompactLoad). This indirection guards the member access.
template <typename T, bool HasIdx = has_dst_idx_v<T>>
struct dst_idx_value {
    static constexpr uint32_t value = static_cast<uint32_t>(-1);
};
template <typename T>
struct dst_idx_value<T, true> {
    static constexpr uint32_t value = T::dst_idx;
};
template <typename T>
inline constexpr uint32_t dst_idx_value_v = dst_idx_value<T>::value;

// chain_has_loads_v<T>: true when T is an SfpuChain that contains at least one Load op.
template <typename T>
struct chain_has_loads : std::false_type {};
template <typename... Ops>
struct chain_has_loads<SfpuChain<Ops...>> : std::bool_constant<(is_load_op_v<Ops> || ...)> {};
template <typename T>
inline constexpr bool chain_has_loads_v =
    chain_has_loads<std::remove_cv_t<std::remove_reference_t<T>>>::value;

// Retrieve the I-th element of an SfpuChain by instance (preserves runtime fields).
template <std::size_t I, typename First, typename... Rest>
ALWI constexpr auto& chain_get(const SfpuChain<First, Rest...>& c) {
    if constexpr (I == 0) {
        return c.first;
    } else {
        return chain_get<I - 1>(c.rest);
    }
}

// One-init, k-exec over DEST slots [base_dst, base_dst + chunk_size).
// Load ops in the chain manage their own CB lifecycle via exec() (wait/pop/copy).
template <typename... Ops, std::size_t... I>
ALWI void apply_post_chain_batched_impl(
    const SfpuChain<Ops...>& chain,
    uint32_t base_dst,
    uint32_t chunk_size,
    std::index_sequence<I...>) {
    auto run_one = [&](const auto& op) {
        op.init();
        for (uint32_t k = 0; k < chunk_size; ++k) {
            op.exec(base_dst + k);
        }
    };
    (run_one(chain_get<I>(chain)), ...);
}

template <typename... Ops>
ALWI void apply_post_chain_batched(
    const SfpuChain<Ops...>& chain, uint32_t base_dst, uint32_t chunk_size) {
    // If the chain has Load ops, init the unpacker for copy_tile before running.
    if constexpr ((is_load_op_v<Ops> || ...)) {
        copy_tile_to_dst_init_short(FirstLoadCB<SfpuChain<Ops...>>::value);
    }
    apply_post_chain_batched_impl(chain, base_dst, chunk_size, std::index_sequence_for<Ops...>{});
}

// post_op_needs_reinit_v<T>: true when binary_op must re-call binary_init before each
// tile's exec (instead of once before the tile loop). Two triggers:
//   1. PostOp sets T::needs_parent_reinit = true (e.g. DestReuseOp, which calls
//      binary_dest_reuse_tiles_init and thus clobbers the unpack pipeline).
//   2. PostOp is an SfpuChain containing Load ops (Load calls copy_tile_to_dst_init_short
//      which also clobbers the unpack pipeline).
template <typename T, typename = void>
struct has_needs_parent_reinit : std::false_type {};
template <typename T>
struct has_needs_parent_reinit<T, std::void_t<decltype(T::needs_parent_reinit)>>
    : std::bool_constant<T::needs_parent_reinit> {};
template <typename T>
inline constexpr bool post_op_needs_reinit_v =
    has_needs_parent_reinit<T>::value || (is_sfpu_chain_v<T> && chain_has_loads_v<T>);

}  // namespace detail

// =============================================================================
// DestReuseOp PostOp Implementation
// =============================================================================

template <
    uint32_t CB,
    EltwiseBinaryType OpType,
    EltwiseBinaryReuseDestType ReuseType,
    Dst Slot,
    LoadPolicy Policy,
    DestReuseReconfig Reconfig>
ALWI void DestReuseOp<CB, OpType, ReuseType, Slot, Policy, Reconfig>::operator()(uint32_t dst_idx) const {
    if constexpr (do_wait) {
        // Wait for enough tiles to cover cb_tile_idx (minimum 1 tile for index 0).
        cb_wait_front(CB, cb_tile_idx + 1);
    }
    if constexpr (Reconfig == DestReuseReconfig::Input) {
        // DEST_TO_SRCA: CB feeds SRCB, so reconfig srcb. Inverse for DEST_TO_SRCB.
        if constexpr (ReuseType == EltwiseBinaryReuseDestType::DEST_TO_SRCA) {
            reconfig_data_format_srcb(CB);
        } else {
            reconfig_data_format_srca(CB);
        }
    }
    binary_dest_reuse_tiles_init<OpType, ReuseType>(CB);
    // DEST slot is Slot + dst_idx to handle both per-tile (dst_idx=0) and
    // per-chunk (dst_idx=k) policies correctly.
    binary_dest_reuse_tiles<OpType, ReuseType>(CB, cb_tile_idx, static_cast<uint32_t>(Slot) + dst_idx);
    if constexpr (do_pop) {
        cb_pop_front(CB, 1);
    }
}

// =============================================================================
// Unified LLK Calls - Single Init and Exec for All Broadcast Modes
// =============================================================================

template <BinaryOpType op_type, BroadcastDim bcast_dim>
ALWI void binary_init(uint32_t icb_a, uint32_t icb_b) {
    constexpr EltwiseBinaryType elt_type = map_to_eltwise_type<op_type>();
    constexpr BroadcastType bcast_type = map_to_broadcast_type<bcast_dim>();

    // MUL uses configured MATH_FIDELITY; ADD/SUB always use LoFi (matches eltwise_binary.h)
    if constexpr (op_type == BinaryOpType::MUL) {
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
    if constexpr (op_type == BinaryOpType::MUL) {
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
    if constexpr (reconfig_srca(reconfig) && reconfig_srcb(reconfig)) {
        reconfig_data_format(icb_a, icb_b);
    } else if constexpr (reconfig_srca(reconfig)) {
        reconfig_data_format_srca(icb_a);
    } else if constexpr (reconfig_srcb(reconfig)) {
        reconfig_data_format_srcb(icb_b);
    }
    if constexpr (reconfig_output(reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // Initialization: done once here unless PostOp modifies the unpack pipeline,
    // in which case it is re-done per tile (see binary_exec site below).
    if constexpr (init && !detail::post_op_needs_reinit_v<PostOp>) {
        binary_init<op_type, bcast_dim>(icb_a, icb_b);
    }

    // Same-CB flag: when icb_a == icb_b, skip all duplicate wait/pop for icb_b.
    const bool same_cb = (!is_square && icb_a == icb_b);

    // Upfront waits
    if constexpr (waits_upfront(input_a_policy)) {
        cb_wait_front(icb_a, total_tiles_a);
    }

    // B policy: ROW and SCALAR always wait upfront
    if constexpr (!is_square) {
        if constexpr (bcast_dim == BroadcastDim::ROW || bcast_dim == BroadcastDim::SCALAR) {
            if constexpr (!waits_caller_managed(input_b_policy)) {
                if (!same_cb) cb_wait_front(icb_b, b_tile_count);
            }
        } else if constexpr (waits_upfront(input_b_policy)) {
            if (!same_cb) cb_wait_front(icb_b, b_tile_count);
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
                    if (!same_cb) cb_wait_front(icb_b, chunk_size);
                } else if constexpr (bcast_dim == BroadcastDim::COL) {
                    if (!same_cb) cb_wait_front(icb_b, onetile);
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
                        if (!same_cb) cb_wait_front(icb_b, onetile);
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

                // Per-tile reinit when PostOp modifies the unpack pipeline (e.g. DestReuseMul).
                if constexpr (init && detail::post_op_needs_reinit_v<PostOp>) {
                    binary_init<op_type, bcast_dim>(icb_a, icb_b);
                }

                // Execute (unified LLK call)
                binary_exec<op_type, bcast_dim>(icb_a, icb_b, tile_a, tile_b, dst_idx);

                // Post-operation dispatch:
                //   non-chain PostOp  -> call per tile (existing behavior, every policy)
                //   chain + per-tile  -> once-init, 1-exec per tile (chunk_size == 1)
                //   chain + per-chunk -> handled below the wt loop, not here
                if constexpr (!detail::is_sfpu_chain_v<PostOp>) {
                    post_op(dst_idx);
                } else if constexpr (waits_per_tile(input_a_policy)) {
                    detail::apply_post_chain_batched(post_op, dst_idx, 1u);
                }

                // Per-tile input_b pop — decoupled from input_a's policy.
                // For NONE broadcast, pop B after each tile regardless of A's streaming mode.
                // COL broadcast pops once per row (see end of ht loop), not per tile.
                if constexpr (!is_square && pops_per_tile(input_b_policy)) {
                    if constexpr (bcast_dim == BroadcastDim::NONE) {
                        if (!same_cb) cb_pop_front(icb_b, onetile);
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

            // Chunk-scoped chain post-op: once-init, chunk_size-exec over [base_dst, base_dst + chunk_size).
            // Fires only for chain PostOp in per-chunk / bulk / upfront / caller-managed policies.
            if constexpr (detail::is_sfpu_chain_v<PostOp> && !waits_per_tile(input_a_policy)) {
                detail::apply_post_chain_batched(post_op, base_dst, chunk_size);
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
                    if (!same_cb) cb_pop_front(icb_b, chunk_size);
                } else if constexpr (bcast_dim == BroadcastDim::COL) {
                    if (!same_cb) cb_pop_front(icb_b, onetile);
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
            if (!same_cb) cb_pop_front(icb_b, onetile);
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
        if (!same_cb) cb_pop_front(icb_b, b_tile_count);
    }

    // B pop for ROW/SCALAR (unless caller-managed, never, or already popped at end)
    if constexpr (!is_square) {
        if constexpr (bcast_dim == BroadcastDim::ROW || bcast_dim == BroadcastDim::SCALAR) {
            if constexpr (
                !pops_caller_managed(input_b_policy) && !pops_never(input_b_policy) && !pops_at_end(input_b_policy)) {
                if (!same_cb) cb_pop_front(icb_b, b_tile_count);
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
