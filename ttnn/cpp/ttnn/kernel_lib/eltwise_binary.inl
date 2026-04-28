// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_binary.inl
 * @brief Implementation for `eltwise_binary.hpp`. Only included by it.
 */

namespace compute_kernel_lib::eltwise {

namespace detail {

// =============================================================================
// LLK type mappers
// =============================================================================

template <BinaryOpType OpType>
constexpr EltwiseBinaryType to_elt_type() {
    if constexpr (OpType == BinaryOpType::ADD) return EltwiseBinaryType::ELWADD;
    else if constexpr (OpType == BinaryOpType::SUB) return EltwiseBinaryType::ELWSUB;
    else                                            return EltwiseBinaryType::ELWMUL;
}

template <BroadcastDim Bcast>
constexpr BroadcastType to_bcast_type() {
    if constexpr (Bcast == BroadcastDim::NONE)   return BroadcastType::NONE;
    else if constexpr (Bcast == BroadcastDim::ROW) return BroadcastType::ROW;
    else if constexpr (Bcast == BroadcastDim::COL) return BroadcastType::COL;
    else                                          return BroadcastType::SCALAR;
}

template <DestReuseType R>
constexpr EltwiseBinaryReuseDestType to_reuse_type() {
    return (R == DestReuseType::ToSrcA) ? EltwiseBinaryReuseDestType::DEST_TO_SRCA
                                        : EltwiseBinaryReuseDestType::DEST_TO_SRCB;
}

constexpr bool reconfig_input(BinaryDataFormatReconfig m) {
    return m == BinaryDataFormatReconfig::INPUT ||
           m == BinaryDataFormatReconfig::INPUT_AND_OUTPUT;
}
constexpr bool reconfig_output(BinaryDataFormatReconfig m) {
    return m == BinaryDataFormatReconfig::OUTPUT ||
           m == BinaryDataFormatReconfig::INPUT_AND_OUTPUT;
}

// =============================================================================
// Raw-LLK init/exec for FPU binary
// =============================================================================

template <BinaryOpType OpType, BroadcastDim Bcast>
ALWI void binary_short_init(uint32_t icb_a, uint32_t icb_b) {
    constexpr EltwiseBinaryType elt = to_elt_type<OpType>();
    constexpr BroadcastType btype   = to_bcast_type<Bcast>();
    if constexpr (OpType == BinaryOpType::MUL) {
        MATH((llk_math_eltwise_binary_init_with_operands<elt, btype, MATH_FIDELITY>(icb_a, icb_b)));
    } else {
        MATH((llk_math_eltwise_binary_init_with_operands<elt, btype, MathFidelity::LoFi>(icb_a, icb_b)));
    }
    UNPACK((llk_unpack_AB_init<btype>(icb_a, icb_b)));
}

template <BinaryOpType OpType, BroadcastDim Bcast>
ALWI void binary_exec_one(
    uint32_t icb_a, uint32_t icb_b,
    uint32_t itile_a, uint32_t itile_b, uint32_t idst) {
    constexpr EltwiseBinaryType elt = to_elt_type<OpType>();
    constexpr BroadcastType btype   = to_bcast_type<Bcast>();
    UNPACK((llk_unpack_AB<btype>(icb_a, icb_b, itile_a, itile_b)));
    if constexpr (OpType == BinaryOpType::MUL) {
        MATH((llk_math_eltwise_binary<elt, btype, DST_ACCUM_MODE, MATH_FIDELITY,
                                      EltwiseBinaryReuseDestType::NONE>(icb_a, icb_b, idst, true)));
    } else {
        MATH((llk_math_eltwise_binary<elt, btype, DST_ACCUM_MODE, MathFidelity::LoFi,
                                      EltwiseBinaryReuseDestType::NONE>(icb_a, icb_b, idst, true)));
    }
}

// =============================================================================
// Policy classification helpers
// =============================================================================

constexpr bool a_is_per_tile(BinaryInputPolicy p) { return p == BinaryInputPolicy::WaitAndPopPerTile; }
constexpr bool a_is_upfront_wait(BinaryInputPolicy p) {
    return p == BinaryInputPolicy::WaitUpfrontNoPop ||
           p == BinaryInputPolicy::WaitUpfrontPopAtEnd;
}
constexpr bool a_pops_at_end(BinaryInputPolicy p) {
    return p == BinaryInputPolicy::WaitUpfrontPopAtEnd ||
           p == BinaryInputPolicy::NoWaitPopAtEnd;
}
constexpr bool a_caller_managed_wait(BinaryInputPolicy p) {
    return p == BinaryInputPolicy::NoWaitNoPop ||
           p == BinaryInputPolicy::NoWaitPopAtEnd;
}

}  // namespace detail

// =============================================================================
// DestReuseOp method definitions
// =============================================================================

template <uint32_t CB, BinaryOpType OpType, DestReuseType Reuse, Dst Slot,
          DestReuseInputPolicy Policy, DestReuseReconfig Reconfig>
ALWI void DestReuseOp<CB, OpType, Reuse, Slot, Policy, Reconfig>::init() const {
    if constexpr (Reconfig == DestReuseReconfig::Input) {
        if constexpr (Reuse == DestReuseType::ToSrcB) {
            // CB feeds srca; reconfig srca to CB's format.
            reconfig_data_format_srca(CB);
        } else {
            reconfig_data_format_srcb(CB);
        }
    }
    binary_dest_reuse_tiles_init<detail::to_elt_type<OpType>(),
                                 detail::to_reuse_type<Reuse>()>(CB);
}

template <uint32_t CB, BinaryOpType OpType, DestReuseType Reuse, Dst Slot,
          DestReuseInputPolicy Policy, DestReuseReconfig Reconfig>
ALWI void DestReuseOp<CB, OpType, Reuse, Slot, Policy, Reconfig>::exec(uint32_t offset) const {
    constexpr bool do_wait =
        (Policy == DestReuseInputPolicy::WaitNoPop || Policy == DestReuseInputPolicy::WaitAndPop);
    constexpr bool do_pop =
        (Policy == DestReuseInputPolicy::WaitAndPop || Policy == DestReuseInputPolicy::NoWaitPop);
    if constexpr (do_wait) {
        cb_wait_front(CB, 1);
    }
    binary_dest_reuse_tiles<detail::to_elt_type<OpType>(),
                            detail::to_reuse_type<Reuse>()>(CB, 0, dst_idx + offset);
    if constexpr (do_pop) {
        cb_pop_front(CB, 1);
    }
}

// =============================================================================
// binary_op<...> implementation
// =============================================================================

template <
    BinaryOpType OpType,
    BroadcastDim Bcast,
    BinaryInputPolicy APolicy,
    BinaryInputPolicy BPolicy,
    BinaryOutputPolicy OutPolicy,
    BinaryDataFormatReconfig Reconfig,
    typename PostOp>
ALWI void binary_op(
    uint32_t icb_a,
    uint32_t icb_b,
    uint32_t ocb,
    BinaryInputBlockShape shape,
    PostOp post_op) {
    static_assert(detail::is_eltwise_chain_v<PostOp>,
                  "binary_op PostOp must be an EltwiseChain<...>. Wrap single "
                  "ops in eltwise_chain(op, ...).");

    using detail::a_is_per_tile;
    using detail::a_is_upfront_wait;
    using detail::a_pops_at_end;
    using detail::a_caller_managed_wait;

    const uint32_t Ht = shape.rows;
    const uint32_t Wt = shape.cols;
    const uint32_t total_a = Ht * Wt;

    // B tile count given broadcast.
    constexpr bool b_is_scalar = (Bcast == BroadcastDim::SCALAR);
    constexpr bool b_is_row    = (Bcast == BroadcastDim::ROW);
    constexpr bool b_is_col    = (Bcast == BroadcastDim::COL);
    constexpr bool b_is_none   = (Bcast == BroadcastDim::NONE);

    const uint32_t b_count = b_is_scalar ? 1
                           : b_is_row    ? Wt
                           : b_is_col    ? Ht
                                         : total_a;

    // Same-CB dedup is only meaningful for NONE broadcast.
    const bool same_cb = b_is_none && (icb_a == icb_b);

    // ---- Reconfig + initial binary init ----
    if constexpr (detail::reconfig_input(Reconfig)) {
        reconfig_data_format(icb_a, icb_b);
    }
    if constexpr (detail::reconfig_output(Reconfig)) {
        pack_reconfig_data_format(ocb);
    }
    detail::binary_short_init<OpType, Bcast>(icb_a, icb_b);

    // ---- A wait classifications ----
    constexpr bool a_per_tile_wait = a_is_per_tile(APolicy);
    constexpr bool a_upfront_w     = a_is_upfront_wait(APolicy);

    // ---- B wait classification ----
    // ROW + SCALAR force upfront wait unless caller-managed.
    constexpr bool b_force_upfront = (b_is_row || b_is_scalar);
    constexpr bool b_caller_mgd    = a_caller_managed_wait(BPolicy);
    constexpr bool b_per_tile_wait =
        !b_force_upfront && !b_caller_mgd && a_is_per_tile(BPolicy);
    constexpr bool b_upfront_w =
        b_caller_mgd ? false
                     : (b_force_upfront ? true : a_is_upfront_wait(BPolicy));

    // ---- Upfront waits ----
    if constexpr (a_upfront_w) cb_wait_front(icb_a, total_a);
    if constexpr (b_upfront_w) {
        // Skip if same_cb already waited on icb_a above.
        if (!(same_cb && a_upfront_w)) {
            cb_wait_front(icb_b, b_count);
        }
    }

    // ---- Output bulk reserve ----
    if constexpr (OutPolicy == BinaryOutputPolicy::Bulk) {
        cb_reserve_back(ocb, total_a);
    }

    // Re-init binary's MOP every iteration if the PostOp chain ran ANY FPU-
    // clashing element — CopyTile (programs unpack MOP via
    // copy_tile_to_dst_init_short), DestReuseOp (reprograms FPU init), or any
    // other element that opted into clashes_with_fpu. The chain's own reinit
    // trait excludes CopyTile, but binary_op's outer loop must include it.
    constexpr bool needs_clash_reinit =
        chain_has_any_copy_tile_v<PostOp> || chain_has_non_copy_tile_fpu_clash_v<PostOp>;
    constexpr bool postop_is_empty = std::is_same_v<PostOp, EltwiseChain<>>;

    uint32_t bulk_pack_idx = 0;

    for (uint32_t ht = 0; ht < Ht; ++ht) {
        // COL streaming: wait B once per row.
        if constexpr (b_is_col && a_is_per_tile(BPolicy) && !b_caller_mgd) {
            cb_wait_front(icb_b, 1);
        }

        for (uint32_t wt = 0; wt < Wt; ++wt) {
            // PostOp clobbered binary's unpack MOP last iter — reinit before exec.
            if (needs_clash_reinit && (wt > 0 || ht > 0)) {
                detail::binary_short_init<OpType, Bcast>(icb_a, icb_b);
            }

            // Per-tile waits.
            if constexpr (a_per_tile_wait) cb_wait_front(icb_a, 1);
            if constexpr (b_per_tile_wait) {
                if (!same_cb) cb_wait_front(icb_b, 1);
            }

            // Tile indices.
            uint32_t tile_a, tile_b;
            if constexpr (a_per_tile_wait) {
                tile_a = 0;
            } else {
                tile_a = ht * Wt + wt;
            }
            if constexpr (b_is_scalar) {
                tile_b = 0;
            } else if constexpr (b_is_row) {
                tile_b = wt;
            } else if constexpr (b_is_col) {
                if constexpr (a_is_per_tile(BPolicy)) {
                    tile_b = 0;       // per-row wait at offset 0
                } else {
                    tile_b = ht;      // upfront — index by row
                }
            } else {  // NONE
                if (same_cb) {
                    tile_b = tile_a;
                } else if constexpr (b_per_tile_wait) {
                    tile_b = 0;
                } else {
                    tile_b = ht * Wt + wt;
                }
            }

            tile_regs_acquire();
            detail::binary_exec_one<OpType, Bcast>(icb_a, icb_b, tile_a, tile_b, 0);
            if constexpr (!postop_is_empty) {
                post_op.apply(0);
            }
            tile_regs_commit();
            tile_regs_wait();

            if constexpr (OutPolicy == BinaryOutputPolicy::PerTile) {
                cb_reserve_back(ocb, 1);
                pack_tile(0, ocb);
                cb_push_back(ocb, 1);
            } else {
                pack_tile(0, ocb, bulk_pack_idx);
                bulk_pack_idx++;
            }
            tile_regs_release();

            // Per-tile pops.
            if constexpr (a_per_tile_wait) cb_pop_front(icb_a, 1);
            if constexpr (b_per_tile_wait) {
                if (!same_cb) cb_pop_front(icb_b, 1);
            }
        }

        // COL streaming: pop B at end of row.
        if constexpr (b_is_col && a_is_per_tile(BPolicy) && !b_caller_mgd) {
            cb_pop_front(icb_b, 1);
        }
    }

    // ---- Bulk push ----
    if constexpr (OutPolicy == BinaryOutputPolicy::Bulk) {
        cb_push_back(ocb, total_a);
    }

    // ---- At-end pops ----
    if constexpr (a_pops_at_end(APolicy)) cb_pop_front(icb_a, total_a);
    if constexpr (a_pops_at_end(BPolicy)) {
        if (!(same_cb && a_pops_at_end(APolicy))) {
            cb_pop_front(icb_b, b_count);
        }
    }

    // ROW / SCALAR: pop the upfront-waited B unless policy already handled it.
    if constexpr (b_force_upfront && !b_caller_mgd && !a_pops_at_end(BPolicy)
                  && BPolicy != BinaryInputPolicy::WaitUpfrontNoPop) {
        if (!(same_cb && a_pops_at_end(APolicy))) {
            cb_pop_front(icb_b, b_count);
        }
    }
}

// =============================================================================
// Convenience aliases
// =============================================================================

template <BroadcastDim Bcast, BinaryInputPolicy APolicy, BinaryInputPolicy BPolicy,
          BinaryOutputPolicy OutPolicy, BinaryDataFormatReconfig Reconfig,
          typename PostOp>
ALWI void add(uint32_t icb_a, uint32_t icb_b, uint32_t ocb,
              BinaryInputBlockShape shape, PostOp post_op) {
    binary_op<BinaryOpType::ADD, Bcast, APolicy, BPolicy, OutPolicy, Reconfig, PostOp>(
        icb_a, icb_b, ocb, shape, post_op);
}

template <BroadcastDim Bcast, BinaryInputPolicy APolicy, BinaryInputPolicy BPolicy,
          BinaryOutputPolicy OutPolicy, BinaryDataFormatReconfig Reconfig,
          typename PostOp>
ALWI void sub(uint32_t icb_a, uint32_t icb_b, uint32_t ocb,
              BinaryInputBlockShape shape, PostOp post_op) {
    binary_op<BinaryOpType::SUB, Bcast, APolicy, BPolicy, OutPolicy, Reconfig, PostOp>(
        icb_a, icb_b, ocb, shape, post_op);
}

template <BroadcastDim Bcast, BinaryInputPolicy APolicy, BinaryInputPolicy BPolicy,
          BinaryOutputPolicy OutPolicy, BinaryDataFormatReconfig Reconfig,
          typename PostOp>
ALWI void mul(uint32_t icb_a, uint32_t icb_b, uint32_t ocb,
              BinaryInputBlockShape shape, PostOp post_op) {
    binary_op<BinaryOpType::MUL, Bcast, APolicy, BPolicy, OutPolicy, Reconfig, PostOp>(
        icb_a, icb_b, ocb, shape, post_op);
}

template <BinaryInputPolicy Policy, BinaryOutputPolicy OutPolicy,
          BinaryDataFormatReconfig Reconfig, typename PostOp>
ALWI void square(uint32_t icb, uint32_t ocb,
                 BinaryInputBlockShape shape, PostOp post_op) {
    binary_op<BinaryOpType::MUL, BroadcastDim::NONE, Policy, Policy, OutPolicy, Reconfig, PostOp>(
        icb, icb, ocb, shape, post_op);
}

// =============================================================================
// SFPU binary chain elements
// =============================================================================

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::init() const { add_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const {
    add_binary_tile(a, b, c);
}

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::init() const { sub_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const {
    sub_binary_tile(a, b, c);
}

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::init() const { mul_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const {
    mul_binary_tile(a, b, c);
}

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::init() const { div_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const {
    div_binary_tile(a, b, c);
}

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMax<In0, In1, Out>::init() const { binary_max_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMax<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const {
    binary_max_tile(a, b, c);
}

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMin<In0, In1, Out>::init() const { binary_min_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMin<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const {
    binary_min_tile(a, b, c);
}

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::init() const { eq_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const {
    eq_binary_tile(a, b, c);
}

}  // namespace compute_kernel_lib::eltwise
