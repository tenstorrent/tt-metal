// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "api/compute/eltwise_binary.h"

/**
 * @file eltwise_helpers.hpp
 * @brief Unified eltwise operation helper for compute kernels.
 *
 * Provides eltwise_op() — a single pipeline that handles SFPU-only, FPU-only, and
 * FPU+SFPU chains with CB lifecycle management, optional broadcast, amortized packing,
 * and pack-side data format reconfiguration.
 *
 * FPU ops (FpuAdd, FpuSub, FpuMul) and SFPU ops (Exp, Relu, …) are equal citizens:
 * both are chain elements with init()/exec() methods, composable via sfpu_chain().
 *
 * ## Usage
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"
 *   using namespace compute_kernel_lib;
 *
 *   // SFPU-only: exp
 *   auto chain = sfpu_chain(Load<cb_in, Dst::D0>{}, Exp<>{});
 *   eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
 *
 *   // FPU-only: add
 *   auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0>{});
 *   eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
 *
 *   // FPU + SFPU: add then relu
 *   auto chain = sfpu_chain(FpuAdd<cb_in0, cb_in1, Dst::D0>{}, Relu<>{});
 *   eltwise_op<cb_out>(chain, EltwiseTileShape::flat(num_tiles));
 *
 *   // ROW broadcast: A[Ht,Wt] + B[1,Wt]
 *   auto chain = sfpu_chain(FpuAdd<cb_a, cb_b, Dst::D0, BroadcastDim::ROW>{});
 *   eltwise_op<cb_out>(chain, EltwiseTileShape::of(Ht, Wt));
 *
 *   // Multi-exec: init fires once per eltwise_op call (cheap, idempotent for hardware regs)
 *   for (uint32_t block = 0; block < num_blocks; ++block) {
 *       eltwise_op<cb_out>(chain, EltwiseTileShape::flat(tiles_per_block));
 *   }
 *
 * ## Chain element contract
 *
 * - init()               — called once before the tile loop (hoisted)
 * - exec()               — zero-arg: for Load and SFPU ops
 * - exec(ht, wt, cols)   — three-arg: for FpuOp (broadcast-aware tile indexing)
 * - wait_b_upfront(shape) / pop_b_upfront(shape) — for FpuOp B-input lifecycle
 *
 * ## Design notes
 *
 * - EltwiseOutputPolicy::Bulk (default): cb_reserve_back(N) once before loop, cb_push_back(N) after
 * - EltwiseOutputPolicy::PerTile: reserve/push one tile at a time (streaming, migration compatibility)
 * - pack_slot is a compile-time Dst; static_assert validates it against Chain::stride
 * - shape.rows * shape.cols = total output tiles; for non-broadcast use EltwiseTileShape::flat(N)
 */

namespace compute_kernel_lib {

// =============================================================================
// EltwiseTileShape — replaces plain num_tiles; supports 2D broadcast indexing
// =============================================================================

struct EltwiseTileShape {
    uint32_t rows;
    uint32_t cols;

    static constexpr EltwiseTileShape of(uint32_t r, uint32_t c) { return {r, c}; }
    // Flat (1D) shape: no broadcast; 2D loop degenerates to a single row
    static constexpr EltwiseTileShape flat(uint32_t n) { return {1, n}; }
};

// =============================================================================
// Policy Enums
// =============================================================================

enum class EltwiseOutputPolicy {
    PerTile,  // reserve-1 / push-1 per tile — streaming, migration compatibility
    Bulk,     // reserve-N upfront / push-N at end — amortized (default)
};

// =============================================================================
// Internal helpers
// =============================================================================

namespace detail {

template <BroadcastDim Bcast>
ALWI constexpr uint32_t b_tile_idx(uint32_t ht, uint32_t wt, uint32_t cols) {
    if constexpr (Bcast == BroadcastDim::ROW) {
        return wt;
    }
    if constexpr (Bcast == BroadcastDim::COL) {
        return ht;
    }
    if constexpr (Bcast == BroadcastDim::SCALAR) {
        return 0;
    }
    return ht * cols + wt;  // NONE
}

template <BroadcastDim Bcast>
ALWI constexpr uint32_t b_tile_count(EltwiseTileShape shape) {
    if constexpr (Bcast == BroadcastDim::ROW) {
        return shape.cols;
    }
    if constexpr (Bcast == BroadcastDim::COL) {
        return shape.rows;
    }
    if constexpr (Bcast == BroadcastDim::SCALAR) {
        return 1;
    }
    return shape.rows * shape.cols;  // NONE
}

// Default B input policy derived from broadcast dimension (matches observed kernel patterns)
template <BroadcastDim Bcast>
inline constexpr BinaryInputPolicy default_policy_b_v =
    (Bcast == BroadcastDim::ROW)      ? BinaryInputPolicy::WaitUpfrontNoPop
    : (Bcast == BroadcastDim::COL)    ? BinaryInputPolicy::WaitUpfrontPopAtEnd
    : (Bcast == BroadcastDim::SCALAR) ? BinaryInputPolicy::WaitUpfrontPopAtEnd
                                      : BinaryInputPolicy::WaitAndPopPerTile;  // NONE

// True if chain contains any Load element (needs copy_tile_to_dst_init_short)
template <typename Chain>
struct chain_has_load : std::false_type {};
template <typename... Ops>
struct chain_has_load<SfpuChain<Ops...>> : std::bool_constant<(is_load_op_v<Ops> || ...)> {};
template <typename Chain>
inline constexpr bool chain_has_load_v = chain_has_load<Chain>::value;

// SFINAE: does T have exec(uint32_t, uint32_t, uint32_t)?  (FpuOp signature)
template <typename T, typename = void>
struct has_fpu_exec : std::false_type {};
template <typename T>
struct has_fpu_exec<T, std::void_t<decltype(std::declval<const T>().exec(0u, 0u, 0u))>> : std::true_type {};
template <typename T>
inline constexpr bool has_fpu_exec_v = has_fpu_exec<T>::value;

// SFINAE: does T have wait_b_upfront(EltwiseTileShape)?
template <typename T, typename = void>
struct has_wait_b : std::false_type {};
template <typename T>
struct has_wait_b<T, std::void_t<decltype(std::declval<const T>().wait_b_upfront(EltwiseTileShape{}))>>
    : std::true_type {};
template <typename T>
inline constexpr bool has_wait_b_v = has_wait_b<T>::value;

// SFINAE: does T have pop_b_upfront(EltwiseTileShape)?
template <typename T, typename = void>
struct has_pop_b : std::false_type {};
template <typename T>
struct has_pop_b<T, std::void_t<decltype(std::declval<const T>().pop_b_upfront(EltwiseTileShape{}))>> : std::true_type {
};
template <typename T>
inline constexpr bool has_pop_b_v = has_pop_b<T>::value;

// True if chain contains a DestReuseOp (has clashes_with_fpu and is not a LoadTag).
// Signals eltwise_op to re-call copy_tile_to_dst_init_short per tile.
template <typename T, typename = void>
struct has_dest_reuse_flag : std::false_type {};
template <typename T>
struct has_dest_reuse_flag<T, std::void_t<decltype(T::clashes_with_fpu)>>
    : std::bool_constant<T::clashes_with_fpu && !is_load_op_v<T>> {};
template <typename T>
inline constexpr bool has_dest_reuse_flag_v = has_dest_reuse_flag<T>::value;

template <typename Chain>
struct chain_has_dest_reuse : std::false_type {};
template <typename... Ops>
struct chain_has_dest_reuse<SfpuChain<Ops...>> : std::bool_constant<(has_dest_reuse_flag_v<Ops> || ...)> {};
template <typename Chain>
inline constexpr bool chain_has_dest_reuse_v = chain_has_dest_reuse<Chain>::value;

}  // namespace detail

// =============================================================================
// Chain traversal free functions — operate on SfpuChain without modifying it
// =============================================================================

// Base cases (empty chain)
ALWI void chain_init_all(const SfpuChain<>&) {}
ALWI void chain_exec_eltwise(const SfpuChain<>&, uint32_t, uint32_t, uint32_t) {}
ALWI void chain_wait_b_upfront(const SfpuChain<>&, EltwiseTileShape) {}
ALWI void chain_pop_b_upfront(const SfpuChain<>&, EltwiseTileShape) {}

// Recursive cases
template <typename First, typename... Rest>
ALWI void chain_init_all(const SfpuChain<First, Rest...>& chain) {
    chain.first.init();
    chain_init_all(chain.rest);
}

template <typename First, typename... Rest>
ALWI void chain_exec_eltwise(const SfpuChain<First, Rest...>& chain, uint32_t ht, uint32_t wt, uint32_t cols) {
    if constexpr (detail::has_fpu_exec_v<First>) {
        chain.first.exec(ht, wt, cols);
    } else {
        chain.first.exec();
    }
    chain_exec_eltwise(chain.rest, ht, wt, cols);
}

template <typename First, typename... Rest>
ALWI void chain_wait_b_upfront(const SfpuChain<First, Rest...>& chain, EltwiseTileShape shape) {
    if constexpr (detail::has_wait_b_v<First>) {
        chain.first.wait_b_upfront(shape);
    }
    chain_wait_b_upfront(chain.rest, shape);
}

template <typename First, typename... Rest>
ALWI void chain_pop_b_upfront(const SfpuChain<First, Rest...>& chain, EltwiseTileShape shape) {
    if constexpr (detail::has_pop_b_v<First>) {
        chain.first.pop_b_upfront(shape);
    }
    chain_pop_b_upfront(chain.rest, shape);
}

// =============================================================================
// DestReuseOp — FPU chain element that operates on DEST + one CB operand
// =============================================================================

enum class DestReuseInputPolicy {
    WaitAndPop,  // standalone: wait 1, binary_dest_reuse, pop 1 per tile
    NoWaitPop,   // paired with Load<WaitNoPop>: no wait, binary_dest_reuse, pop 1 per tile
};

/**
 * @brief FPU binary chain element: old_DEST[Slot] OP Cb[tile] → DEST[Slot].
 *
 * Computes one of:
 *   DEST_TO_SRCA: old_DEST OP cb_tile  (e.g. ELWMUL+SRCA → old_DEST * cb_tile)
 *   DEST_TO_SRCB: cb_tile OP old_DEST  (e.g. ELWSUB+SRCB → cb_tile - old_DEST)
 *
 * Must follow a prior operation that leaves a value in DEST[Slot].
 *
 * init() is a no-op — binary_dest_reuse_tiles_init is called per-tile inside exec()
 * because it clobbers the copy_tile MOP used by preceding Load elements.  When this
 * element is present, eltwise_op re-calls copy_tile_to_dst_init_short each tile.
 *
 * Typical use (tanhshrink, hardswish):
 *   sfpu_chain(
 *       Load<cb_x, Dst::D0, LoadPolicy::WaitNoPop>{},    // wait, copy, NO pop
 *       Tanh<Dst::D0>{},                                  // D0 = tanh(x)
 *       DestReuseOp<cb_x, EltwiseBinaryType::ELWSUB,
 *                   EltwiseBinaryReuseDestType::DEST_TO_SRCB,
 *                   Dst::D0, DestReuseInputPolicy::NoWaitPop>{})
 *   // D0 = cb_x[0] - tanh(x) = x - tanh(x); cb_x tile popped by DestReuseOp
 */
template <
    uint32_t Cb,
    EltwiseBinaryType BinOp = EltwiseBinaryType::ELWMUL,
    EltwiseBinaryReuseDestType ReuseType = EltwiseBinaryReuseDestType::DEST_TO_SRCA,
    Dst Slot = Dst::D0,
    DestReuseInputPolicy Policy = DestReuseInputPolicy::WaitAndPop>
struct DestReuseOp {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    // Signals to eltwise_op that this element clobbers the copy_tile unpack MOP,
    // requiring per-tile re-init of copy_tile_to_dst_init_short.
    static constexpr bool clashes_with_fpu = true;

    // No-op: real init happens inside exec() per tile to avoid MOP clobbering.
    ALWI void init() const {}

    ALWI void exec(uint32_t /*ht*/, uint32_t /*wt*/, uint32_t /*cols*/) const {
        if constexpr (Policy == DestReuseInputPolicy::WaitAndPop) {
            cb_wait_front(Cb, 1);
        }
        binary_dest_reuse_tiles_init<BinOp, ReuseType>(Cb);
        binary_dest_reuse_tiles<BinOp, ReuseType>(Cb, 0, dst_idx);
        cb_pop_front(Cb, 1);
    }
};

// =============================================================================
// FpuBinaryOp — CRTP base for FPU binary chain elements (FpuAdd, FpuSub, FpuMul)
// =============================================================================

/**
 * @brief CRTP base for FPU binary chain elements.
 *
 * Derived must define:
 *   void do_init()                                    — calls binary_init<op,bcast>(CbIn0, CbIn1)
 *   void do_exec(uint32_t tile_a, uint32_t tile_b, uint32_t dst) — calls binary_exec<op,bcast>(...)
 *
 * Invalid combinations caught at compile time via static_assert:
 *   ROW/COL/SCALAR broadcast with WaitAndPopPerTile PolicyB — B tile count != 1 per tile.
 *
 * @tparam PolicyB  Defaults to default_policy_b_v<Bcast> (upfront for broadcast, per-tile for NONE).
 * @tparam PolicyA  Always WaitAndPopPerTile by default; override for preloaded A inputs.
 */
template <
    typename Derived,
    uint32_t CbIn0,
    uint32_t CbIn1,
    Dst Out,
    BroadcastDim Bcast,
    BinaryInputPolicy PolicyA,
    BinaryInputPolicy PolicyB>
struct FpuBinaryOp {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Out);
    static constexpr uint32_t max_dst = dst_idx;

    static_assert(
        Bcast == BroadcastDim::NONE || PolicyB != BinaryInputPolicy::WaitAndPopPerTile,
        "ROW/COL/SCALAR broadcast: B has fewer tiles than A. "
        "Use WaitUpfrontNoPop (ROW) or WaitUpfrontPopAtEnd (COL/SCALAR) for PolicyB.");

    // Called once before the tile loop by chain_init_all
    ALWI void init() const { static_cast<const Derived*>(this)->do_init(); }

    // Per-tile execution: handles A wait/pop + binary compute
    ALWI void exec(uint32_t ht, uint32_t wt, uint32_t cols) const {
        const uint32_t tile_a = ht * cols + wt;
        const uint32_t tile_b = detail::b_tile_idx<Bcast>(ht, wt, cols);

        if constexpr (PolicyA == BinaryInputPolicy::WaitAndPopPerTile) {
            cb_wait_front(CbIn0, 1);
        }
        static_cast<const Derived*>(this)->do_exec(tile_a, tile_b, dst_idx);
        if constexpr (PolicyA == BinaryInputPolicy::WaitAndPopPerTile) {
            cb_pop_front(CbIn0, 1);
        }
    }

    // B upfront: called by chain_wait/pop_b_upfront before/after tile loop
    ALWI void wait_b_upfront(EltwiseTileShape shape) const {
        if constexpr (
            PolicyB == BinaryInputPolicy::WaitUpfrontNoPop || PolicyB == BinaryInputPolicy::WaitUpfrontPopAtEnd) {
            cb_wait_front(CbIn1, detail::b_tile_count<Bcast>(shape));
        }
    }
    ALWI void pop_b_upfront(EltwiseTileShape shape) const {
        if constexpr (PolicyB == BinaryInputPolicy::WaitUpfrontPopAtEnd) {
            cb_pop_front(CbIn1, detail::b_tile_count<Bcast>(shape));
        }
    }
};

// =============================================================================
// FpuAdd, FpuSub, FpuMul — concrete FPU binary chain elements
// =============================================================================

template <
    uint32_t CbIn0,
    uint32_t CbIn1,
    Dst Out = Dst::D0,
    BroadcastDim Bcast = BroadcastDim::NONE,
    BinaryInputPolicy PolicyA = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy PolicyB = detail::default_policy_b_v<Bcast>>
struct FpuAdd
    : FpuBinaryOp<FpuAdd<CbIn0, CbIn1, Out, Bcast, PolicyA, PolicyB>, CbIn0, CbIn1, Out, Bcast, PolicyA, PolicyB> {
    ALWI void do_init() const { binary_init<BinaryOpType::ADD, Bcast>(CbIn0, CbIn1); }
    ALWI void do_exec(uint32_t ta, uint32_t tb, uint32_t d) const {
        binary_exec<BinaryOpType::ADD, Bcast>(CbIn0, CbIn1, ta, tb, d);
    }
};

template <
    uint32_t CbIn0,
    uint32_t CbIn1,
    Dst Out = Dst::D0,
    BroadcastDim Bcast = BroadcastDim::NONE,
    BinaryInputPolicy PolicyA = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy PolicyB = detail::default_policy_b_v<Bcast>>
struct FpuSub
    : FpuBinaryOp<FpuSub<CbIn0, CbIn1, Out, Bcast, PolicyA, PolicyB>, CbIn0, CbIn1, Out, Bcast, PolicyA, PolicyB> {
    ALWI void do_init() const { binary_init<BinaryOpType::SUB, Bcast>(CbIn0, CbIn1); }
    ALWI void do_exec(uint32_t ta, uint32_t tb, uint32_t d) const {
        binary_exec<BinaryOpType::SUB, Bcast>(CbIn0, CbIn1, ta, tb, d);
    }
};

template <
    uint32_t CbIn0,
    uint32_t CbIn1,
    Dst Out = Dst::D0,
    BroadcastDim Bcast = BroadcastDim::NONE,
    BinaryInputPolicy PolicyA = BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy PolicyB = detail::default_policy_b_v<Bcast>>
struct FpuMul
    : FpuBinaryOp<FpuMul<CbIn0, CbIn1, Out, Bcast, PolicyA, PolicyB>, CbIn0, CbIn1, Out, Bcast, PolicyA, PolicyB> {
    ALWI void do_init() const { binary_init<BinaryOpType::MUL, Bcast>(CbIn0, CbIn1); }
    ALWI void do_exec(uint32_t ta, uint32_t tb, uint32_t d) const {
        binary_exec<BinaryOpType::MUL, Bcast>(CbIn0, CbIn1, ta, tb, d);
    }
};

// =============================================================================
// eltwise_op — primary entry point
// =============================================================================

/**
 * @brief Unified eltwise pipeline: handles SFPU-only, FPU-only, and FPU+SFPU chains.
 *
 * Contract:
 *  - chain_init_all(chain) is called once before the tile loop
 *  - B upfront waits called before loop; B upfront pops called after loop
 *  - Per tile: chain_exec_eltwise(chain, ht, wt, cols)
 *  - pack_slot is validated against Chain::stride at compile time
 *  - shape = EltwiseTileShape::flat(N) for SFPU/non-broadcast; ::of(Ht, Wt) for broadcast
 *
 * @tparam cb_out          Output circular buffer index
 * @tparam pack_slot       DEST slot to pack from (D0 for all standard linear chains)
 * @tparam output_policy   PerTile or Bulk (default Bulk — amortized reserve/push)
 * @tparam Chain           SfpuChain<...> type (deduced)
 */
template <
    uint32_t cb_out,
    Dst pack_slot = Dst::D0,
    EltwiseOutputPolicy output_policy = EltwiseOutputPolicy::Bulk,
    typename Chain>
ALWI void eltwise_op(Chain chain, EltwiseTileShape shape);

}  // namespace compute_kernel_lib

#include "eltwise_helpers.inl"
