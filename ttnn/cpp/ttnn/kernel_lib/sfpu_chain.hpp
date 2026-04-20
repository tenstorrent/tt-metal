// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include "api/compute/common_globals.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/compute_kernel_api.h"
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

/**
 * @file sfpu_chain.hpp
 * @brief Chain/pipeline infrastructure: Dst, policy enums, CRTP bases, Load/CompactLoad,
 *        SfpuChain, sfpu_chain(), sfpu_pipeline(), and sfpu_op().
 *
 * This is the minimal chain-only header. It does not pull in any op-specific LLK headers
 * or op struct declarations. Include sfpu_helpers.hpp for the full set of op structs
 * and named convenience aliases.
 *
 * Typical usage with custom op types:
 *   #include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
 *   // define your own op structs inheriting from UnaryOp/BinaryOp/TernaryOp
 *   // then use sfpu_chain() and sfpu_pipeline() as normal
 */

namespace compute_kernel_lib {

using namespace ckernel;

// =============================================================================
// DEST Slot Enum
// =============================================================================

/**
 * @brief Self-documenting DEST register slot indices
 *
 * Used as compile-time template parameters for op structs and Load.
 * Maximum 8 slots (hardware limit in half-sync fp16 mode).
 */
enum class Dst : uint32_t { D0 = 0, D1 = 1, D2 = 2, D3 = 3, D4 = 4, D5 = 5, D6 = 6, D7 = 7 };

// =============================================================================
// Approximation Mode Enums (self-documenting template params)
// =============================================================================

/**
 * @brief Approximation mode for SFPU operations
 *
 * Controls precision vs speed tradeoff in operations like exp, log, tanh, sigmoid.
 * - Exact: Full precision (default for most ops)
 * - Approx: Reduced precision, faster execution
 */
enum class Approx : bool { Exact = false, Fast = true };

/**
 * @brief Legacy compatibility mode for recip/rsqrt
 *
 * - Off: Use new optimized implementation (default for rsqrt)
 * - On: Use legacy implementation (default for recip)
 */
enum class Legacy : bool { Off = false, On = true };

// =============================================================================
// Tag Types for Compile-Time Dispatch
// =============================================================================

/** @brief Base tag for Load ops — pipeline handles these specially */
struct LoadTag {};

/** @brief Compile-time predicate: true if T is a Load op */
template <typename T>
constexpr bool is_load_op_v = std::is_base_of_v<LoadTag, T>;

// =============================================================================
// Policy Enums
// =============================================================================

/**
 * @brief Input synchronization policy for SFPU pipeline
 *
 * Controls how Load ops wait for and consume input tiles:
 * - WaitAndPopPerTile: Wait/pop one tile per Load (streaming, default)
 * - WaitUpfrontNoPop: Wait for all tiles upfront, don't pop (persistent reuse)
 * - NoWaitNoPop: Caller manages wait/pop externally
 */
enum class SfpuInputPolicy { WaitAndPopPerTile, WaitUpfrontNoPop, NoWaitNoPop };

/**
 * @brief Output synchronization policy for SFPU pipeline
 *
 * Controls when to reserve/push output tiles:
 * - PerTile: Reserve/push one tile at a time (default, streaming)
 * - Bulk: Reserve all upfront, push all at end (block output)
 */
enum class SfpuOutputPolicy { PerTile, Bulk };

/**
 * @brief Data format reconfiguration mode for SFPU pipeline
 *
 * Controls whether unpacker/packer are reconfigured before the pipeline runs:
 * - NONE: Skip all reconfiguration
 * - INPUT: Reconfigure unpacker only
 * - OUTPUT: Reconfigure packer only
 * - INPUT_AND_OUTPUT: Reconfigure both (default, safest)
 */
enum class SfpuDataFormatReconfig { NONE = 0, INPUT = 1, OUTPUT = 2, INPUT_AND_OUTPUT = 3 };

/**
 * @brief Controls DEST batching behavior in sfpu_pipeline
 *
 * When Auto, the pipeline automatically fills DEST with as many chain iterations
 * as possible (DEST_AUTO_LIMIT / chain_stride), calling each op's init() once
 * and exec() multiple times. This amortizes init overhead.
 *
 * - Auto: batch_size = DEST_AUTO_LIMIT / chain_stride (maximizes throughput)
 * - Disabled: batch_size = 1 (original per-tile behavior, init+exec each tile)
 */
enum class SfpuBatching { Auto, Disabled };

// =============================================================================
// CRTP Base Templates — eliminate per-op boilerplate
// =============================================================================

/**
 * @brief CRTP base for unary SFPU ops (single DEST slot)
 *
 * Provides: dst_idx, max_dst, static_assert, exec(), apply().
 * Derived must define: init(), call(uint32_t d0).
 * call() receives the already-offset-adjusted slot index.
 */
template <typename Derived, Dst Slot>
struct UnaryOp {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void exec(uint32_t offset = 0) const { static_cast<const Derived*>(this)->call(dst_idx + offset); }
    ALWI void apply(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->init();
        exec(offset);
    }
};

/**
 * @brief CRTP base for binary SFPU ops (two input slots + one output slot)
 *
 * Provides: in0, in1, out, max_dst, static_asserts, exec(), apply().
 * Derived must define: init(), call(uint32_t a, uint32_t b, uint32_t c).
 */
template <typename Derived, Dst In0, Dst In1, Dst Out>
struct BinaryOp {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static constexpr uint32_t max_dst = (in0 > in1) ? ((in0 > out) ? in0 : out) : ((in1 > out) ? in1 : out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void exec(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->call(in0 + offset, in1 + offset, out + offset);
    }
    ALWI void apply(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->init();
        exec(offset);
    }
};

/**
 * @brief CRTP base for ternary SFPU ops (three input slots + one output slot)
 *
 * Provides: in0, in1, in2, out, max_dst, static_asserts, exec(), apply().
 * Derived must define: init(), call(uint32_t a, uint32_t b, uint32_t c, uint32_t d).
 */
template <typename Derived, Dst In0, Dst In1, Dst In2, Dst Out>
struct TernaryOp {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t in2 = static_cast<uint32_t>(In2);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static constexpr uint32_t max_dst = (in0 > in1)
                                            ? ((in0 > in2) ? ((in0 > out) ? in0 : out) : ((in2 > out) ? in2 : out))
                                            : ((in1 > in2) ? ((in1 > out) ? in1 : out) : ((in2 > out) ? in2 : out));
    static_assert(in0 < 8 && in1 < 8 && in2 < 8 && out < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void exec(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->call(in0 + offset, in1 + offset, in2 + offset, out + offset);
    }
    ALWI void apply(uint32_t offset = 0) const {
        static_cast<const Derived*>(this)->init();
        exec(offset);
    }
};

// =============================================================================
// Compile-time Helpers
// =============================================================================

namespace detail {

/** @brief Variadic compile-time max for uint32_t values */
template <uint32_t First, uint32_t... Rest>
struct CxMax {
    static constexpr uint32_t value = (First > CxMax<Rest...>::value) ? First : CxMax<Rest...>::value;
};
template <uint32_t Only>
struct CxMax<Only> {
    static constexpr uint32_t value = Only;
};
template <uint32_t... Vs>
inline constexpr uint32_t cx_max_v = CxMax<Vs...>::value;

}  // namespace detail

// =============================================================================
// Load Op (user-facing) and CompactLoad (internal, produced by sfpu_chain)
// =============================================================================

/**
 * @brief Per-Load CB lifecycle policy
 *
 * The Load op owns the decision of whether to wait on the CB and whether to pop
 * after copying. The pipeline does NOT override this.
 *
 * - WaitAndPop:  wait for tile, copy, pop (default; streaming input)
 * - WaitNoPop:   wait for tile, copy, don't pop (persistent tile reused across
 *                iterations, e.g. a mask or scaler)
 * - NoWaitNoPop: don't wait, don't pop (caller owns CB lifecycle externally,
 *                e.g. sharded / pre-loaded inputs)
 */
enum class LoadPolicy {
    WaitAndPop,
    WaitNoPop,
    NoWaitNoPop,
};

constexpr bool load_does_wait(LoadPolicy p) { return p == LoadPolicy::WaitAndPop || p == LoadPolicy::WaitNoPop; }
constexpr bool load_does_pop(LoadPolicy p) { return p == LoadPolicy::WaitAndPop; }

/**
 * @brief User-facing Load: copies a tile from CB into DEST[Slot]
 *
 * sfpu_chain() automatically compacts adjacent same-CB Loads into a single
 * CompactLoad element (shared wait + N copies + shared pop), reflecting the
 * fact that the Loads describe the same physical tile fanned out to multiple
 * DEST slots. When Loads in a merged group carry different policies, the
 * group's wait and pop flags are the disjunction ("OR") of the individual
 * Loads — any Load that wants to wait triggers the wait, any Load that wants
 * to pop triggers the pop.
 *
 * @tparam CB      Circular buffer index
 * @tparam Slot    DEST slot receiving the tile
 * @tparam Policy  CB lifecycle policy (see LoadPolicy; default WaitAndPop)
 */
template <uint32_t CB, Dst Slot, LoadPolicy Policy = LoadPolicy::WaitAndPop>
struct Load : LoadTag {
    static constexpr uint32_t cb = CB;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static constexpr LoadPolicy policy = Policy;
    static constexpr bool do_wait = load_does_wait(Policy);
    static constexpr bool do_pop = load_does_pop(Policy);
    static_assert(static_cast<uint32_t>(Slot) < 8, "DEST slot exceeds maximum capacity (8)");
};

/**
 * @brief Compacted Load: multiple DEST slots from the same CB, with wait/pop control
 *
 * Produced by sfpu_chain() compile-time transformation. Adjacent same-CB Loads
 * whose (DoWait, DoPop) flags ALSO match are merged into one CompactLoad;
 * differing flags stay as separate CompactLoad elements. The flags originate
 * from the source `Load<CB, Slot, DoWait, DoPop>` — the pipeline does not
 * override them.
 *
 * @tparam CB      Circular buffer index
 * @tparam DoWait  If true, exec() calls cb_wait_front before copying
 * @tparam DoPop   If true, exec() calls cb_pop_front after copying
 * @tparam Slots   DEST slot indices to copy into
 */
template <uint32_t CB, bool DoWait, bool DoPop, Dst... Slots>
struct CompactLoad : LoadTag {
    static constexpr uint32_t cb = CB;
    static constexpr bool do_wait = DoWait;
    static constexpr bool do_pop = DoPop;
    static constexpr uint32_t max_dst = detail::cx_max_v<static_cast<uint32_t>(Slots)...>;
    static constexpr uint32_t num_slots = sizeof...(Slots);
    static_assert(((static_cast<uint32_t>(Slots) < 8) && ...), "DEST slot exceeds maximum capacity (8)");

    ALWI void init() const;
    ALWI void exec(uint32_t offset = 0) const;
    ALWI void apply(uint32_t offset = 0) const {
        init();
        exec(offset);
    }
};

// =============================================================================
// SfpuChain Combinator
// =============================================================================

/** @brief Compile-time max helper */
template <typename T>
constexpr T cx_max(T a, T b) {
    return (a > b) ? a : b;
}

/**
 * @brief Variadic chain of ops (CompactLoad + compute)
 *
 * After sfpu_chain() transformation, all elements have init()/exec()/apply():
 * - apply(offset): init()+exec(offset) per element, sequentially
 * - apply_batched(num_iters, stride): init() once then exec() num_iters times per element
 * - exec_only(offset): exec(offset) without init, for single-op optimization
 *
 * Zero runtime overhead — all dispatch is resolved at compile time.
 */
template <typename... Ops>
struct SfpuChain;

/** @brief Base case: empty chain */
template <>
struct SfpuChain<> {
    static constexpr uint32_t max_dst = 0;
    static constexpr uint32_t stride = 1;
    static constexpr uint32_t num_compute_ops = 0;

    ALWI void apply(uint32_t = 0) const {}
    ALWI void apply_batched(uint32_t, uint32_t) const {}
    ALWI void exec_only(uint32_t = 0) const {}
};

/** @brief Recursive case: first op + rest of chain */
template <typename First, typename... Rest>
struct SfpuChain<First, Rest...> {
    First first;
    SfpuChain<Rest...> rest;

    static constexpr uint32_t max_dst = cx_max(First::max_dst, SfpuChain<Rest...>::max_dst);
    static constexpr uint32_t stride = max_dst + 1;
    static constexpr uint32_t num_compute_ops = (is_load_op_v<First> ? 0 : 1) + SfpuChain<Rest...>::num_compute_ops;

    constexpr SfpuChain() = default;
    constexpr SfpuChain(First f, Rest... r) : first(f), rest(r...) {}

    /** @brief Execute all elements in sequence: init+exec per element */
    ALWI void apply(uint32_t offset = 0) const {
        first.apply(offset);
        rest.apply(offset);
    }

    /** @brief Batched: init once per element, exec num_iters times */
    ALWI void apply_batched(uint32_t num_iters, uint32_t chain_stride) const {
        first.init();
        for (uint32_t k = 0; k < num_iters; ++k) {
            first.exec(k * chain_stride);
        }
        rest.apply_batched(num_iters, chain_stride);
    }

    /** @brief Exec-only: exec(offset) without init (single-element optimization) */
    ALWI void exec_only(uint32_t offset = 0) const {
        first.exec(offset);
        rest.exec_only(offset);
    }
};

// =============================================================================
// Compile-time Chain Transformation (Load compaction + wait/pop annotation)
// =============================================================================

namespace detail {

// --- Type list helpers ---

template <typename... Ts>
struct TypeList {};

// Build SfpuChain from TypeList
template <typename TL>
struct ChainFromList;
template <typename... Ts>
struct ChainFromList<TypeList<Ts...>> {
    using type = SfpuChain<Ts...>;
};

// Append element to TypeList
template <typename TL, typename T>
struct Append;
template <typename... Ts, typename T>
struct Append<TypeList<Ts...>, T> {
    using type = TypeList<Ts..., T>;
};

// --- Step 1: Compact adjacent same-CB Loads into CompactLoad ---
//
// Rationale: adjacent same-CB Loads in a chain describe ONE physical tile being
// fanned out to multiple DEST slots (e.g. hardswish/mish: load `x` to D0 and D1,
// transform D0, multiply by D1). Merging is always correct on same-CB adjacency
// — the flags describe the TILE's lifecycle, not per-copy behaviour, so they
// are combined with OR when Loads merge: `group.do_wait = OR(load.do_wait)`,
// `group.do_pop = OR(load.do_pop)`. Mixing `Load` and `LoadPersistent` on the
// same tile therefore yields "wait once, don't pop" (the persistent wins on
// pop; wait is triggered by either).

// Phase 1: Check if last element is CompactLoad<CB, ...> (any flags)
template <typename TL, uint32_t CB>
struct LastIsCompactLoadFromCB {
    static constexpr bool value = false;
};
template <uint32_t CB, bool W, bool P, Dst... S>
struct LastIsCompactLoadFromCB<TypeList<CompactLoad<CB, W, P, S...>>, CB> {
    static constexpr bool value = true;
};
template <typename First, typename Second, typename... Rest, uint32_t CB>
struct LastIsCompactLoadFromCB<TypeList<First, Second, Rest...>, CB> {
    static constexpr bool value = LastIsCompactLoadFromCB<TypeList<Second, Rest...>, CB>::value;
};

// Phase 2: Prepend element to TypeList (preserves order when rebuilding)
template <typename T, typename TL>
struct Prepend;
template <typename T, typename... Ts>
struct Prepend<T, TypeList<Ts...>> {
    using type = TypeList<T, Ts...>;
};

// Phase 3: Replace last CompactLoad<CB,...> by appending Slot and OR'ing flags.
// NewW/NewP are the incoming Load's flags; ExistingW/P come from the trailing
// CompactLoad. The merged group inherits the disjunction.
template <typename TL, uint32_t CB, bool NewW, bool NewP, Dst NewSlot>
struct ReplaceLastLoad;
template <uint32_t CB, bool W, bool P, Dst... Slots, bool NewW, bool NewP, Dst NewSlot>
struct ReplaceLastLoad<TypeList<CompactLoad<CB, W, P, Slots...>>, CB, NewW, NewP, NewSlot> {
    using type = TypeList<CompactLoad<CB, (W || NewW), (P || NewP), Slots..., NewSlot>>;
};
template <typename First, typename... Rest, uint32_t CB, bool NewW, bool NewP, Dst NewSlot>
struct ReplaceLastLoad<TypeList<First, Rest...>, CB, NewW, NewP, NewSlot> {
    using type =
        typename Prepend<First, typename ReplaceLastLoad<TypeList<Rest...>, CB, NewW, NewP, NewSlot>::type>::type;
};

// AppendLoad: merge on CB match (any flags), else append new CompactLoad
template <typename TL, uint32_t CB, Dst Slot, bool W, bool P, bool Merge = LastIsCompactLoadFromCB<TL, CB>::value>
struct AppendLoad;

// No merge: first Load for this CB — its flags become the group's flags
template <typename... Elems, uint32_t CB, Dst Slot, bool W, bool P>
struct AppendLoad<TypeList<Elems...>, CB, Slot, W, P, false> {
    using type = TypeList<Elems..., CompactLoad<CB, W, P, Slot>>;
};

// Merge: extend slot list of the trailing same-CB CompactLoad, OR'ing flags
template <typename TL, uint32_t CB, Dst Slot, bool W, bool P>
struct AppendLoad<TL, CB, Slot, W, P, true> {
    using type = typename ReplaceLastLoad<TL, CB, W, P, Slot>::type;
};

// Fold step: dispatch Load vs non-Load via helper
template <typename Acc, typename Elem, bool IsLoad = is_load_op_v<Elem>>
struct CompactStep;

// Load: derive (do_wait, do_pop) from LoadPolicy and thread to AppendLoad
template <typename Acc, uint32_t CB, Dst Slot, LoadPolicy Policy>
struct CompactStep<Acc, Load<CB, Slot, Policy>, true> {
    using type = typename AppendLoad<Acc, CB, Slot, load_does_wait(Policy), load_does_pop(Policy)>::type;
};

// Non-Load: pass through
template <typename... AccElems, typename Elem>
struct CompactStep<TypeList<AccElems...>, Elem, false> {
    using type = TypeList<AccElems..., Elem>;
};

// Fold over all ops to produce compacted TypeList
template <typename Acc, typename... Remaining>
struct CompactFold {
    using type = Acc;
};
template <typename Acc, typename Head, typename... Tail>
struct CompactFold<Acc, Head, Tail...> {
    using type = typename CompactFold<typename CompactStep<Acc, Head>::type, Tail...>::type;
};

// --- Step 2: Check for multi-group same-CB (static_assert) ---

// Check if CB appears in any load element in a TypeList
template <uint32_t CB, typename TL>
struct HasCBInList {
    static constexpr bool value = false;
};
// Non-load: skip
template <uint32_t CB, typename First, typename... Rest>
struct HasCBInList<CB, TypeList<First, Rest...>> {
    static constexpr bool value = HasCBInList<CB, TypeList<Rest...>>::value;
};
// CompactLoad match: check CB
template <uint32_t CB, uint32_t CB2, bool W, bool P, Dst... S, typename... Rest>
struct HasCBInList<CB, TypeList<CompactLoad<CB2, W, P, S...>, Rest...>> {
    static constexpr bool value = (CB == CB2) || HasCBInList<CB, TypeList<Rest...>>::value;
};

// Validate no CB appears in multiple CompactLoad groups
template <typename TL>
struct NoMultiGroupCB {
    static constexpr bool value = true;
};
// Non-load: skip
template <typename First, typename... Rest>
struct NoMultiGroupCB<TypeList<First, Rest...>> {
    static constexpr bool value = NoMultiGroupCB<TypeList<Rest...>>::value;
};
// CompactLoad: check this CB doesn't appear later
template <uint32_t CB, bool W, bool P, Dst... S, typename... Rest>
struct NoMultiGroupCB<TypeList<CompactLoad<CB, W, P, S...>, Rest...>> {
    static constexpr bool value =
        !HasCBInList<CB, TypeList<Rest...>>::value && NoMultiGroupCB<TypeList<Rest...>>::value;
};

// --- Step 3: Wait/pop annotation is a no-op now ---
// Each CompactLoad's (do_wait, do_pop) is inherited directly from its source
// Load<CB, Slot, W, P> type. The pipeline honours those flags at exec() time;
// no post-hoc annotation is required.

}  // namespace detail

// =============================================================================
// sfpu_chain() Factory
// =============================================================================

/**
 * @brief Factory function — compacts adjacent same-CB Loads and returns transformed chain
 *
 * Adjacent Loads on the same CB are merged into one CompactLoad regardless of
 * policy. The merged group's wait/pop behaviour is OR'd across the individual
 * Loads' policies (any Load that wants to wait triggers the wait; any Load
 * that wants to pop triggers the pop).
 *
 * Usage:
 *   // Fan-out: both slots come from tile[0] of cb=0, single wait+pop.
 *   auto c1 = sfpu_chain(Load<0, Dst::D0>{}, Load<0, Dst::D1>{}, Exp<>{});
 *   // → SfpuChain<CompactLoad<0, true, true, D0, D1>, Exp<>>
 *
 *   // Mixed: data streams, mask is persistent across iterations.
 *   auto c2 = sfpu_chain(
 *       Load<cb_data, Dst::D0>{},                                    // WaitAndPop
 *       Load<cb_mask, Dst::D1, LoadPolicy::WaitNoPop>{},             // persistent
 *       Mask<DataFormat::Float16_b>{});
 *   // → SfpuChain<CompactLoad<cb_data, true, true, D0>,
 *   //             CompactLoad<cb_mask, true, false, D1>,
 *   //             Mask<...>>
 */
template <typename... Ops>
constexpr ALWI auto sfpu_chain(Ops...) {
    using Compacted = typename detail::CompactFold<detail::TypeList<>, Ops...>::type;
    static_assert(
        detail::NoMultiGroupCB<Compacted>::value,
        "Same CB appears in multiple non-adjacent Load groups. "
        "Place all Loads from the same CB adjacent in the chain, or use separate CBs.");
    return typename detail::ChainFromList<Compacted>::type{};
}

// =============================================================================
// Pipeline Function Declaration
// =============================================================================

template <
    SfpuBatching batching = SfpuBatching::Auto,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT,
    typename Chain>
ALWI void sfpu_pipeline(Chain chain, uint32_t ocb, uint32_t num_tiles, Dst pack_slot = Dst::D0);

// =============================================================================
// Convenience: Single Unary Op Declaration
// =============================================================================

template <
    uint32_t ICB,
    SfpuBatching batching = SfpuBatching::Auto,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT,
    typename Op>
ALWI void sfpu_op(uint32_t ocb, uint32_t num_tiles, Op op);

}  // namespace compute_kernel_lib

#include "sfpu_chain.inl"
