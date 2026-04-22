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
 * @brief Chain/pipeline infrastructure: Dst, policy enums, CRTP bases, Load,
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
 * - OUTPUT: Reconfigure packer (let Load elements handle input reconfig)
 * - NONE: Skip all reconfiguration
 */
enum class SfpuDataFormatReconfig { NONE = 0, OUTPUT = 2 };

// SfpuBatching removed — pipeline always processes one tile per acquire cycle.
// Will be re-introduced when batching is needed.

// =============================================================================
// CRTP Base Templates — eliminate per-op boilerplate
// =============================================================================

/**
 * @brief CRTP base for unary SFPU ops (single DEST slot)
 *
 * Provides: dst_idx, max_dst, static_assert, exec(), apply().
 * Derived must define: init(), call(uint32_t d0).
 * call() receives the already-offset-adjusted slot index.
 *
 * tile_step is forwarded from sfpu_pipeline for chains that contain a
 * WaitUpfrontPopAtEnd Load; pure compute ops ignore it.
 */
template <typename Derived, Dst Slot>
struct UnaryOp {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void exec() const { static_cast<const Derived*>(this)->call(dst_idx); }
    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        exec();
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
    ALWI void exec() const { static_cast<const Derived*>(this)->call(in0, in1, out); }
    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        exec();
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
    ALWI void exec() const { static_cast<const Derived*>(this)->call(in0, in1, in2, out); }
    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        exec();
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
// Load Op
// =============================================================================

/**
 * @brief Per-Load CB lifecycle policy.
 *
 * The Load op owns the decision of whether to wait on the CB and whether to pop
 * after copying. The pipeline does NOT override per-tile wait/pop.
 *
 * - WaitAndPop:         wait for tile, copy tile 0, pop (streaming input)
 * - WaitNoPop:          wait for tile, copy tile 0, no pop (persistent / fan-out first)
 * - NoWaitPop:          no wait, copy tile 0, pop (fan-out last / pre-waited single)
 * - NoWaitNoPop:        no wait, no pop (caller owns CB lifecycle externally)
 * - WaitUpfrontPopAtEnd: pipeline issues cb_wait_front(CB, num_tiles) once before the
 *                        tile loop and cb_pop_front(CB, num_tiles) once after. Load::exec
 *                        self-advances cb_tile_idx each call (rising index into the
 *                        pre-waited block). Pipeline resets the index at end of call so
 *                        the chain can be reused across blocks.
 *                        Fix for GAP-11 (pre-waited block access).
 *                        Static assert: at most one upfront Load per CB in a chain.
 */
enum class LoadPolicy {
    WaitAndPop,
    WaitNoPop,
    NoWaitPop,
    NoWaitNoPop,
    WaitUpfrontPopAtEnd,
};

constexpr bool load_does_wait(LoadPolicy p) { return p == LoadPolicy::WaitAndPop || p == LoadPolicy::WaitNoPop; }
constexpr bool load_does_pop(LoadPolicy p) { return p == LoadPolicy::WaitAndPop || p == LoadPolicy::NoWaitPop; }
constexpr bool load_is_upfront(LoadPolicy p) { return p == LoadPolicy::WaitUpfrontPopAtEnd; }

/**
 * @brief Data-format reconfig mode for Load.
 *
 * - None:  no reconfig; CB's data format must already match the current unpack state.
 * - Input: reconfig the input SRC before the copy. Load always reconfigures SRCA
 *          because copy_tile goes through unpack A. (Contrast with DestReuseOp,
 *          which picks srca or srcb based on its ReuseType — here there is no
 *          choice: Load is always a single-stream SRCA copy.)
 */
enum class LoadReconfig {
    None,
    Input,
};

/**
 * @brief Copy a tile from CB into DEST[Slot] with the given CB-lifecycle policy.
 *
 * For fan-out (same CB tile to multiple DEST slots), write multiple Loads with
 * explicit policies so the CB is waited/popped exactly once:
 *   Load<cb, Dst::D0, LoadPolicy::WaitNoPop>{},   // wait once
 *   Load<cb, Dst::D1, LoadPolicy::NoWaitPop>{},   // no redundant wait; pop once
 *
 * To index into a pre-waited block of tiles, use WaitUpfrontPopAtEnd — the
 * pipeline bulk-waits N upfront, exec() advances the index (0, 1, 2, ...), and
 * the pipeline bulk-pops N at the end. Do not set the tile index externally;
 * it is internal pipeline state.
 *
 * @tparam CB        Circular buffer index
 * @tparam Slot      DEST slot receiving the tile
 * @tparam Policy    CB lifecycle policy (see LoadPolicy; default WaitAndPop)
 * @tparam Reconfig  Data-format reconfig for srca (default None)
 */
template <uint32_t CB, Dst Slot, LoadPolicy Policy = LoadPolicy::WaitAndPop, LoadReconfig Reconfig = LoadReconfig::None>
struct Load : LoadTag {
    static constexpr bool clashes_with_fpu = true;
    static constexpr uint32_t cb = CB;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static constexpr LoadPolicy policy = Policy;
    static constexpr bool do_wait = load_does_wait(Policy);
    static constexpr bool do_pop = load_does_pop(Policy);
    static constexpr bool is_upfront = load_is_upfront(Policy);
    static constexpr bool do_reconfig = (Reconfig == LoadReconfig::Input);
    static_assert(static_cast<uint32_t>(Slot) < 8, "DEST slot exceeds maximum capacity (8)");

    // init() programs the copy_tile MOP and (optionally) the SRCA data format.
    // The pipeline hoists this across the tile loop when the chain is hoist-safe
    // (see chain_is_hoist_safe_v); otherwise it fires per tile via apply().
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const {
        init();
        exec();
    }

    // Reset internal tile counter back to 0. Called by sfpu_pipeline after all
    // tiles are processed so the chain can be reused across blocks.
    ALWI void reset_tile_idx() const {
        if constexpr (is_upfront) {
            cb_tile_idx_ = 0;
        }
    }

    // Upfront CB lifecycle: pipeline bulk-waits N tiles before the tile loop and
    // bulk-pops N tiles after. Only fires for WaitUpfrontPopAtEnd loads.
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

// =============================================================================
// SfpuChain Combinator
// =============================================================================

/** @brief Compile-time max helper */
template <typename T>
constexpr T cx_max(T a, T b) {
    return (a > b) ? a : b;
}

// -----------------------------------------------------------------------------
// Chain-protocol method detectors
//
// Only Load (and any future upfront-capable op) implements wait_upfront /
// pop_upfront / reset_tile_idx. For every other chain element the call is a
// no-op, so the chain cascade guards each call with `if constexpr` and these
// traits. No per-element boilerplate required.
// -----------------------------------------------------------------------------

namespace detail {

template <typename T, typename = void>
struct has_wait_upfront : std::false_type {};
template <typename T>
struct has_wait_upfront<T, std::void_t<decltype(std::declval<const T&>().wait_upfront(uint32_t{}))>> : std::true_type {
};

template <typename T, typename = void>
struct has_pop_upfront : std::false_type {};
template <typename T>
struct has_pop_upfront<T, std::void_t<decltype(std::declval<const T&>().pop_upfront(uint32_t{}))>> : std::true_type {};

template <typename T, typename = void>
struct has_reset_tile_idx : std::false_type {};
template <typename T>
struct has_reset_tile_idx<T, std::void_t<decltype(std::declval<const T&>().reset_tile_idx())>> : std::true_type {};

}  // namespace detail

/**
 * @brief Variadic chain of ops (Load + compute)
 *
 * apply(offset) calls init() + exec(offset) for each element in order.
 * Zero runtime overhead — all dispatch is resolved at compile time.
 */
template <typename... Ops>
struct SfpuChain;

/** @brief Base case: empty chain */
template <>
struct SfpuChain<> {
    static constexpr uint32_t max_dst = 0;
    static constexpr uint32_t stride = 1;
    ALWI void apply() const {}
    ALWI void apply_no_load_init() const {}
    ALWI void init_any_load() const {}
    ALWI void reset_tile_idx() const {}
    ALWI void wait_upfront(uint32_t) const {}
    ALWI void pop_upfront(uint32_t) const {}
};
// Note: SfpuChain<> keeps the methods so the recursive cascade in the
// non-empty specialization can call them unconditionally on `rest` without
// needing to detect the terminator.

/** @brief Recursive case: first op + rest of chain */
template <typename First, typename... Rest>
struct SfpuChain<First, Rest...> {
    First first;
    SfpuChain<Rest...> rest;

    static constexpr uint32_t max_dst = cx_max(First::max_dst, SfpuChain<Rest...>::max_dst);
    static constexpr uint32_t stride = max_dst + 1;

    constexpr SfpuChain() = default;
    constexpr SfpuChain(First f, Rest... r) : first(f), rest(r...) {}

    ALWI void apply() const {
        first.apply();
        rest.apply();
    }

    // Hoisted-init path: Load elements skip their init() (already done once
    // before the tile loop); every other element runs full apply(). Used by
    // sfpu_pipeline when chain_is_hoist_safe_v<Chain>.
    ALWI void apply_no_load_init() const {
        if constexpr (is_load_op_v<First>) {
            first.exec();
        } else {
            first.apply();
        }
        rest.apply_no_load_init();
    }

    // Call init() on the first Load found (walking the chain). Used to perform
    // the hoisted copy_tile MOP/reconfig setup once before the tile loop. Any
    // Load's init programs the same state when all Loads share a CB, which is
    // the precondition for the hoist path.
    ALWI void init_any_load() const {
        if constexpr (is_load_op_v<First>) {
            first.init();
        } else {
            rest.init_any_load();
        }
    }

    // Reset cb_tile_idx on all WaitUpfrontPopAtEnd Load elements so the chain
    // can be reused across blocks without re-construction.
    ALWI void reset_tile_idx() const {
        if constexpr (detail::has_reset_tile_idx<First>::value) {
            first.reset_tile_idx();
        }
        rest.reset_tile_idx();
    }

    // Bulk CB wait/pop for all WaitUpfrontPopAtEnd Load elements.
    // Called by sfpu_pipeline once before the tile loop (wait) and once after (pop).
    ALWI void wait_upfront(uint32_t n) const {
        if constexpr (detail::has_wait_upfront<First>::value) {
            first.wait_upfront(n);
        }
        rest.wait_upfront(n);
    }
    ALWI void pop_upfront(uint32_t n) const {
        if constexpr (detail::has_pop_upfront<First>::value) {
            first.pop_upfront(n);
        }
        rest.pop_upfront(n);
    }
};

// =============================================================================
// sfpu_chain() Factory
// =============================================================================

/**
 * @brief Factory function — returns an SfpuChain holding the given ops.
 *
 * No compile-time transformation. Each Load manages its own CB lifecycle at
 * exec() time (wait/copy/pop) per its LoadPolicy. For fan-out (same CB tile to
 * multiple DEST slots), specify explicit policies so the CB is waited/popped
 * exactly once:
 *
 *   sfpu_chain(
 *       Load<cb_input, Dst::D0, LoadPolicy::WaitNoPop>{},    // wait once
 *       Load<cb_input, Dst::D1, LoadPolicy::WaitAndPop>{},   // cb_wait idempotent; pop once
 *       Exp<Approx::Fast, Approx::Fast, Dst::D0>{},
 *       SfpuMul<Dst::D0, Dst::D1, Dst::D0>{});
 *
 *   // Data streams, mask is persistent across iterations.
 *   sfpu_chain(
 *       Load<cb_data, Dst::D0>{},                                // WaitAndPop (default)
 *       Load<cb_mask, Dst::D1, LoadPolicy::WaitNoPop>{},         // persistent
 *       Mask<DataFormat::Float16_b>{});
 */
template <typename... Ops>
constexpr ALWI auto sfpu_chain(Ops... ops) {
    return SfpuChain<Ops...>{ops...};
}

// =============================================================================
// Chain Traits (defined after SfpuChain to use its full definition)
// =============================================================================

/** @brief True if T is a non-Load element with clashes_with_fpu = true.
 *
 * This identifies elements like DestReuseOp that run binary_dest_reuse_tiles_init,
 * which clobbers the copy_tile_to_dst_init_short MOP state. After such an element
 * runs, any subsequent Load in the next tile iteration needs copy_tile_to_dst_init_short
 * to be called again.
 *
 * Load ops have clashes_with_fpu = true too (they SET up the copy state), but they
 * do not clobber it — only non-Load elements that run FPU binary_dest_reuse_tiles_init do.
 */
template <typename T, typename = void>
struct has_non_load_fpu_clash : std::false_type {};
template <typename T>
struct has_non_load_fpu_clash<T, std::void_t<decltype(T::clashes_with_fpu)>>
    : std::bool_constant<T::clashes_with_fpu && !is_load_op_v<T>> {};

/** @brief True if any element in Chain clobbers copy_tile init state (e.g. contains DestReuseOp). */
template <typename Chain>
struct chain_has_non_load_fpu_clash : std::false_type {};
template <typename... Ops>
struct chain_has_non_load_fpu_clash<SfpuChain<Ops...>>
    : std::bool_constant<(has_non_load_fpu_clash<Ops>::value || ...)> {};
template <typename T>
inline constexpr bool chain_has_non_load_fpu_clash_v =
    chain_has_non_load_fpu_clash<std::remove_cv_t<std::remove_reference_t<T>>>::value;

/** @brief True if Chain contains at least one Load element. */
template <typename Chain>
struct chain_has_any_load : std::false_type {};
template <typename... Ops>
struct chain_has_any_load<SfpuChain<Ops...>> : std::bool_constant<(is_load_op_v<Ops> || ...)> {};
template <typename T>
inline constexpr bool chain_has_any_load_v = chain_has_any_load<std::remove_cv_t<std::remove_reference_t<T>>>::value;

namespace detail {

/** @brief CB of the first Load in a chain, or 0 if none (sentinel). */
template <typename Chain>
struct FirstLoadCB {
    static constexpr uint32_t value = 0;
};
template <typename First, typename... Rest>
struct FirstLoadCB<SfpuChain<First, Rest...>> {
    static constexpr uint32_t value = FirstLoadCB<SfpuChain<Rest...>>::value;
};
template <uint32_t CB, Dst Slot, LoadPolicy Policy, LoadReconfig Reconfig, typename... Rest>
struct FirstLoadCB<SfpuChain<Load<CB, Slot, Policy, Reconfig>, Rest...>> {
    static constexpr uint32_t value = CB;
};

// All Loads in Chain have CB == Target (or Chain has no Loads).
template <uint32_t Target, typename Chain>
struct chain_all_loads_cb_eq;

template <uint32_t Target>
struct chain_all_loads_cb_eq<Target, SfpuChain<>> : std::true_type {};

// Load first element: check match, recurse.
template <uint32_t Target, uint32_t CB, Dst Slot, LoadPolicy Policy, LoadReconfig Reconfig, typename... Rest>
struct chain_all_loads_cb_eq<Target, SfpuChain<Load<CB, Slot, Policy, Reconfig>, Rest...>>
    : std::bool_constant<(CB == Target) && chain_all_loads_cb_eq<Target, SfpuChain<Rest...>>::value> {};

// Non-Load first element: skip and recurse.
template <uint32_t Target, typename First, typename... Rest>
struct chain_all_loads_cb_eq<Target, SfpuChain<First, Rest...>> : chain_all_loads_cb_eq<Target, SfpuChain<Rest...>> {};

}  // namespace detail

/** @brief True if every Load in Chain reads from the same CB (or Chain has no Loads). */
template <typename Chain>
inline constexpr bool chain_loads_share_cb_v = detail::chain_all_loads_cb_eq<
    detail::FirstLoadCB<std::remove_cv_t<std::remove_reference_t<Chain>>>::value,
    std::remove_cv_t<std::remove_reference_t<Chain>>>::value;

/** @brief True when Load::init() can be safely hoisted out of the per-tile cascade.
 *
 * Hoist is safe when: chain has at least one Load, no non-Load element clobbers
 * the copy_tile MOP (no DestReuseOp), and every Load reads the same CB (one
 * hoisted init programs the MOP for all).
 */
template <typename Chain>
inline constexpr bool chain_is_hoist_safe_v =
    chain_has_any_load_v<Chain> && !chain_has_non_load_fpu_clash_v<Chain> && chain_loads_share_cb_v<Chain>;

// -----------------------------------------------------------------------------
// Upfront-CB duplicate detection
//
// An element is an "upfront CB input" when it exposes `static constexpr bool
// is_upfront == true` and `static constexpr uint32_t cb`. Load sets these;
// DestReuseOp sets these. The pipeline bulk-waits and bulk-pops `num_tiles`
// on each upfront CB once per chain call — two upfront elements sharing a
// CB would double-pop and desync it. We statically forbid it, uniformly
// across Load and DestReuseOp (and future CB-input ops that opt in via the
// same static members).
// -----------------------------------------------------------------------------

namespace detail {

// SFINAE detector for T::is_upfront + T::cb. Any chain element that declares
// these static constexpr members participates in upfront-CB bookkeeping.
template <typename T, typename = void>
struct upfront_cb_info {
    static constexpr bool is_upfront = false;
    static constexpr uint32_t cb = 0;
};
template <typename T>
struct upfront_cb_info<T, std::void_t<decltype(T::is_upfront), decltype(T::cb)>> {
    static constexpr bool is_upfront = T::is_upfront;
    static constexpr uint32_t cb = T::cb;
};

// Does any upfront CB-input element in Chain reference TargetCB?
template <uint32_t TargetCB, typename Chain>
struct chain_has_upfront_cb : std::false_type {};

template <uint32_t TargetCB>
struct chain_has_upfront_cb<TargetCB, SfpuChain<>> : std::false_type {};

template <uint32_t TargetCB, typename First, typename... Rest>
struct chain_has_upfront_cb<TargetCB, SfpuChain<First, Rest...>>
    : std::bool_constant<
          (upfront_cb_info<First>::is_upfront && upfront_cb_info<First>::cb == TargetCB) ||
          chain_has_upfront_cb<TargetCB, SfpuChain<Rest...>>::value> {};

// Does Chain contain two upfront CB-input elements sharing the same CB?
// (For every upfront element, check if any later upfront element reuses its CB.)
template <typename Chain>
struct chain_has_duplicate_upfront_cbs : std::false_type {};

template <>
struct chain_has_duplicate_upfront_cbs<SfpuChain<>> : std::false_type {};

template <typename First, typename... Rest>
struct chain_has_duplicate_upfront_cbs<SfpuChain<First, Rest...>>
    : std::bool_constant<
          (upfront_cb_info<First>::is_upfront &&
           chain_has_upfront_cb<upfront_cb_info<First>::cb, SfpuChain<Rest...>>::value) ||
          chain_has_duplicate_upfront_cbs<SfpuChain<Rest...>>::value> {};

}  // namespace detail

template <typename Chain>
inline constexpr bool chain_has_duplicate_upfront_cbs_v =
    detail::chain_has_duplicate_upfront_cbs<std::remove_cv_t<std::remove_reference_t<Chain>>>::value;

// =============================================================================
// Pipeline Function Declaration
// =============================================================================

template <
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::NONE,
    typename Chain>
ALWI void sfpu_pipeline(Chain& chain, uint32_t ocb, uint32_t num_tiles, Dst pack_slot = Dst::D0);

// =============================================================================
// Convenience: Single Unary Op Declaration
// =============================================================================

template <
    uint32_t ICB,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::NONE,
    typename Op>
ALWI void sfpu_op(uint32_t ocb, uint32_t num_tiles, Op op);

}  // namespace compute_kernel_lib

#include "sfpu_chain.inl"
