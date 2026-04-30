// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "api/compute/common_globals.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/debug/assert.h"

#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

/**
 * @file eltwise_chain.hpp
 * @brief V2 eltwise helper foundation: Dst slot enum, flag enums, CRTP bases,
 *        CopyTile element, EltwiseChain combinator, eltwise_pipeline runner.
 *
 * NOTE: this is the V2 eltwise helper family. It does NOT call into
 * `sfpu_helpers.{hpp,inl}` or `binary_op_helpers.{hpp,inl}` — op structs in
 * `eltwise_math.hpp` etc. dispatch directly to the underlying
 * `compute_kernel_api/eltwise_unary/...h` LLK wrappers (which is the public
 * compute LLK surface — `exp_tile_init`, `exp_tile`, etc.) or to
 * `tile_move_copy.h` / `pack.h` / `reg_api.h` for tile movement and DEST
 * lifecycle. The legacy V1 helpers remain in place for callers that have not
 * migrated; this V2 family is independent.
 *
 * ## Caller-facing example (single unary Exp)
 *
 * ```cpp
 * #include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
 * #include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
 *
 * void kernel_main() {
 *     constexpr uint32_t cb_in  = tt::CBIndex::c_0;
 *     constexpr uint32_t cb_out = tt::CBIndex::c_16;
 *     constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
 *
 *     compute_kernel_hw_startup(cb_in, cb_out);
 *     init_sfpu(cb_in, cb_out);
 *
 *     compute_kernel_lib::eltwise_pipeline<cb_out>(num_tiles,
 *         compute_kernel_lib::eltwise_chain(
 *             compute_kernel_lib::CopyTile<cb_in>{},
 *             compute_kernel_lib::Exp<>{}));
 * }
 * ```
 *
 * ## Design notes
 *
 *  - DEST capacity uses `DEST_AUTO_LIMIT` from `dest_helpers.hpp`. Never
 *    literal 8/16.
 *  - Per-tile init is the default. Hoisting is opt-in via the `EnableHoist`
 *    template flag on `eltwise_pipeline` AND the chain trait
 *    `chain_is_hoist_safe_v` must be true. A wrong hoist silently produces
 *    wrong output (lessons §3.4).
 *  - Same-CB CopyTile dedup is performed by the pipeline (lessons §3.6) —
 *    the user writes N CopyTiles for N DEST slots; the pipeline detects
 *    matching CBs and emits one `cb_wait_front` + one `cb_pop_front`.
 *  - `clobbers_sfpu_lut` is a per-op trait. Two LUT-clobbering ops in the
 *    same chain disqualify hoisting (their inits race the LUT).
 */

namespace compute_kernel_lib {

// =============================================================================
// Dst slot enum
// =============================================================================

/**
 * @brief Self-documenting DEST register slot indices.
 *
 * Bound at compile time by `DEST_AUTO_LIMIT` (sync mode + DST_ACCUM_MODE).
 */
enum class Dst : uint32_t { D0 = 0, D1 = 1, D2 = 2, D3 = 3, D4 = 4, D5 = 5, D6 = 6, D7 = 7 };

// =============================================================================
// Flag enums (lessons §1.3)
// =============================================================================

enum class Approx : bool { Exact = false, Fast = true };
enum class Legacy : bool { Off = false, On = true };
enum class FP32DestAcc : bool { Off = false, On = true };
enum class MathFidelity : uint32_t { LoFi = 0, HiFi2 = 2, HiFi3 = 3, HiFi4 = 4 };
enum class OutputActivation : uint32_t { None = 0, Relu = 1, Relu6 = 2 };
enum class VectorMode : int { R = 1, C = 2, RC = 4 };
enum class RoundingMode : uint32_t { None = 0, Up = 1, Down = 2, Nearest = 3, Trunc = 4 };

// =============================================================================
// Broadcast / DEST reuse / op-type enums
// =============================================================================

enum class BroadcastDim : uint32_t {
    NONE = 0,    // [Ht, Wt]
    ROW = 1,     // [1, Wt] — replicate down
    COL = 2,     // [Ht, 1] — replicate right
    SCALAR = 3,  // [1, 1]  — replicate everywhere
};

enum class BroadcastSide : uint32_t { LHS = 0, RHS = 1 };

enum class DestReuseType : uint32_t {
    DEST_TO_SRCA = 0,
    DEST_TO_SRCB = 1,
};

enum class DestReuseReconfig : uint32_t {
    None = 0,
    Input = 1,  // srca (DEST_TO_SRCB) or srcb (DEST_TO_SRCA) format reconfig
};

enum class CopyTileReconfig : uint32_t {
    None = 0,
    Input = 1,  // srca format reconfig before copy_tile_to_dst
};

enum class BinaryOpType : uint32_t { Add = 0, Sub = 1, Mul = 2 };

// =============================================================================
// CopyTilePolicy (6 corners) — lessons §2.1
// =============================================================================

/**
 * Two orthogonal axes (when wait, when pop).
 *
 * | Policy                       | Wait shape          | Pop shape       |
 * |------------------------------|---------------------|-----------------|
 * | WaitAndPop                   | per-tile            | per-tile        |
 * | WaitNoPop                    | per-tile            | none            |
 * | NoWaitPop                    | none                | per-tile        |
 * | NoWaitNoPop                  | none                | none            |
 * | WaitUpfrontPopAtEnd          | upfront, count=N    | upfront, count=N|
 * | CumulativeWaitUpfrontEndPop  | wait(base+i*step)   | upfront, count=N|
 *
 * `CumulativeWaitUpfrontEndPop` is needed by 6 production kernels (rmsnorm /
 * layernorm pre-allgather variants + DIT layernorm + fused-rmsnorm). The
 * runner emits a growing-window `cb_wait_front(cb, base + (i+1)*step)` per
 * tile-batch and pops once at end-of-loop.
 */
enum class CopyTilePolicy : uint32_t {
    WaitAndPop = 0,
    WaitNoPop = 1,
    NoWaitPop = 2,
    NoWaitNoPop = 3,
    WaitUpfrontPopAtEnd = 4,
    CumulativeWaitUpfrontEndPop = 5,
};

// =============================================================================
// CbIndexMode — lessons §2.7
// =============================================================================

enum class CbIndexMode : uint32_t {
    FirstTile = 0,  // always tile 0 (streaming consume)
    BlockIter = 1,  // tile i — index is per-tile loop counter
    Pinned = 2,     // fixed runtime index (scalar / mask / broadcast-once)
    Absolute = 3,   // arbitrary runtime index inside the waited window
};

// =============================================================================
// Tag types and traits
// =============================================================================

struct CopyTileTag {};

namespace detail {

template <typename T, typename = void>
struct has_clobbers_sfpu_lut : std::false_type {};
template <typename T>
struct has_clobbers_sfpu_lut<T, std::void_t<decltype(T::clobbers_sfpu_lut)>>
    : std::bool_constant<T::clobbers_sfpu_lut> {};
template <typename T>
inline constexpr bool clobbers_sfpu_lut_v = has_clobbers_sfpu_lut<T>::value;

template <typename T, typename = void>
struct has_clashes_with_fpu : std::false_type {};
template <typename T>
struct has_clashes_with_fpu<T, std::void_t<decltype(T::clashes_with_fpu)>> : std::bool_constant<T::clashes_with_fpu> {};
template <typename T>
inline constexpr bool clashes_with_fpu_v = has_clashes_with_fpu<T>::value;

template <typename T>
inline constexpr bool is_copy_tile_op_v = std::is_base_of_v<CopyTileTag, T>;

}  // namespace detail

// =============================================================================
// CRTP bases — lessons §1.1
// =============================================================================

/**
 * @brief Unary SFPU op base — derived defines `init()` and `call(uint32_t dst)`.
 */
template <typename Derived, Dst Slot>
struct UnaryOp {
    static_assert(
        static_cast<uint32_t>(Slot) < DEST_AUTO_LIMIT,
        "UnaryOp DEST slot exceeds DEST_AUTO_LIMIT in this sync/accum mode");

    static constexpr Dst dst_slot = Slot;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static constexpr bool is_copy_tile_op = false;
    // Default trait values; ops can override by re-declaring in their struct.
    static constexpr bool clashes_with_fpu = false;
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI void exec() const { static_cast<const Derived*>(this)->call(dst_idx); }
    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        exec();
    }
};

/**
 * @brief Binary SFPU op base — three DEST slots (in0, in1, out).
 */
template <typename Derived, Dst In0, Dst In1, Dst Out>
struct BinaryOp {
    static_assert(static_cast<uint32_t>(In0) < DEST_AUTO_LIMIT, "BinaryOp In0 exceeds DEST_AUTO_LIMIT");
    static_assert(static_cast<uint32_t>(In1) < DEST_AUTO_LIMIT, "BinaryOp In1 exceeds DEST_AUTO_LIMIT");
    static_assert(static_cast<uint32_t>(Out) < DEST_AUTO_LIMIT, "BinaryOp Out exceeds DEST_AUTO_LIMIT");

    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static constexpr uint32_t max_dst = (in0 > in1 ? (in0 > out ? in0 : out) : (in1 > out ? in1 : out));
    static constexpr bool is_copy_tile_op = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI void exec() const { static_cast<const Derived*>(this)->call(in0, in1, out); }
    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        exec();
    }
};

/**
 * @brief Ternary SFPU op base — four DEST slots (in0, in1, in2, out). Strict order.
 */
template <typename Derived, Dst In0, Dst In1, Dst In2, Dst Out>
struct TernaryOp {
    static_assert(static_cast<uint32_t>(In0) < DEST_AUTO_LIMIT, "TernaryOp In0 exceeds DEST_AUTO_LIMIT");
    static_assert(static_cast<uint32_t>(In1) < DEST_AUTO_LIMIT, "TernaryOp In1 exceeds DEST_AUTO_LIMIT");
    static_assert(static_cast<uint32_t>(In2) < DEST_AUTO_LIMIT, "TernaryOp In2 exceeds DEST_AUTO_LIMIT");
    static_assert(static_cast<uint32_t>(Out) < DEST_AUTO_LIMIT, "TernaryOp Out exceeds DEST_AUTO_LIMIT");

    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t in2 = static_cast<uint32_t>(In2);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static constexpr bool is_copy_tile_op = false;
    static constexpr bool clashes_with_fpu = false;
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI void exec() const { static_cast<const Derived*>(this)->call(in0, in1, in2, out); }
    ALWI void apply() const {
        static_cast<const Derived*>(this)->init();
        exec();
    }
};

// =============================================================================
// CopyTile element — lessons §2.7, §2.8, §3.5
// =============================================================================

/**
 * @brief One CB tile → one DEST slot. Fan-out to N slots is N CopyTile elements.
 *
 * Wait/pop is governed by `Policy`. Index mode is governed by `IndexMode`:
 *
 * | Policy ↓  /  IndexMode →    | FirstTile | BlockIter | Pinned | Absolute |
 * |-----------------------------|-----------|-----------|--------|----------|
 * | WaitAndPop                  | OK        | static    | OK*    | static   |
 * | WaitNoPop                   | OK        | static    | OK*    | static   |
 * | NoWaitPop                   | OK        | static    | OK*    | static   |
 * | NoWaitNoPop                 | OK        | OK        | OK     | OK       |
 * | WaitUpfrontPopAtEnd         | OK        | OK        | OK     | OK       |
 * | CumulativeWaitUpfrontEndPop | OK*       | OK        | OK     | OK       |
 *
 * `static` cells are rejected by `static_assert` below. For the single-tile-
 * window policies, `Pinned` is only legal when the runtime index is 0 (caller
 * responsibility — runtime ASSERT in the runner). `OK*` for FirstTile under
 * cumulative means tile 0 of the most recent chunk, which is rarely useful;
 * BlockIter is the canonical mode for that policy.
 */
template <
    uint32_t CB,
    Dst Slot = Dst::D0,
    CopyTilePolicy Policy = CopyTilePolicy::WaitAndPop,
    CbIndexMode IndexMode = CbIndexMode::FirstTile,
    CopyTileReconfig Reconfig = CopyTileReconfig::None>
struct CopyTile : CopyTileTag {
    static_assert(static_cast<uint32_t>(Slot) < DEST_AUTO_LIMIT, "CopyTile DEST slot exceeds DEST_AUTO_LIMIT");

    // Compatibility matrix — single-tile-window policies admit only FirstTile
    // or Pinned (runtime k==0). BlockIter / Absolute structurally unsafe.
    static_assert(
        !(Policy == CopyTilePolicy::WaitAndPop || Policy == CopyTilePolicy::WaitNoPop ||
          Policy == CopyTilePolicy::NoWaitPop) ||
            (IndexMode == CbIndexMode::FirstTile || IndexMode == CbIndexMode::Pinned),
        "BlockIter / Absolute index mode requires a multi-tile window policy "
        "(WaitUpfrontPopAtEnd, NoWaitNoPop, or CumulativeWaitUpfrontEndPop).");

    static constexpr uint32_t cb = CB;
    static constexpr Dst dst_slot = Slot;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static constexpr CopyTilePolicy policy = Policy;
    static constexpr CbIndexMode index_mode = IndexMode;
    static constexpr CopyTileReconfig reconfig = Reconfig;

    static constexpr bool is_copy_tile_op = true;
    static constexpr bool clashes_with_fpu = true;  // unpack MOP
    static constexpr bool clobbers_sfpu_lut = false;

    static constexpr bool is_upfront =
        (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) || (Policy == CopyTilePolicy::CumulativeWaitUpfrontEndPop);

    static constexpr bool policy_does_per_tile_wait =
        (Policy == CopyTilePolicy::WaitAndPop) || (Policy == CopyTilePolicy::WaitNoPop);
    static constexpr bool policy_does_per_tile_pop =
        (Policy == CopyTilePolicy::WaitAndPop) || (Policy == CopyTilePolicy::NoWaitPop);
    static constexpr bool policy_does_upfront_pop =
        (Policy == CopyTilePolicy::WaitUpfrontPopAtEnd) || (Policy == CopyTilePolicy::CumulativeWaitUpfrontEndPop);

    // Runtime index — used only for Pinned / Absolute. For FirstTile and
    // BlockIter the index is determined entirely by the loop counter.
    uint32_t cb_tile_idx_runtime = 0;

    // Pipeline-private monotone counter for BlockIter under upfront / cumulative
    // wait policies. Reset by the pipeline at end-of-block.
    mutable uint32_t cb_tile_idx_ = 0;
};

// =============================================================================
// EltwiseChain combinator and traits
// =============================================================================

namespace detail {

template <typename... Es>
constexpr bool any_is_copy_tile() {
    return (is_copy_tile_op_v<Es> || ... || false);
}

template <typename... Es>
constexpr bool any_non_copy_tile_fpu_clash() {
    return (((!is_copy_tile_op_v<Es>) && clashes_with_fpu_v<Es>) || ... || false);
}

template <typename... Es>
constexpr uint32_t count_lut_clobbers() {
    return ((clobbers_sfpu_lut_v<Es> ? 1u : 0u) + ... + 0u);
}

template <typename... Es>
constexpr uint32_t chain_max_dst() {
    if constexpr (sizeof...(Es) == 0) {
        return 0;
    } else {
        uint32_t m = 0;
        ((m = (Es::max_dst > m ? Es::max_dst : m)), ...);
        return m;
    }
}

}  // namespace detail

template <typename... Elements>
struct EltwiseChain {
    static constexpr size_t num_elements = sizeof...(Elements);

    static constexpr bool has_any_copy_tile = detail::any_is_copy_tile<Elements...>();
    static constexpr bool has_non_copy_tile_fpu_clash = detail::any_non_copy_tile_fpu_clash<Elements...>();
    static constexpr uint32_t max_dst = detail::chain_max_dst<Elements...>();

    // Stride is the DEST footprint of one tile — chain-author writes ops on
    // the smallest legal slot set, so stride = max_dst + 1.
    static constexpr uint32_t stride = max_dst + 1;

    static_assert(stride <= DEST_AUTO_LIMIT, "Chain DEST footprint exceeds DEST_AUTO_LIMIT");

    std::tuple<Elements...> elements;

    ALWI explicit EltwiseChain(Elements... e) : elements{e...} {}
};

template <typename Chain>
inline constexpr bool chain_has_any_copy_tile_v = Chain::has_any_copy_tile;

template <typename Chain>
inline constexpr bool chain_has_non_copy_tile_fpu_clash_v = Chain::has_non_copy_tile_fpu_clash;

namespace detail {

template <typename Chain>
struct LutClobberCollision {
    static constexpr uint32_t value = 0;
};
template <typename... Es>
struct LutClobberCollision<EltwiseChain<Es...>> {
    static constexpr uint32_t value = count_lut_clobbers<Es...>();
};

}  // namespace detail

/**
 * @brief True if two or more elements in the chain mutate the SFPU LUT.
 *
 * Init for one LUT op overwrites the LUT the previous op's init programmed.
 * If both inits are hoisted, the second wins and the first op produces wrong
 * output every iteration. Disqualifies hoisting.
 */
template <typename Chain>
inline constexpr bool chain_has_lut_clobber_collision_v = (detail::LutClobberCollision<Chain>::value > 1);

/**
 * @brief Hoist-safety gate (lessons §3.4).
 *
 * Per-tile init is the default. Hoisting is allowed only when:
 *   1. The chain has at least one CopyTile load (otherwise nothing to amortize).
 *   2. No non-CopyTile element clobbers FPU state (DestReuseOp, BinaryFpuElement).
 *   3. At most one element clobbers the SFPU LUT.
 */
template <typename Chain>
inline constexpr bool chain_is_hoist_safe_v =
    Chain::has_any_copy_tile && (!Chain::has_non_copy_tile_fpu_clash) && (!chain_has_lut_clobber_collision_v<Chain>);

/**
 * @brief Variadic factory for EltwiseChain.
 */
template <typename... Elements>
ALWI constexpr auto eltwise_chain(Elements... e) {
    return EltwiseChain<Elements...>(e...);
}

// =============================================================================
// eltwise_pipeline — the single runner (declaration; impl in .inl)
// =============================================================================

/**
 * @brief Wrap an EltwiseChain in a tile_regs lifecycle and CB lifecycle.
 *
 * Algorithm per tile (default — EnableHoist=false, no upfront elements):
 *   1. tile_regs_acquire().
 *   2. For each element in declared order:
 *      - CopyTile: cb_wait_front (per-tile policy) + init() + copy_tile + (cb_pop_front).
 *      - Other:    init() + exec().
 *      Same-CB CopyTile dedup: wait on first user, pop on last user.
 *   3. tile_regs_commit() → tile_regs_wait().
 *   4. cb_reserve_back(OutCB, 1) → pack_tile(out_slot, OutCB) → cb_push_back.
 *   5. tile_regs_release().
 *
 * With EnableHoist=true (and chain_is_hoist_safe_v<Chain>): inits are run once
 * before the loop; per-tile body only runs exec(). FPU-clashing chains are
 * never hoisted.
 *
 * Upfront-policy CopyTile: waits emitted once before the loop with count =
 * num_tiles; pops once at the end. CumulativeWaitUpfrontEndPop emits a growing
 * cb_wait_front each iteration (count = i+1) and pops once at end.
 */
template <uint32_t OutCB, bool EnableHoist = false, typename Chain>
ALWI void eltwise_pipeline(uint32_t num_tiles, Chain chain);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl"
