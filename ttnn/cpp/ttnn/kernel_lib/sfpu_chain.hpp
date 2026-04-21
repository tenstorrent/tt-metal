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
// Load Op
// =============================================================================

/**
 * @brief Per-Load CB lifecycle policy — 2x2 matrix over {wait?} x {pop?}.
 *
 * The Load op owns the decision of whether to wait on the CB and whether to pop
 * after copying. The pipeline does NOT override this.
 *
 * - WaitAndPop:  wait for tile, copy, pop (default; streaming input)
 * - WaitNoPop:   wait for tile, copy, don't pop (persistent tile reused across
 *                iterations, e.g. a mask or scaler, or the first of a fan-out pair)
 * - NoWaitPop:   don't wait, copy, pop (caller pre-waited a batch of tiles upfront
 *                and wants this Load to consume one; or the last of a fan-out pair
 *                where the first already waited — OR use WaitAndPop since
 *                cb_wait_front is idempotent)
 * - NoWaitNoPop: don't wait, don't pop (caller owns CB lifecycle externally,
 *                e.g. sharded / pre-loaded inputs)
 */
enum class LoadPolicy {
    WaitAndPop,
    WaitNoPop,
    NoWaitPop,
    NoWaitNoPop,
};

constexpr bool load_does_wait(LoadPolicy p) { return p == LoadPolicy::WaitAndPop || p == LoadPolicy::WaitNoPop; }
constexpr bool load_does_pop(LoadPolicy p) { return p == LoadPolicy::WaitAndPop || p == LoadPolicy::NoWaitPop; }

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
 * The CB tile index defaults to 0 (streaming) but is a runtime field so callers
 * with a pre-waited batch can pick a specific tile:
 *   auto load = Load<cb, Dst::D0, LoadPolicy::NoWaitNoPop>{};
 *   load.cb_tile_idx = 3;
 *   sfpu_chain(load, SomeOp{});
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
    static_assert(static_cast<uint32_t>(Slot) < 8, "DEST slot exceeds maximum capacity (8)");

    uint32_t cb_tile_idx = 0;

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
 * @brief Variadic chain of ops (Load + compute)
 *
 * All elements have init()/exec()/apply():
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
