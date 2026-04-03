// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include "api/compute/common_globals.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/cbrt.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/hardmish.h"
#include "api/compute/eltwise_unary/hardtanh.h"
#include "api/compute/eltwise_unary/activations.h"
#include "api/compute/eltwise_unary/softplus.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/log1p.h"
#include "api/compute/eltwise_unary/xielu.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/eltwise_unary/erf_erfc.h"
#include "api/compute/eltwise_unary/erfinv.h"
#include "api/compute/eltwise_unary/isinf_isnan.h"
#include "api/compute/eltwise_unary/logical_not.h"
#include "api/compute/eltwise_unary/i0.h"
#include "api/compute/eltwise_unary/i1.h"
#include "api/compute/eltwise_unary/lgamma.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/elu.h"
#include "api/compute/eltwise_unary/selu.h"
#include "api/compute/eltwise_unary/clamp.h"
#include "api/compute/eltwise_unary/threshold.h"
#include "api/compute/eltwise_unary/prelu.h"
#include "api/compute/eltwise_unary/rounding.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/eltwise_unary/identity.h"
#include "api/compute/eltwise_unary/dropout.h"
// NOTE: bitwise_and/or/xor/not and left_shift/right_shift headers are intentionally
// excluded — their underlying ckernel headers use `using namespace sfpi` and define
// operators (e.g. operator&) that create ambiguous overloads when combined with
// reduce.h or other LLK headers.  Include them directly in kernels that need them.
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsub.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fmod.h"
#include "api/compute/eltwise_unary/remainder.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rand.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/lerp.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/eltwise_unary/addcdiv.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/common_types.hpp"

/**
 * @file sfpu_helpers.hpp
 * @brief SFPU operation helpers: slot-bound op structs, chain combinator, and pipeline
 *
 * Provides a composable API for SFPU compute kernels. Each SFPU operation is represented
 * as a lightweight struct with compile-time DEST slot binding. Multiple ops can be combined
 * into a chain, then executed by a pipeline that manages DEST registers, CB synchronization,
 * and data format reconfiguration.
 *
 * PREREQUISITE: Call init_sfpu(icb, ocb) at the start of your kernel before using
 * the pipeline functions.
 *
 * ## Core Concepts
 *
 * - **Dst enum**: Self-documenting DEST slot offset names (D0..D7), used as template parameters
 * - **Load<CB, Slot>**: Copies a tile from circular buffer CB into DEST[Slot + offset]
 * - **Op structs**: Each wraps a single SFPU LLK call pair (init + exec), with offset support
 *   - init(): configures SFPU hardware for this operation
 *   - exec(offset): executes on DEST[slot + offset]
 *   - apply(offset): convenience for init(); exec(offset);
 * - **SfpuChain**: Variadic combinator with stride auto-detection and batched execution
 * - **sfpu_pipeline()**: Auto-batched loop that fills DEST with multiple chain iterations
 *
 * ## DEST Slot Capacity
 *
 * The maximum valid DEST slot depends on sync and accumulation modes:
 * - SyncHalf + fp16: 8 slots (D0-D7)
 * - SyncHalf + fp32: 4 slots (D0-D3)
 * - SyncFull + fp16: 16 slots
 * - SyncFull + fp32: 8 slots (D0-D7)
 *
 * Compile-time static_assert validates slot indices against the max (8).
 * Runtime ASSERT validates against DEST_AUTO_LIMIT (which may be 4 in fp32 half-sync).
 *
 * ── Examples ────────────────────────────────────────────────────────────────
 *
 *   #include "ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp"
 *   using namespace compute_kernel_lib;
 *
 *   init_sfpu(cb_in, cb_out);
 *
 *   // 1. Single unary op — exp on all tiles (most common pattern)
 *   sfpu_op<cb_in>(cb_out, num_tiles, Exp<>{});
 *
 *   // 2. Named alias — same as above but even shorter
 *   sfpu_exp<cb_in>(cb_out, num_tiles);
 *
 *   // 3. Softplus chain: exp(x) + 1 -> log
 *   //    Load x into D0, load ones into D1, then exp -> add -> log
 *   constexpr uint32_t cb_x = 0, cb_ones = 1, cb_out = 2;
 *   auto chain = sfpu_chain(
 *       Load<cb_x, Dst::D0>{},
 *       Load<cb_ones, Dst::D1>{},
 *       Exp<Dst::D0>{},
 *       SfpuAdd<Dst::D0, Dst::D1, Dst::D0>{},
 *       Log<Dst::D0>{}
 *   );
 *   sfpu_pipeline(chain, cb_out, 1 [num_tiles]);
 *
 *   // 4. Hardswish: x * hardsigmoid(x)
 *   //    Load x into D0 and D1, hardsigmoid on D0, mul D0*D1->D0
 *   constexpr uint32_t cb_input = 0, cb_output = 2;
 *   auto chain = sfpu_chain(
 *       Load<cb_input, Dst::D0>{},
 *       Load<cb_input, Dst::D1>{},
 *       Hardsigmoid<Dst::D0>{},
 *       SfpuMul<Dst::D0, Dst::D1, Dst::D0>{}
 *   );
 *   sfpu_pipeline<SfpuInputPolicy::WaitAndPopPerTile, SfpuOutputPolicy::Bulk>(
 *       chain, cb_output, per_core_block_dim);
 *
 *   // 5. Tanhshrink: x - tanh(x)
 *   auto chain = sfpu_chain(
 *       Load<cb_input, Dst::D0>{},
 *       Load<cb_input, Dst::D1>{},
 *       Tanh<Dst::D1>{},
 *       SfpuSub<Dst::D0, Dst::D1, Dst::D0>{}
 *   );
 *   sfpu_pipeline<SfpuInputPolicy::WaitAndPopPerTile, SfpuOutputPolicy::Bulk>(
 *       chain, cb_output, per_core_block_dim);
 *
 *   // 6. Parameterized op — hardtanh with min/max
 *   sfpu_op<cb_in>(cb_out, num_tiles, Hardtanh<>{min_val, max_val});
 *
 *   // 7. Auto-batching (default): DEST is filled with max iterations
 *   //    For Exp (stride=1, fp16 half-sync): batch_size = 8 tiles per acquire
 *   sfpu_op<cb_in>(cb_out, num_tiles, Exp<>{});
 *
 *   // 8. Disable batching (original per-tile behavior)
 *   sfpu_op<cb_in, SfpuBatching::Disabled>(cb_out, num_tiles, Exp<>{});
 *
 *   // 9. Batched chain: stride=2 (uses D0,D1), auto batch = DEST_AUTO_LIMIT/2
 *   auto chain = sfpu_chain(
 *       Load<cb_x, Dst::D0>{}, Load<cb_ones, Dst::D1>{},
 *       Exp<Dst::D0>{}, SfpuAdd<Dst::D0, Dst::D1, Dst::D0>{}
 *   );
 *   sfpu_pipeline(chain, cb_out, num_tiles);  // Auto by default
 *   sfpu_pipeline<SfpuBatching::Disabled>(chain, cb_out, num_tiles);  // Opt out
 *
 *   // 10. WaitUpfrontNoPop — tiles persist in CB for reuse
 *   sfpu_op<cb_in, SfpuBatching::Auto, SfpuInputPolicy::WaitUpfrontNoPop>(
 *       cb_out, num_tiles, Exp<>{});
 *
 *   // 11. Skip data format reconfiguration
 *   sfpu_op<cb_in, SfpuBatching::Auto, SfpuInputPolicy::WaitAndPopPerTile,
 *           SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::NONE>(
 *       cb_out, num_tiles, Sigmoid<>{});
 */

namespace compute_kernel_lib {

// Pull in ckernel namespace so SFPU LLK functions (exp_tile, sigmoid_tile, etc.)
// are visible from op struct methods during two-phase template lookup.
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
 * @brief User-facing Load: copies a tile from CB into DEST[Slot]
 *
 * Users write Load<cb, Dst::D0> in their chains. sfpu_chain() automatically
 * compacts adjacent same-CB Loads into CompactLoad elements.
 */
template <uint32_t CB, Dst Slot>
struct Load : LoadTag {
    static constexpr uint32_t cb = CB;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static constexpr uint32_t max_dst = dst_idx;
    static_assert(static_cast<uint32_t>(Slot) < 8, "DEST slot exceeds maximum capacity (8)");
};

/**
 * @brief Compacted Load: multiple DEST slots from the same CB, with wait/pop control
 *
 * Produced by sfpu_chain() compile-time transformation. Adjacent same-CB Loads
 * are merged into one CompactLoad. Wait/pop flags are set based on whether this
 * CB appears in other CompactLoad groups elsewhere in the chain.
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
// Op Struct Declarations
// =============================================================================

// --- Simple Math ---

template <Approx approx = Approx::Exact, Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Exp : UnaryOp<Exp<approx, fast, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Log : UnaryOp<Log<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct LogWithBase : UnaryOp<LogWithBase<Slot>, Slot> {
    uint32_t base_scale;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Log1p : UnaryOp<Log1p<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Sqrt : UnaryOp<Sqrt<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Legacy legacy = Legacy::Off, Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Rsqrt : UnaryOp<Rsqrt<legacy, approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Cbrt : UnaryOp<Cbrt<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Legacy legacy = Legacy::On, Dst Slot = Dst::D0>
struct Recip : UnaryOp<Recip<legacy, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Abs : UnaryOp<Abs<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Neg : UnaryOp<Neg<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Square : UnaryOp<Square<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Sign : UnaryOp<Sign<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Signbit : UnaryOp<Signbit<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Exp2 : UnaryOp<Exp2<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Expm1 : UnaryOp<Expm1<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Power : UnaryOp<Power<Slot>, Slot> {
    uint32_t exponent;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct PowerIterative : UnaryOp<PowerIterative<Slot>, Slot> {
    uint32_t int_exponent;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Rpow : UnaryOp<Rpow<Slot>, Slot> {
    uint32_t base_val;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Activations ---

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Sigmoid : UnaryOp<Sigmoid<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Tanh : UnaryOp<Tanh<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Gelu : UnaryOp<Gelu<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Silu : UnaryOp<Silu<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Relu : UnaryOp<Relu<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Hardmish : UnaryOp<Hardmish<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Hardsigmoid : UnaryOp<Hardsigmoid<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Hardtanh : UnaryOp<Hardtanh<Slot>, Slot> {
    uint32_t param_min;
    uint32_t param_max;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Softsign : UnaryOp<Softsign<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Softplus : UnaryOp<Softplus<Slot>, Slot> {
    uint32_t beta;
    uint32_t beta_recip;
    uint32_t threshold;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Xielu : UnaryOp<Xielu<Slot>, Slot> {
    uint32_t alpha_p;
    uint32_t alpha_n;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Trigonometry ---

template <Dst Slot = Dst::D0>
struct Sin : UnaryOp<Sin<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Cos : UnaryOp<Cos<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Tan : UnaryOp<Tan<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Asin : UnaryOp<Asin<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Acos : UnaryOp<Acos<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Atan : UnaryOp<Atan<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Sinh : UnaryOp<Sinh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Cosh : UnaryOp<Cosh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Asinh : UnaryOp<Asinh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Acosh : UnaryOp<Acosh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Atanh : UnaryOp<Atanh<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Error / Special Functions ---

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Erf : UnaryOp<Erf<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Erfc : UnaryOp<Erfc<approx, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Erfinv : UnaryOp<Erfinv<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct I0 : UnaryOp<I0<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct I1 : UnaryOp<I1<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Lgamma : UnaryOp<Lgamma<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Predicates & Comparisons ---

template <Dst Slot = Dst::D0>
struct Isinf : UnaryOp<Isinf<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Isposinf : UnaryOp<Isposinf<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Isneginf : UnaryOp<Isneginf<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Isnan : UnaryOp<Isnan<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Isfinite : UnaryOp<Isfinite<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <DataFormat df = DataFormat::Float16_b, Dst Slot = Dst::D0>
struct LogicalNot : UnaryOp<LogicalNot<df, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Gtz : UnaryOp<Gtz<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Ltz : UnaryOp<Ltz<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Lez : UnaryOp<Lez<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Gez : UnaryOp<Gez<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Eqz : UnaryOp<Eqz<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Nez : UnaryOp<Nez<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryEq : UnaryOp<UnaryEq<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryNe : UnaryOp<UnaryNe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryGt : UnaryOp<UnaryGt<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryGe : UnaryOp<UnaryGe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryLt : UnaryOp<UnaryLt<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct UnaryLe : UnaryOp<UnaryLe<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Additional Activations ---

template <Dst Slot = Dst::D0>
struct Elu : UnaryOp<Elu<Slot>, Slot> {
    uint32_t alpha;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Selu : UnaryOp<Selu<Slot>, Slot> {
    uint32_t scale;
    uint32_t alpha;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Celu : UnaryOp<Celu<Slot>, Slot> {
    uint32_t alpha;
    uint32_t alpha_recip;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Softshrink : UnaryOp<Softshrink<Slot>, Slot> {
    uint32_t lambda;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Clamp : UnaryOp<Clamp<Slot>, Slot> {
    uint32_t param_min;
    uint32_t param_max;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Threshold : UnaryOp<Threshold<Slot>, Slot> {
    uint32_t threshold;
    uint32_t value;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Prelu : UnaryOp<Prelu<Slot>, Slot> {
    uint32_t weight;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Rounding ---

template <Dst Slot = Dst::D0>
struct Floor : UnaryOp<Floor<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Ceil : UnaryOp<Ceil<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Trunc : UnaryOp<Trunc<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Round : UnaryOp<Round<Slot>, Slot> {
    int32_t decimals;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Frac : UnaryOp<Frac<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct StochasticRound : UnaryOp<StochasticRound<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Type / Identity / Bitwise ---

template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot = Dst::D0>
struct Typecast : UnaryOp<Typecast<in_dtype, out_dtype, Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Identity : UnaryOp<Identity<Slot>, Slot> {
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// NOTE: BitwiseNot, BitwiseAnd, BitwiseOr, BitwiseXor, LeftShift, RightShift
// are excluded — their ckernel headers pollute the global namespace with operators
// that conflict with reduce.h.  See note at top of includes.

// --- Scalar Arithmetic ---

template <Dst Slot = Dst::D0>
struct AddScalar : UnaryOp<AddScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct SubScalar : UnaryOp<SubScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct MulScalar : UnaryOp<MulScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct DivScalar : UnaryOp<DivScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct RsubScalar : UnaryOp<RsubScalar<Slot>, Slot> {
    uint32_t scalar;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Rsub : UnaryOp<Rsub<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <RoundingMode rounding_mode = RoundingMode::None, Dst Slot = Dst::D0>
struct Rdiv : UnaryOp<Rdiv<rounding_mode, Slot>, Slot> {
    uint32_t value;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Fmod : UnaryOp<Fmod<Slot>, Slot> {
    uint32_t param0;
    uint32_t param1;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Remainder : UnaryOp<Remainder<Slot>, Slot> {
    uint32_t param0;
    uint32_t param1;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct Dropout : UnaryOp<Dropout<Slot>, Slot> {
    uint32_t probability;
    uint32_t scale_factor;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Fill and Random ---

template <Dst Slot = Dst::D0>
struct FillTile : UnaryOp<FillTile<Slot>, Slot> {
    float fill_val;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct FillTileBitcast : UnaryOp<FillTileBitcast<Slot>, Slot> {
    uint32_t param0;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

template <Dst Slot = Dst::D0>
struct RandTile : UnaryOp<RandTile<Slot>, Slot> {
    uint32_t from;
    uint32_t scale;
    ALWI void init() const;
    ALWI void call(uint32_t d0) const;
};

// --- Binary SFPU Ops ---

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuAdd : BinaryOp<SfpuAdd<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuSub : BinaryOp<SfpuSub<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMul : BinaryOp<SfpuMul<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuDiv : BinaryOp<SfpuDiv<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuRsub : BinaryOp<SfpuRsub<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuPow : BinaryOp<SfpuPow<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuEq : BinaryOp<SfpuEq<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

// --- Ternary SFPU Ops ---

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Where : TernaryOp<Where<df, In0, In1, In2, Out>, In0, In1, In2, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Lerp : TernaryOp<Lerp<df, In0, In1, In2, Out>, In0, In1, In2, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Addcmul : TernaryOp<Addcmul<df, In0, In1, In2, Out>, In0, In1, In2, Out> {
    uint32_t value;
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Addcdiv : TernaryOp<Addcdiv<df, In0, In1, In2, Out>, In0, In1, In2, Out> {
    uint32_t value;
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const;
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

// --- Step 1: Compact adjacent same-CB Loads into CompactLoad (wait=true,pop=true initially) ---

// Helper: append a Load, merging with trailing same-CB CompactLoad if present.
// Uses a two-phase approach: first check if merge is possible, then act.

// Phase 1: Check if last element is CompactLoad<CB,...>
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

// Phase 3: Replace last CompactLoad<CB,...> by appending a new Slot
template <typename TL, uint32_t CB, Dst NewSlot>
struct ReplaceLastLoad;
template <uint32_t CB, bool W, bool P, Dst... Slots, Dst NewSlot>
struct ReplaceLastLoad<TypeList<CompactLoad<CB, W, P, Slots...>>, CB, NewSlot> {
    using type = TypeList<CompactLoad<CB, W, P, Slots..., NewSlot>>;
};
template <typename First, typename... Rest, uint32_t CB, Dst NewSlot>
struct ReplaceLastLoad<TypeList<First, Rest...>, CB, NewSlot> {
    using type = typename Prepend<First, typename ReplaceLastLoad<TypeList<Rest...>, CB, NewSlot>::type>::type;
};

// AppendLoad: dispatch merge vs new
template <typename TL, uint32_t CB, Dst Slot, bool Merge = LastIsCompactLoadFromCB<TL, CB>::value>
struct AppendLoad;

// No merge: append new CompactLoad
template <typename... Elems, uint32_t CB, Dst Slot>
struct AppendLoad<TypeList<Elems...>, CB, Slot, false> {
    using type = TypeList<Elems..., CompactLoad<CB, true, true, Slot>>;
};

// Merge: replace last CompactLoad with extended version
template <typename TL, uint32_t CB, Dst Slot>
struct AppendLoad<TL, CB, Slot, true> {
    using type = typename ReplaceLastLoad<TL, CB, Slot>::type;
};

// Fold step: dispatch Load vs non-Load via helper
template <typename Acc, typename Elem, bool IsLoad = is_load_op_v<Elem>>
struct CompactStep;

// Load: use AppendLoad to merge or create
template <typename Acc, uint32_t CB, Dst Slot>
struct CompactStep<Acc, Load<CB, Slot>, true> {
    using type = typename AppendLoad<Acc, CB, Slot>::type;
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

// --- Step 3: Annotate wait/pop based on first/last CB appearance ---
// After compaction + multi-group check, each CB appears exactly once.
// All CompactLoads get wait=true, pop=true (which is the initial default from Step 1).
// No further annotation needed for single-group-per-CB.

}  // namespace detail

/**
 * @brief Factory function — compacts adjacent same-CB Loads and returns transformed chain
 *
 * Usage: auto chain = sfpu_chain(Load<0, Dst::D0>{}, Load<0, Dst::D1>{}, Exp<>{});
 * Produces: SfpuChain<CompactLoad<0, true, true, D0, D1>, Exp<>>
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

/**
 * @brief SFPU pipeline: batched or per-tile streaming with DEST management and CB sync
 *
 * When batching is Auto (default), the pipeline automatically computes the maximum
 * number of chain iterations that fit in DEST (DEST_AUTO_LIMIT / chain_stride).
 * Each op's init() is called once and exec() is called batch_size times at
 * increasing DEST offsets. This amortizes init overhead across multiple tiles.
 *
 * When batching is Disabled, the pipeline processes one tile at a time (original behavior).
 *
 * Per batch iteration:
 *   1. tile_regs_acquire()
 *   2. For each tile in batch: Load chain tiles into DEST at offset k*stride
 *   3. For each compute op: init() once, exec(offset) for each tile in batch
 *   4. tile_regs_commit() / tile_regs_wait()
 *   5. pack_tile for each tile in batch
 *   6. tile_regs_release()
 *
 * @tparam batching       Batching mode: Auto (fill DEST) or Disabled (per-tile)
 * @tparam input_policy   How Load ops synchronize with input CBs (default: WaitAndPopPerTile)
 * @tparam output_policy  How output tiles are pushed (default: PerTile)
 * @tparam reconfig       Data format reconfiguration mode (default: INPUT_AND_OUTPUT)
 * @tparam Chain          SfpuChain<...> type (deduced)
 *
 * @param chain      The SfpuChain instance
 * @param ocb        Output circular buffer
 * @param num_tiles  Number of tiles to process
 * @param pack_slot  DEST slot offset to pack from (default: D0)
 */
template <
    SfpuBatching batching = SfpuBatching::Auto,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT,
    typename Chain>
ALWI void sfpu_pipeline(Chain chain, uint32_t ocb, uint32_t num_tiles, Dst pack_slot = Dst::D0);

// =============================================================================
// Convenience: Single Unary Op
// =============================================================================

/**
 * @brief Single unary SFPU op: load from ICB to DST[0], apply Op, pack to ocb
 *
 * This is the most common SFPU pattern. Equivalent to:
 *   sfpu_pipeline(sfpu_chain(Load<ICB, Dst::D0>{}, op), ocb, num_tiles);
 *
 * @tparam ICB           Input circular buffer (compile-time)
 * @tparam batching      Batching mode: Auto (fill DEST) or Disabled (per-tile)
 * @tparam input_policy  Input synchronization policy
 * @tparam output_policy Output synchronization policy
 * @tparam reconfig      Data format reconfiguration mode
 * @tparam Op            Op struct type (deduced from op parameter)
 *
 * @param ocb        Output circular buffer
 * @param num_tiles  Number of tiles to process
 * @param op         Op struct instance (default-constructed if parameterless)
 */
template <
    uint32_t ICB,
    SfpuBatching batching = SfpuBatching::Auto,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT,
    typename Op>
ALWI void sfpu_op(uint32_t ocb, uint32_t num_tiles, Op op);

// =============================================================================
// Named Convenience Alias Declarations
// =============================================================================

// All aliases below have the same template params:
//   <uint32_t ICB,
//    SfpuBatching batching = Auto,
//    SfpuInputPolicy input_policy = WaitAndPopPerTile,
//    SfpuOutputPolicy output_policy = PerTile,
//    SfpuDataFormatReconfig reconfig = INPUT_AND_OUTPUT>
// Signature: ALWI void sfpu_NAME(uint32_t ocb, uint32_t num_tiles);

// --- Math ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles);

// --- Activations ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sigmoid(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_tanh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gelu(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_silu(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_relu(uint32_t ocb, uint32_t num_tiles);

// --- Trigonometry ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sin(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_cos(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_tan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_asin(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_acos(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_atan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sinh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_cosh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_asinh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_acosh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_atanh(uint32_t ocb, uint32_t num_tiles);

// --- Error / Special Functions ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erf(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erfc(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erfinv(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_i0(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_i1(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_lgamma(uint32_t ocb, uint32_t num_tiles);

// --- Predicates ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isinf(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isnan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isfinite(uint32_t ocb, uint32_t num_tiles);

// --- Comparisons ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gtz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_ltz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_lez(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gez(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_eqz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_nez(uint32_t ocb, uint32_t num_tiles);

// --- Rounding ---
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_floor(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_ceil(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_trunc(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuBatching B = SfpuBatching::Auto,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_frac(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_helpers.inl"
