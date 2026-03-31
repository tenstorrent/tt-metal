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
#include "api/compute/eltwise_unary/bitwise_and.h"
#include "api/compute/eltwise_unary/bitwise_or.h"
#include "api/compute/eltwise_unary/bitwise_xor.h"
#include "api/compute/eltwise_unary/bitwise_not.h"
#include "api/compute/eltwise_unary/left_shift.h"
#include "api/compute/eltwise_unary/right_shift.h"
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
 * - **Dst enum**: Self-documenting DEST slot names (D0..D7), used as template parameters
 * - **Load<CB, Slot>**: Copies a tile from circular buffer CB into DEST[Slot]
 * - **Op structs**: Each wraps a single SFPU LLK call pair (init + exec)
 * - **SfpuChain**: Variadic combinator that stores a sequence of Load + compute ops
 * - **sfpu_pipeline()**: Per-tile streaming loop with DEST acquire/commit/pack
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
 *   // 7. WaitUpfrontNoPop — tiles persist in CB for reuse
 *   sfpu_op<cb_in, SfpuInputPolicy::WaitUpfrontNoPop>(cb_out, num_tiles, Exp<>{});
 *
 *   // 8. Skip data format reconfiguration
 *   sfpu_op<cb_in, SfpuInputPolicy::WaitAndPopPerTile,
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

// =============================================================================
// Load Op
// =============================================================================

/**
 * @brief Load a tile from circular buffer CB into DEST[Slot]
 *
 * The pipeline handles copy_tile() and _with_dt tracking for Loads.
 * Multiple Loads from the same CB share a single wait (streaming mode).
 *
 * @tparam CB   Compile-time circular buffer index
 * @tparam Slot DEST register to copy into
 */
template <uint32_t CB, Dst Slot>
struct Load : LoadTag {
    static constexpr uint32_t cb = CB;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(static_cast<uint32_t>(Slot) < 8, "DEST slot exceeds maximum capacity (8)");
};

// =============================================================================
// Op Struct Declarations
// =============================================================================

// --- Simple Math ---

template <Approx approx = Approx::Exact, Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Exp {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Log {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct LogWithBase {
    uint32_t base_scale;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Log1p {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Sqrt {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Legacy legacy = Legacy::Off, Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Rsqrt {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Cbrt {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Legacy legacy = Legacy::On, Dst Slot = Dst::D0>
struct Recip {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Abs {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Neg {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Square {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Sign {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Signbit {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Exp2 {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Expm1 {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Power {
    uint32_t exponent;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct PowerIterative {
    uint32_t int_exponent;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Rpow {
    uint32_t base_val;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Activations ---

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Sigmoid {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Tanh {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Gelu {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Silu {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Relu {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Hardmish {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Hardsigmoid {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Hardtanh {
    uint32_t param_min;
    uint32_t param_max;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Softsign {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Softplus {
    uint32_t beta;
    uint32_t beta_recip;
    uint32_t threshold;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Xielu {
    uint32_t alpha_p;
    uint32_t alpha_n;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Trigonometry ---

template <Dst Slot = Dst::D0>
struct Sin {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Cos {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Tan {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Asin {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Acos {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Atan {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Sinh {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Cosh {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Asinh {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Acosh {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Atanh {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Error / Special Functions ---

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Erf {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Erfc {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Erfinv {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct I0 {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct I1 {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Lgamma {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Predicates & Comparisons ---

template <Dst Slot = Dst::D0>
struct Isinf {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Isposinf {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Isneginf {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Isnan {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Isfinite {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <DataFormat df = DataFormat::Float16_b, Dst Slot = Dst::D0>
struct LogicalNot {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Gtz {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Ltz {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Lez {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Gez {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Eqz {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Nez {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct UnaryEq {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct UnaryNe {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct UnaryGt {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct UnaryGe {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct UnaryLt {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct UnaryLe {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Additional Activations ---

template <Dst Slot = Dst::D0>
struct Elu {
    uint32_t alpha;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Selu {
    uint32_t scale;
    uint32_t alpha;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Celu {
    uint32_t alpha;
    uint32_t alpha_recip;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Softshrink {
    uint32_t lambda;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Clamp {
    uint32_t param_min;
    uint32_t param_max;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Threshold {
    uint32_t threshold;
    uint32_t value;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Prelu {
    uint32_t weight;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Rounding ---

template <Dst Slot = Dst::D0>
struct Floor {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Ceil {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Trunc {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Round {
    int32_t decimals;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Frac {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct StochasticRound {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Type / Identity / Bitwise ---

template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot = Dst::D0>
struct Typecast {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Identity {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct BitwiseNot {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct BitwiseAnd {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct BitwiseOr {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct BitwiseXor {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct LeftShift {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct RightShift {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Scalar Arithmetic ---

template <Dst Slot = Dst::D0>
struct AddScalar {
    uint32_t scalar;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct SubScalar {
    uint32_t scalar;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct MulScalar {
    uint32_t scalar;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct DivScalar {
    uint32_t scalar;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct RsubScalar {
    uint32_t scalar;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Rsub {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <RoundingMode rounding_mode = RoundingMode::None, Dst Slot = Dst::D0>
struct Rdiv {
    uint32_t value;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Fmod {
    uint32_t param0;
    uint32_t param1;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Remainder {
    uint32_t param0;
    uint32_t param1;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct Dropout {
    uint32_t probability;
    uint32_t scale_factor;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Fill and Random ---

template <Dst Slot = Dst::D0>
struct FillTile {
    float fill_val;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct FillTileBitcast {
    uint32_t param0;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst Slot = Dst::D0>
struct RandTile {
    uint32_t from;
    uint32_t scale;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Binary SFPU Ops ---

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuAdd {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuSub {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMul {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuDiv {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuRsub {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuPow {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuEq {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// --- Ternary SFPU Ops ---

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Where {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t in2 = static_cast<uint32_t>(In2);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8 && in1 < 8 && in2 < 8 && out < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Lerp {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t in2 = static_cast<uint32_t>(In2);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8 && in1 < 8 && in2 < 8 && out < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Addcmul {
    uint32_t value;
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t in2 = static_cast<uint32_t>(In2);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8 && in1 < 8 && in2 < 8 && out < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

template <
    DataFormat df = DataFormat::Float16_b,
    Dst In0 = Dst::D0,
    Dst In1 = Dst::D1,
    Dst In2 = Dst::D2,
    Dst Out = Dst::D0>
struct Addcdiv {
    uint32_t value;
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t in2 = static_cast<uint32_t>(In2);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8 && in1 < 8 && in2 < 8 && out < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const;
    ALWI void exec() const;
    ALWI void apply() const;
};

// =============================================================================
// SfpuChain Combinator
// =============================================================================

/**
 * @brief Variadic chain of SFPU ops (Load + compute)
 *
 * Stores all ops by value using recursive template instantiation.
 * - apply(): Calls apply() on each compute op in sequence, skipping Loads
 * - for_each_load(): Iterates over Load ops for the pipeline's load phase
 *
 * Zero runtime overhead — all dispatch is resolved at compile time via if constexpr.
 */
template <typename... Ops>
struct SfpuChain;

/** @brief Base case: empty chain */
template <>
struct SfpuChain<> {
    ALWI void apply() const {}

    template <typename F>
    ALWI void for_each_load(F&&) const {}
};

/** @brief Recursive case: first op + rest of chain */
template <typename First, typename... Rest>
struct SfpuChain<First, Rest...> {
    First first;
    SfpuChain<Rest...> rest;

    constexpr SfpuChain(First f, Rest... r) : first(f), rest(r...) {}

    /** @brief Execute all compute ops in sequence (skips Loads) */
    ALWI void apply() const {
        if constexpr (!is_load_op_v<First>) {
            first.apply();
        }
        rest.apply();
    }

    /** @brief Iterate over all Load ops, calling f(load) for each */
    template <typename F>
    ALWI void for_each_load(F&& f) const {
        if constexpr (is_load_op_v<First>) {
            f(first);
        }
        rest.for_each_load(static_cast<F&&>(f));
    }
};

/**
 * @brief Factory function — deduces SfpuChain template parameters
 *
 * Usage: auto chain = sfpu_chain(Load<0, Dst::D0>{}, Exp<Dst::D0>{}, Log<Dst::D0>{});
 */
template <typename... Ops>
constexpr ALWI SfpuChain<Ops...> sfpu_chain(Ops... ops) {
    return SfpuChain<Ops...>(ops...);
}

// =============================================================================
// Pipeline Function Declaration
// =============================================================================

/**
 * @brief SFPU pipeline: per-tile streaming with DEST management and CB sync
 *
 * For each tile:
 *   1. tile_regs_acquire()
 *   2. For each Load in chain: copy_tile(cb, tile_idx, dst_slot) with _with_dt tracking
 *   3. For each compute op in chain: op.apply() (init + exec)
 *   4. tile_regs_commit() / tile_regs_wait()
 *   5. pack_tile(pack_slot, ocb)
 *   6. tile_regs_release()
 *
 * @tparam input_policy   How Load ops synchronize with input CBs (default: WaitAndPopPerTile)
 * @tparam output_policy  How output tiles are pushed (default: PerTile)
 * @tparam reconfig       Data format reconfiguration mode (default: INPUT_AND_OUTPUT)
 * @tparam Chain          SfpuChain<...> type (deduced)
 *
 * @param chain      The SfpuChain instance
 * @param ocb        Output circular buffer
 * @param num_tiles  Number of tiles to process
 * @param pack_slot  DEST slot to pack from (default: D0, typically the slot where the final result lands)
 */
template <
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
//    SfpuInputPolicy input_policy = WaitAndPopPerTile,
//    SfpuOutputPolicy output_policy = PerTile,
//    SfpuDataFormatReconfig reconfig = INPUT_AND_OUTPUT>
// Signature: ALWI void sfpu_NAME(uint32_t ocb, uint32_t num_tiles);

// --- Math ---
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles);

// --- Activations ---
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sigmoid(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_tanh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gelu(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_silu(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_relu(uint32_t ocb, uint32_t num_tiles);

// --- Trigonometry ---
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sin(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_cos(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_tan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_asin(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_acos(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_atan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sinh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_cosh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_asinh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_acosh(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_atanh(uint32_t ocb, uint32_t num_tiles);

// --- Error / Special Functions ---
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erf(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erfc(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_erfinv(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_i0(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_i1(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_lgamma(uint32_t ocb, uint32_t num_tiles);

// --- Predicates ---
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isinf(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isnan(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_isfinite(uint32_t ocb, uint32_t num_tiles);

// --- Comparisons ---
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gtz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_ltz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_lez(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gez(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_eqz(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_nez(uint32_t ocb, uint32_t num_tiles);

// --- Rounding ---
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_floor(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_ceil(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_trunc(uint32_t ocb, uint32_t num_tiles);
template <
    uint32_t ICB,
    SfpuInputPolicy P = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy O = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig R = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_frac(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_helpers.inl"
