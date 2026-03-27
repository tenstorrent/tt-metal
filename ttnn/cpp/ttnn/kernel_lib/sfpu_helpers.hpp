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

/** @brief Base tag for compute ops — pipeline calls apply() on these */
struct ComputeTag {};

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
// Unary Group A — Simple Math (18 ops)
// =============================================================================

/**
 * @brief Exponential: DST[Slot] = exp(DST[Slot])
 * @tparam approx Approximation mode (default: Exact)
 * @tparam fast Fast+approximate mode (default: Fast)
 * @tparam Slot DEST slot to operate on
 * LLK: exp_tile_init<approx,fast>() -> exp_tile<approx,fast>(idst)
 */
template <Approx approx = Approx::Exact, Approx fast = Approx::Fast, Dst Slot = Dst::D0>
struct Exp : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { exp_tile_init<static_cast<bool>(approx), static_cast<bool>(fast)>(); }
    ALWI void exec() const { exp_tile<static_cast<bool>(approx), static_cast<bool>(fast)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Natural logarithm: DST[Slot] = log(DST[Slot])
 * @tparam approx Approximation mode (default: Exact)
 * @tparam Slot DEST slot to operate on
 * LLK: log_tile_init<approx>() -> log_tile<approx>(idst)
 */
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Log : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { log_tile_init<static_cast<bool>(approx)>(); }
    ALWI void exec() const { log_tile<static_cast<bool>(approx)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Logarithm with custom base: DST[Slot] = log_base(DST[Slot])
 * @tparam Slot DEST slot to operate on
 * @param base_scale IEEE754 bit-cast of 1/ln(base) (e.g. 0x3fb8aa3b for log2)
 * LLK: log_with_base_tile_init() -> log_with_base_tile(idst, base_scale)
 */
template <Dst Slot = Dst::D0>
struct LogWithBase : ComputeTag {
    uint32_t base_scale;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { log_with_base_tile_init(); }
    ALWI void exec() const { log_with_base_tile(dst_idx, base_scale); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Log(1 + x): DST[Slot] = log1p(DST[Slot])
 * @tparam fast_and_approx Fast approximation mode (default: false)
 * @tparam Slot DEST slot to operate on
 * LLK: log1p_tile_init<fast_and_approx>() -> log1p_tile<fast_and_approx>(idst)
 */
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Log1p : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { log1p_tile_init<static_cast<bool>(approx)>(); }
    ALWI void exec() const { log1p_tile<static_cast<bool>(approx)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Square root: DST[Slot] = sqrt(DST[Slot])
 * @tparam fast_approx Fast approximation mode (default: false)
 * @tparam Slot DEST slot to operate on
 * LLK: sqrt_tile_init() -> sqrt_tile<fast_approx>(idst)
 */
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Sqrt : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { sqrt_tile_init(); }
    ALWI void exec() const { sqrt_tile<static_cast<bool>(approx)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Reciprocal square root: DST[Slot] = 1/sqrt(DST[Slot])
 * @tparam legacy Legacy compatibility mode (default: Off)
 * @tparam approx Fast approximation mode (default: Exact)
 * @tparam Slot DEST slot to operate on
 * LLK: rsqrt_tile_init<legacy>() -> rsqrt_tile<legacy,approx>(idst)
 */
template <Legacy legacy = Legacy::Off, Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Rsqrt : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { rsqrt_tile_init<static_cast<bool>(legacy)>(); }
    ALWI void exec() const { rsqrt_tile<static_cast<bool>(legacy), static_cast<bool>(approx)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Cube root: DST[Slot] = cbrt(DST[Slot])
 * @tparam Slot DEST slot to operate on
 * LLK: cbrt_tile_init() -> cbrt_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Cbrt : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { cbrt_tile_init(); }
    ALWI void exec() const { cbrt_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Reciprocal: DST[Slot] = 1/DST[Slot]
 * @tparam legacy Legacy compatibility mode (default: On for recip)
 * @tparam Slot DEST slot to operate on
 * LLK: recip_tile_init<legacy>() -> recip_tile<legacy>(idst)
 */
template <Legacy legacy = Legacy::On, Dst Slot = Dst::D0>
struct Recip : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { recip_tile_init<static_cast<bool>(legacy)>(); }
    ALWI void exec() const { recip_tile<static_cast<bool>(legacy)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Absolute value: DST[Slot] = |DST[Slot]|
 * @tparam Slot DEST slot to operate on
 * LLK: abs_tile_init() -> abs_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Abs : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { abs_tile_init(); }
    ALWI void exec() const { abs_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Negation: DST[Slot] = -DST[Slot]
 * @tparam Slot DEST slot to operate on
 * LLK: negative_tile_init() -> negative_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Neg : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { negative_tile_init(); }
    ALWI void exec() const { negative_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Element-wise square: DST[Slot] = DST[Slot]^2
 * @tparam Slot DEST slot to operate on
 * LLK: square_tile_init() -> square_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Square : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { square_tile_init(); }
    ALWI void exec() const { square_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Sign function: DST[Slot] = sign(DST[Slot])
 * @tparam Slot DEST slot to operate on
 * LLK: sign_tile_init() -> sign_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Sign : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { sign_tile_init(); }
    ALWI void exec() const { sign_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Sign bit extraction: DST[Slot] = signbit(DST[Slot])
 * @tparam Slot DEST slot to operate on
 * LLK: signbit_tile_init() -> signbit_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Signbit : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { signbit_tile_init(); }
    ALWI void exec() const { signbit_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Base-2 exponential: DST[Slot] = 2^DST[Slot]
 * @tparam Slot DEST slot to operate on
 * LLK: exp2_tile_init() -> exp2_tile(idst)
 * Note: Hardcoded APPROX=true internally
 */
template <Dst Slot = Dst::D0>
struct Exp2 : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { exp2_tile_init(); }
    ALWI void exec() const { exp2_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Exponential minus one: DST[Slot] = exp(DST[Slot]) - 1
 * @tparam Slot DEST slot to operate on
 * LLK: expm1_tile_init() -> expm1_tile(idst)
 */
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Expm1 : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { expm1_tile_init<static_cast<bool>(approx)>(); }
    ALWI void exec() const { expm1_tile<static_cast<bool>(approx)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Power: DST[Slot] = DST[Slot]^exponent
 * @tparam Slot DEST slot to operate on
 * @param exponent IEEE754 bit-cast float exponent value
 * LLK: power_tile_init() -> power_tile(idst, param0)
 */
template <Dst Slot = Dst::D0>
struct Power : ComputeTag {
    uint32_t exponent;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { power_tile_init(); }
    ALWI void exec() const { power_tile(dst_idx, exponent); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Iterative power: DST[Slot] = DST[Slot]^int_exponent (integer exponent)
 * @tparam Slot DEST slot to operate on
 * @param int_exponent Integer exponent value (bit-cast as uint32_t)
 * LLK: power_tile_init() -> power_tile(idst, param0)
 * Note: Uses same SfpuType as Power; distinguished by integer exponent
 */
template <Dst Slot = Dst::D0>
struct PowerIterative : ComputeTag {
    uint32_t int_exponent;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { power_tile_init(); }
    ALWI void exec() const { power_tile(dst_idx, int_exponent); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Reverse power: DST[Slot] = base_val ^ DST[Slot]
 * @tparam Slot DEST slot to operate on
 * @param base_val IEEE754 bit-cast float base value
 * LLK: rpow_tile_init() -> rpow_tile(idst, base_val)
 * Semantics: base_val ^ tile[slot] (reversed from Power)
 */
template <Dst Slot = Dst::D0>
struct Rpow : ComputeTag {
    uint32_t base_val;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { rpow_tile_init(); }
    ALWI void exec() const { rpow_tile(dst_idx, base_val); }
    ALWI void apply() const {
        init();
        exec();
    }
};

// =============================================================================
// Unary Group B — Activations (11 ops)
// =============================================================================

/**
 * @brief Sigmoid activation: DST[Slot] = 1 / (1 + exp(-DST[Slot]))
 * @tparam Slot DEST slot to operate on
 * LLK: sigmoid_tile_init() -> sigmoid_tile(idst)
 * Note: vec_mode fixed to RC (standard mode)
 */
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Sigmoid : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { sigmoid_tile_init<static_cast<bool>(approx)>(); }
    ALWI void exec() const { sigmoid_tile<(int)VectorMode::RC, static_cast<bool>(approx)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Tanh activation: DST[Slot] = tanh(DST[Slot])
 * @tparam approx Approximation mode (default: Exact)
 * @tparam Slot DEST slot to operate on
 * LLK: tanh_tile_init<approx>() -> tanh_tile<approx>(idst)
 */
template <Approx approx = Approx::Exact, Dst Slot = Dst::D0>
struct Tanh : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { tanh_tile_init<static_cast<bool>(approx)>(); }
    ALWI void exec() const { tanh_tile<static_cast<bool>(approx)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief GELU activation: DST[Slot] = gelu(DST[Slot])
 * @tparam approx Approximation mode (default: Fast — unlike most other ops)
 * @tparam Slot DEST slot to operate on
 * LLK: gelu_tile_init<approx>() -> gelu_tile<approx>(idst)
 */
template <Approx approx = Approx::Fast, Dst Slot = Dst::D0>
struct Gelu : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { gelu_tile_init<static_cast<bool>(approx)>(); }
    ALWI void exec() const { gelu_tile<static_cast<bool>(approx)>(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief SiLU (Swish) activation: DST[Slot] = DST[Slot] * sigmoid(DST[Slot])
 * @tparam Slot DEST slot to operate on
 * LLK: silu_tile_init() -> silu_tile(idst)
 * Note: Always uses non-approx sigmoid internally
 */
template <Dst Slot = Dst::D0>
struct Silu : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { silu_tile_init(); }
    ALWI void exec() const { silu_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief ReLU activation: DST[Slot] = max(0, DST[Slot])
 * @tparam Slot DEST slot to operate on
 * LLK: relu_tile_init() -> relu_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Relu : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { relu_tile_init(); }
    ALWI void exec() const { relu_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Hard Mish activation
 * @tparam Slot DEST slot to operate on
 * LLK: hardmish_tile_init() -> hardmish_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Hardmish : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { hardmish_tile_init(); }
    ALWI void exec() const { hardmish_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Hard Sigmoid activation
 * @tparam Slot DEST slot to operate on
 * LLK: hardsigmoid_tile_init() -> hardsigmoid_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Hardsigmoid : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { hardsigmoid_tile_init(); }
    ALWI void exec() const { hardsigmoid_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Hard Tanh activation: clamp(DST[Slot], min, max)
 * @tparam Slot DEST slot to operate on
 * @param param_min IEEE754 bit-cast float minimum
 * @param param_max IEEE754 bit-cast float maximum
 * LLK: hardtanh_tile_init() -> hardtanh_tile(idst, param0, param1)
 */
template <Dst Slot = Dst::D0>
struct Hardtanh : ComputeTag {
    uint32_t param_min;
    uint32_t param_max;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { hardtanh_tile_init(); }
    ALWI void exec() const { hardtanh_tile(dst_idx, param_min, param_max); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Softsign activation: DST[Slot] = DST[Slot] / (1 + |DST[Slot]|)
 * @tparam Slot DEST slot to operate on
 * LLK: softsign_tile_init() -> softsign_tile(idst)
 */
template <Dst Slot = Dst::D0>
struct Softsign : ComputeTag {
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { softsign_tile_init(); }
    ALWI void exec() const { softsign_tile(dst_idx); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief Softplus activation: DST[Slot] = (1/beta) * log(1 + exp(beta * DST[Slot]))
 * @tparam Slot DEST slot to operate on
 * @param beta IEEE754 bit-cast float beta parameter
 * @param beta_recip IEEE754 bit-cast float 1/beta
 * @param threshold IEEE754 bit-cast float threshold for linear fallback
 * LLK: softplus_tile_init() -> softplus_tile(idst, beta, beta_reciprocal, threshold)
 */
template <Dst Slot = Dst::D0>
struct Softplus : ComputeTag {
    uint32_t beta;
    uint32_t beta_recip;
    uint32_t threshold;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { softplus_tile_init(); }
    ALWI void exec() const { softplus_tile(dst_idx, beta, beta_recip, threshold); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief XIeLU activation (parametric leaky ReLU variant)
 * @tparam Slot DEST slot to operate on
 * @param alpha_p IEEE754 bit-cast float positive slope
 * @param alpha_n IEEE754 bit-cast float negative slope
 * LLK: xielu_tile_init() -> xielu_tile(idst, alpha_p, alpha_n)
 */
template <Dst Slot = Dst::D0>
struct Xielu : ComputeTag {
    uint32_t alpha_p;
    uint32_t alpha_n;
    static constexpr uint32_t dst_idx = static_cast<uint32_t>(Slot);
    static_assert(dst_idx < 8, "DEST slot exceeds maximum capacity (8)");
    ALWI void init() const { xielu_tile_init(); }
    ALWI void exec() const { xielu_tile(dst_idx, alpha_p, alpha_n); }
    ALWI void apply() const {
        init();
        exec();
    }
};

// =============================================================================
// Binary SFPU Group A — Arithmetic (6 ops)
// =============================================================================

/**
 * @brief SFPU binary add: DST[Out] = DST[In0] + DST[In1]
 * @tparam In0 First input DEST slot
 * @tparam In1 Second input DEST slot
 * @tparam Out Output DEST slot
 * LLK: add_binary_tile_init() -> add_binary_tile(idst0, idst1, odst)
 */
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuAdd : ComputeTag {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const { add_binary_tile_init(); }
    ALWI void exec() const { add_binary_tile(in0, in1, out); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief SFPU binary subtract: DST[Out] = DST[In0] - DST[In1]
 * @tparam In0 First input DEST slot
 * @tparam In1 Second input DEST slot
 * @tparam Out Output DEST slot
 * LLK: sub_binary_tile_init() -> sub_binary_tile(idst0, idst1, odst)
 */
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuSub : ComputeTag {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const { sub_binary_tile_init(); }
    ALWI void exec() const { sub_binary_tile(in0, in1, out); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief SFPU binary multiply: DST[Out] = DST[In0] * DST[In1]
 * @tparam In0 First input DEST slot
 * @tparam In1 Second input DEST slot
 * @tparam Out Output DEST slot
 * LLK: mul_binary_tile_init() -> mul_binary_tile(idst0, idst1, odst)
 */
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMul : ComputeTag {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const { mul_binary_tile_init(); }
    ALWI void exec() const { mul_binary_tile(in0, in1, out); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief SFPU binary divide: DST[Out] = DST[In0] / DST[In1]
 * @tparam In0 First input DEST slot (numerator)
 * @tparam In1 Second input DEST slot (denominator)
 * @tparam Out Output DEST slot
 * LLK: div_binary_tile_init() -> div_binary_tile(idst0, idst1, odst)
 */
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuDiv : ComputeTag {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const { div_binary_tile_init(); }
    ALWI void exec() const { div_binary_tile(in0, in1, out); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief SFPU binary reverse subtract: DST[Out] = DST[In1] - DST[In0]
 * @tparam In0 First input DEST slot
 * @tparam In1 Second input DEST slot
 * @tparam Out Output DEST slot
 * LLK: rsub_binary_tile_init() -> rsub_binary_tile(idst0, idst1, odst)
 */
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuRsub : ComputeTag {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const { rsub_binary_tile_init(); }
    ALWI void exec() const { rsub_binary_tile(in0, in1, out); }
    ALWI void apply() const {
        init();
        exec();
    }
};

/**
 * @brief SFPU binary power: DST[Out] = DST[In0] ^ DST[In1]
 * @tparam In0 Base DEST slot
 * @tparam In1 Exponent DEST slot
 * @tparam Out Output DEST slot
 * LLK: power_binary_tile_init() -> power_binary_tile(idst0, idst1, odst)
 */
template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuPow : ComputeTag {
    static constexpr uint32_t in0 = static_cast<uint32_t>(In0);
    static constexpr uint32_t in1 = static_cast<uint32_t>(In1);
    static constexpr uint32_t out = static_cast<uint32_t>(Out);
    static_assert(in0 < 8, "DEST slot In0 exceeds maximum capacity (8)");
    static_assert(in1 < 8, "DEST slot In1 exceeds maximum capacity (8)");
    static_assert(out < 8, "DEST slot Out exceeds maximum capacity (8)");
    ALWI void init() const { power_binary_tile_init(); }
    ALWI void exec() const { power_binary_tile(in0, in1, out); }
    ALWI void apply() const {
        init();
        exec();
    }
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
// Named Convenience Aliases
// =============================================================================

/** @brief Exponential on all tiles. See sfpu_op() for policy documentation. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles);

/** @brief Natural logarithm on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles);

/** @brief Log(1+x) on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles);

/** @brief Square root on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles);

/** @brief Reciprocal square root on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles);

/** @brief Reciprocal on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles);

/** @brief Absolute value on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles);

/** @brief Negation on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles);

/** @brief Sigmoid activation on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_sigmoid(uint32_t ocb, uint32_t num_tiles);

/** @brief Tanh activation on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_tanh(uint32_t ocb, uint32_t num_tiles);

/** @brief GELU activation on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_gelu(uint32_t ocb, uint32_t num_tiles);

/** @brief SiLU (Swish) activation on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_silu(uint32_t ocb, uint32_t num_tiles);

/** @brief ReLU activation on all tiles. */
template <
    uint32_t ICB,
    SfpuInputPolicy input_policy = SfpuInputPolicy::WaitAndPopPerTile,
    SfpuOutputPolicy output_policy = SfpuOutputPolicy::PerTile,
    SfpuDataFormatReconfig reconfig = SfpuDataFormatReconfig::INPUT_AND_OUTPUT>
ALWI void sfpu_relu(uint32_t ocb, uint32_t num_tiles);

}  // namespace compute_kernel_lib

#include "sfpu_helpers.inl"
