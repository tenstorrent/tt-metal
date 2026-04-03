// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file sfpu_helpers.inl
 * @brief Out-of-line method definitions for SFPU op structs, pipeline, and convenience functions
 *
 * This file contains all op struct method definitions, internal helpers, the pipeline
 * implementation, sfpu_op(), and named convenience alias implementations.
 * It should only be included by sfpu_helpers.hpp.
 */

#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
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
// NOTE: bitwise/shift headers excluded — see sfpu_helpers.hpp for rationale.
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
#include "api/debug/assert.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

using namespace ckernel;

// =============================================================================
// Op Method Definitions — Simple Math (18 ops)
// =============================================================================

template <Approx approx, Approx fast, Dst Slot>
ALWI void Exp<approx, fast, Slot>::init() const { exp_tile_init<static_cast<bool>(approx), static_cast<bool>(fast)>(); }
template <Approx approx, Approx fast, Dst Slot>
ALWI void Exp<approx, fast, Slot>::call(uint32_t d0) const { exp_tile<static_cast<bool>(approx), static_cast<bool>(fast)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Log<approx, Slot>::init() const { log_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Log<approx, Slot>::call(uint32_t d0) const { log_tile<static_cast<bool>(approx)>(d0); }

template <Dst Slot>
ALWI void LogWithBase<Slot>::init() const { log_with_base_tile_init(); }
template <Dst Slot>
ALWI void LogWithBase<Slot>::call(uint32_t d0) const { log_with_base_tile(d0, base_scale); }

template <Approx approx, Dst Slot>
ALWI void Log1p<approx, Slot>::init() const { log1p_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Log1p<approx, Slot>::call(uint32_t d0) const { log1p_tile<static_cast<bool>(approx)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Sqrt<approx, Slot>::init() const { sqrt_tile_init(); }
template <Approx approx, Dst Slot>
ALWI void Sqrt<approx, Slot>::call(uint32_t d0) const { sqrt_tile<static_cast<bool>(approx)>(d0); }

template <Legacy legacy, Approx approx, Dst Slot>
ALWI void Rsqrt<legacy, approx, Slot>::init() const { rsqrt_tile_init<static_cast<bool>(legacy)>(); }
template <Legacy legacy, Approx approx, Dst Slot>
ALWI void Rsqrt<legacy, approx, Slot>::call(uint32_t d0) const { rsqrt_tile<static_cast<bool>(legacy), static_cast<bool>(approx)>(d0); }

template <Dst Slot>
ALWI void Cbrt<Slot>::init() const { cbrt_tile_init(); }
template <Dst Slot>
ALWI void Cbrt<Slot>::call(uint32_t d0) const { cbrt_tile(d0); }

template <Legacy legacy, Dst Slot>
ALWI void Recip<legacy, Slot>::init() const { recip_tile_init<static_cast<bool>(legacy)>(); }
template <Legacy legacy, Dst Slot>
ALWI void Recip<legacy, Slot>::call(uint32_t d0) const { recip_tile<static_cast<bool>(legacy)>(d0); }

template <Dst Slot>
ALWI void Abs<Slot>::init() const { abs_tile_init(); }
template <Dst Slot>
ALWI void Abs<Slot>::call(uint32_t d0) const { abs_tile(d0); }

template <Dst Slot>
ALWI void Neg<Slot>::init() const { negative_tile_init(); }
template <Dst Slot>
ALWI void Neg<Slot>::call(uint32_t d0) const { negative_tile(d0); }

template <Dst Slot>
ALWI void Square<Slot>::init() const { square_tile_init(); }
template <Dst Slot>
ALWI void Square<Slot>::call(uint32_t d0) const { square_tile(d0); }

template <Dst Slot>
ALWI void Sign<Slot>::init() const { sign_tile_init(); }
template <Dst Slot>
ALWI void Sign<Slot>::call(uint32_t d0) const { sign_tile(d0); }

template <Dst Slot>
ALWI void Signbit<Slot>::init() const { signbit_tile_init(); }
template <Dst Slot>
ALWI void Signbit<Slot>::call(uint32_t d0) const { signbit_tile(d0); }

template <Dst Slot>
ALWI void Exp2<Slot>::init() const { exp2_tile_init(); }
template <Dst Slot>
ALWI void Exp2<Slot>::call(uint32_t d0) const { exp2_tile(d0); }

template <Approx approx, Dst Slot>
ALWI void Expm1<approx, Slot>::init() const { expm1_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Expm1<approx, Slot>::call(uint32_t d0) const { expm1_tile<static_cast<bool>(approx)>(d0); }

template <Dst Slot>
ALWI void Power<Slot>::init() const { power_tile_init(); }
template <Dst Slot>
ALWI void Power<Slot>::call(uint32_t d0) const { power_tile(d0, exponent); }

template <Dst Slot>
ALWI void PowerIterative<Slot>::init() const { power_tile_init(); }
template <Dst Slot>
ALWI void PowerIterative<Slot>::call(uint32_t d0) const { power_tile(d0, int_exponent); }

template <Dst Slot>
ALWI void Rpow<Slot>::init() const { rpow_tile_init(); }
template <Dst Slot>
ALWI void Rpow<Slot>::call(uint32_t d0) const { rpow_tile(d0, base_val); }

// =============================================================================
// Op Method Definitions — Activations (11 ops)
// =============================================================================

template <Approx approx, Dst Slot>
ALWI void Sigmoid<approx, Slot>::init() const { sigmoid_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Sigmoid<approx, Slot>::call(uint32_t d0) const { sigmoid_tile<(int)VectorMode::RC, static_cast<bool>(approx)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Tanh<approx, Slot>::init() const { tanh_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Tanh<approx, Slot>::call(uint32_t d0) const { tanh_tile<static_cast<bool>(approx)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Gelu<approx, Slot>::init() const { gelu_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Gelu<approx, Slot>::call(uint32_t d0) const { gelu_tile<static_cast<bool>(approx)>(d0); }

template <Dst Slot>
ALWI void Silu<Slot>::init() const { silu_tile_init(); }
template <Dst Slot>
ALWI void Silu<Slot>::call(uint32_t d0) const { silu_tile(d0); }

template <Dst Slot>
ALWI void Relu<Slot>::init() const { relu_tile_init(); }
template <Dst Slot>
ALWI void Relu<Slot>::call(uint32_t d0) const { relu_tile(d0); }

template <Dst Slot>
ALWI void Hardmish<Slot>::init() const { hardmish_tile_init(); }
template <Dst Slot>
ALWI void Hardmish<Slot>::call(uint32_t d0) const { hardmish_tile(d0); }

template <Dst Slot>
ALWI void Hardsigmoid<Slot>::init() const { hardsigmoid_tile_init(); }
template <Dst Slot>
ALWI void Hardsigmoid<Slot>::call(uint32_t d0) const { hardsigmoid_tile(d0); }

template <Dst Slot>
ALWI void Hardtanh<Slot>::init() const { hardtanh_tile_init(); }
template <Dst Slot>
ALWI void Hardtanh<Slot>::call(uint32_t d0) const { hardtanh_tile(d0, param_min, param_max); }

template <Dst Slot>
ALWI void Softsign<Slot>::init() const { softsign_tile_init(); }
template <Dst Slot>
ALWI void Softsign<Slot>::call(uint32_t d0) const { softsign_tile(d0); }

template <Dst Slot>
ALWI void Softplus<Slot>::init() const { softplus_tile_init(); }
template <Dst Slot>
ALWI void Softplus<Slot>::call(uint32_t d0) const { softplus_tile(d0, beta, beta_recip, threshold); }

template <Dst Slot>
ALWI void Xielu<Slot>::init() const { xielu_tile_init(); }
template <Dst Slot>
ALWI void Xielu<Slot>::call(uint32_t d0) const { xielu_tile(d0, alpha_p, alpha_n); }

// =============================================================================
// Op Method Definitions — Trigonometry (11 ops)
// =============================================================================

template <Dst Slot> ALWI void Sin<Slot>::init() const { sin_tile_init(); }
template <Dst Slot> ALWI void Sin<Slot>::call(uint32_t d0) const { sin_tile(d0); }
template <Dst Slot> ALWI void Cos<Slot>::init() const { cos_tile_init(); }
template <Dst Slot> ALWI void Cos<Slot>::call(uint32_t d0) const { cos_tile(d0); }
template <Dst Slot> ALWI void Tan<Slot>::init() const { tan_tile_init(); }
template <Dst Slot> ALWI void Tan<Slot>::call(uint32_t d0) const { tan_tile(d0); }
template <Dst Slot> ALWI void Asin<Slot>::init() const { asin_tile_init(); }
template <Dst Slot> ALWI void Asin<Slot>::call(uint32_t d0) const { asin_tile(d0); }
template <Dst Slot> ALWI void Acos<Slot>::init() const { acos_tile_init(); }
template <Dst Slot> ALWI void Acos<Slot>::call(uint32_t d0) const { acos_tile(d0); }
template <Dst Slot> ALWI void Atan<Slot>::init() const { atan_tile_init(); }
template <Dst Slot> ALWI void Atan<Slot>::call(uint32_t d0) const { atan_tile(d0); }
template <Dst Slot> ALWI void Sinh<Slot>::init() const { sinh_tile_init(); }
template <Dst Slot> ALWI void Sinh<Slot>::call(uint32_t d0) const { sinh_tile(d0); }
template <Dst Slot> ALWI void Cosh<Slot>::init() const { cosh_tile_init(); }
template <Dst Slot> ALWI void Cosh<Slot>::call(uint32_t d0) const { cosh_tile(d0); }
template <Dst Slot> ALWI void Asinh<Slot>::init() const { asinh_tile_init(); }
template <Dst Slot> ALWI void Asinh<Slot>::call(uint32_t d0) const { asinh_tile(d0); }
template <Dst Slot> ALWI void Acosh<Slot>::init() const { acosh_tile_init(); }
template <Dst Slot> ALWI void Acosh<Slot>::call(uint32_t d0) const { acosh_tile(d0); }
template <Dst Slot> ALWI void Atanh<Slot>::init() const { atanh_tile_init(); }
template <Dst Slot> ALWI void Atanh<Slot>::call(uint32_t d0) const { atanh_tile(d0); }

// =============================================================================
// Op Method Definitions — Error / Special Functions (6 ops)
// =============================================================================

template <Approx approx, Dst Slot>
ALWI void Erf<approx, Slot>::init() const { erf_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Erf<approx, Slot>::call(uint32_t d0) const { erf_tile<static_cast<bool>(approx)>(d0); }

template <Approx approx, Dst Slot>
ALWI void Erfc<approx, Slot>::init() const { erfc_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Erfc<approx, Slot>::call(uint32_t d0) const { erfc_tile<static_cast<bool>(approx)>(d0); }

template <Dst Slot> ALWI void Erfinv<Slot>::init() const { erfinv_tile_init(); }
template <Dst Slot> ALWI void Erfinv<Slot>::call(uint32_t d0) const { erfinv_tile(d0); }
template <Dst Slot> ALWI void I0<Slot>::init() const { i0_tile_init(); }
template <Dst Slot> ALWI void I0<Slot>::call(uint32_t d0) const { i0_tile(d0); }
template <Dst Slot> ALWI void I1<Slot>::init() const { i1_tile_init(); }
template <Dst Slot> ALWI void I1<Slot>::call(uint32_t d0) const { i1_tile(d0); }
template <Dst Slot> ALWI void Lgamma<Slot>::init() const { lgamma_stirling_tile_init(); }
template <Dst Slot> ALWI void Lgamma<Slot>::call(uint32_t d0) const { lgamma_stirling_tile(d0); }

// =============================================================================
// Op Method Definitions — Predicates and Comparisons (18 ops)
// =============================================================================

template <Dst Slot> ALWI void Isinf<Slot>::init() const { isinf_tile_init(); }
template <Dst Slot> ALWI void Isinf<Slot>::call(uint32_t d0) const { isinf_tile(d0); }
template <Dst Slot> ALWI void Isposinf<Slot>::init() const { isposinf_tile_init(); }
template <Dst Slot> ALWI void Isposinf<Slot>::call(uint32_t d0) const { isposinf_tile(d0); }
template <Dst Slot> ALWI void Isneginf<Slot>::init() const { isneginf_tile_init(); }
template <Dst Slot> ALWI void Isneginf<Slot>::call(uint32_t d0) const { isneginf_tile(d0); }
template <Dst Slot> ALWI void Isnan<Slot>::init() const { isnan_tile_init(); }
template <Dst Slot> ALWI void Isnan<Slot>::call(uint32_t d0) const { isnan_tile(d0); }
template <Dst Slot> ALWI void Isfinite<Slot>::init() const { isfinite_tile_init(); }
template <Dst Slot> ALWI void Isfinite<Slot>::call(uint32_t d0) const { isfinite_tile(d0); }

template <DataFormat df, Dst Slot>
ALWI void LogicalNot<df, Slot>::init() const { logical_not_tile_init(); }
template <DataFormat df, Dst Slot>
ALWI void LogicalNot<df, Slot>::call(uint32_t d0) const { logical_not_tile<df>(d0); }

template <Dst Slot> ALWI void Gtz<Slot>::init() const { gtz_tile_init(); }
template <Dst Slot> ALWI void Gtz<Slot>::call(uint32_t d0) const { gtz_tile(d0); }
template <Dst Slot> ALWI void Ltz<Slot>::init() const { ltz_tile_init(); }
template <Dst Slot> ALWI void Ltz<Slot>::call(uint32_t d0) const { ltz_tile(d0); }
template <Dst Slot> ALWI void Lez<Slot>::init() const { lez_tile_init(); }
template <Dst Slot> ALWI void Lez<Slot>::call(uint32_t d0) const { lez_tile(d0); }
template <Dst Slot> ALWI void Gez<Slot>::init() const { gez_tile_init(); }
template <Dst Slot> ALWI void Gez<Slot>::call(uint32_t d0) const { gez_tile(d0); }
template <Dst Slot> ALWI void Eqz<Slot>::init() const { eqz_tile_init(); }
template <Dst Slot> ALWI void Eqz<Slot>::call(uint32_t d0) const { eqz_tile(d0); }
template <Dst Slot> ALWI void Nez<Slot>::init() const { nez_tile_init(); }
template <Dst Slot> ALWI void Nez<Slot>::call(uint32_t d0) const { nez_tile(d0); }

template <Dst Slot> ALWI void UnaryEq<Slot>::init() const { unary_eq_tile_init(); }
template <Dst Slot> ALWI void UnaryEq<Slot>::call(uint32_t d0) const { unary_eq_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryNe<Slot>::init() const { unary_ne_tile_init(); }
template <Dst Slot> ALWI void UnaryNe<Slot>::call(uint32_t d0) const { unary_ne_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryGt<Slot>::init() const { unary_gt_tile_init(); }
template <Dst Slot> ALWI void UnaryGt<Slot>::call(uint32_t d0) const { unary_gt_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryGe<Slot>::init() const { unary_ge_tile_init(); }
template <Dst Slot> ALWI void UnaryGe<Slot>::call(uint32_t d0) const { unary_ge_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryLt<Slot>::init() const { unary_lt_tile_init(); }
template <Dst Slot> ALWI void UnaryLt<Slot>::call(uint32_t d0) const { unary_lt_tile(d0, param0); }
template <Dst Slot> ALWI void UnaryLe<Slot>::init() const { unary_le_tile_init(); }
template <Dst Slot> ALWI void UnaryLe<Slot>::call(uint32_t d0) const { unary_le_tile(d0, param0); }

// =============================================================================
// Op Method Definitions — Additional Activations (7 ops)
// =============================================================================

template <Dst Slot> ALWI void Elu<Slot>::init() const { elu_tile_init(); }
template <Dst Slot> ALWI void Elu<Slot>::call(uint32_t d0) const { elu_tile(d0, alpha); }
template <Dst Slot> ALWI void Selu<Slot>::init() const { selu_tile_init(); }
template <Dst Slot> ALWI void Selu<Slot>::call(uint32_t d0) const { selu_tile(d0, scale, alpha); }
template <Dst Slot> ALWI void Celu<Slot>::init() const { celu_tile_init(); }
template <Dst Slot> ALWI void Celu<Slot>::call(uint32_t d0) const { celu_tile(d0, alpha, alpha_recip); }
template <Dst Slot> ALWI void Softshrink<Slot>::init() const { softshrink_tile_init(); }
template <Dst Slot> ALWI void Softshrink<Slot>::call(uint32_t d0) const { softshrink_tile(d0, lambda); }
template <Dst Slot> ALWI void Clamp<Slot>::init() const { clamp_tile_init(); }
template <Dst Slot> ALWI void Clamp<Slot>::call(uint32_t d0) const { clamp_tile(d0, param_min, param_max); }
template <Dst Slot> ALWI void Threshold<Slot>::init() const { threshold_tile_init(); }
template <Dst Slot> ALWI void Threshold<Slot>::call(uint32_t d0) const { threshold_tile(d0, threshold, value); }
template <Dst Slot> ALWI void Prelu<Slot>::init() const { prelu_tile_init(); }
template <Dst Slot> ALWI void Prelu<Slot>::call(uint32_t d0) const { prelu_tile(d0, weight); }

// =============================================================================
// Op Method Definitions — Rounding (6 ops)
// =============================================================================

template <Dst Slot> ALWI void Floor<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Floor<Slot>::call(uint32_t d0) const { floor_tile(d0); }
template <Dst Slot> ALWI void Ceil<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Ceil<Slot>::call(uint32_t d0) const { ceil_tile(d0); }
template <Dst Slot> ALWI void Trunc<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Trunc<Slot>::call(uint32_t d0) const { trunc_tile(d0); }
template <Dst Slot> ALWI void Round<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Round<Slot>::call(uint32_t d0) const { round_tile(d0, decimals); }
template <Dst Slot> ALWI void Frac<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void Frac<Slot>::call(uint32_t d0) const { frac_tile(d0); }
template <Dst Slot> ALWI void StochasticRound<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot> ALWI void StochasticRound<Slot>::call(uint32_t d0) const { stochastic_round_tile(d0); }

// =============================================================================
// Op Method Definitions — Type / Identity / Bitwise (9 ops)
// =============================================================================

template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot>
ALWI void Typecast<in_dtype, out_dtype, Slot>::init() const { typecast_tile_init<in_dtype, out_dtype>(); }
template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot>
ALWI void Typecast<in_dtype, out_dtype, Slot>::call(uint32_t d0) const { typecast_tile<in_dtype, out_dtype>(d0); }

template <Dst Slot> ALWI void Identity<Slot>::init() const { identity_tile_init(); }
template <Dst Slot> ALWI void Identity<Slot>::call(uint32_t d0) const { identity_tile(d0); }
// NOTE: Bitwise/shift op implementations excluded — see sfpu_helpers.hpp for rationale.

// =============================================================================
// Op Method Definitions — Scalar Arithmetic (10 ops)
// =============================================================================

template <Dst Slot> ALWI void AddScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void AddScalar<Slot>::call(uint32_t d0) const { add_unary_tile(d0, scalar); }
template <Dst Slot> ALWI void SubScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void SubScalar<Slot>::call(uint32_t d0) const { sub_unary_tile(d0, scalar); }
template <Dst Slot> ALWI void MulScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void MulScalar<Slot>::call(uint32_t d0) const { mul_unary_tile(d0, scalar); }
template <Dst Slot> ALWI void DivScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void DivScalar<Slot>::call(uint32_t d0) const { div_unary_tile(d0, scalar); }
template <Dst Slot> ALWI void RsubScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot> ALWI void RsubScalar<Slot>::call(uint32_t d0) const { rsub_unary_tile(d0, scalar); }

template <Dst Slot> ALWI void Rsub<Slot>::init() const { rsub_tile_init(); }
template <Dst Slot> ALWI void Rsub<Slot>::call(uint32_t d0) const { rsub_tile(d0, param0); }

template <RoundingMode rounding_mode, Dst Slot>
ALWI void Rdiv<rounding_mode, Slot>::init() const { rdiv_tile_init(); }
template <RoundingMode rounding_mode, Dst Slot>
ALWI void Rdiv<rounding_mode, Slot>::call(uint32_t d0) const { rdiv_tile<rounding_mode>(d0, value); }

template <Dst Slot>
ALWI void Fmod<Slot>::init() const { fmod_tile_init(param0, param1); }
template <Dst Slot>
ALWI void Fmod<Slot>::call(uint32_t d0) const { fmod_tile(d0, param0, param1); }

template <Dst Slot>
ALWI void Remainder<Slot>::init() const { remainder_tile_init(param0, param1); }
template <Dst Slot>
ALWI void Remainder<Slot>::call(uint32_t d0) const { remainder_tile(d0, param0, param1); }

template <Dst Slot>
ALWI void Dropout<Slot>::init() const {}
template <Dst Slot>
ALWI void Dropout<Slot>::call(uint32_t d0) const { dropout_tile(d0, probability, scale_factor); }

// =============================================================================
// Op Method Definitions — Fill and Random (3 ops)
// =============================================================================

template <Dst Slot> ALWI void FillTile<Slot>::init() const { fill_tile_init(); }
template <Dst Slot> ALWI void FillTile<Slot>::call(uint32_t d0) const { fill_tile(d0, fill_val); }
template <Dst Slot> ALWI void FillTileBitcast<Slot>::init() const { fill_tile_init(); }
template <Dst Slot> ALWI void FillTileBitcast<Slot>::call(uint32_t d0) const { fill_tile_bitcast(d0, param0); }
template <Dst Slot> ALWI void RandTile<Slot>::init() const {}
template <Dst Slot> ALWI void RandTile<Slot>::call(uint32_t d0) const { rand_tile(d0, from, scale); }

// =============================================================================
// Op Method Definitions — Binary SFPU (7 ops)
// =============================================================================

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::init() const { add_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { add_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::init() const { sub_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { sub_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::init() const { mul_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { mul_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::init() const { div_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { div_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuRsub<In0, In1, Out>::init() const { rsub_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuRsub<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { rsub_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuPow<In0, In1, Out>::init() const { power_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuPow<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { power_binary_tile(a, b, c); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::init() const { eq_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::call(uint32_t a, uint32_t b, uint32_t c) const { eq_binary_tile(a, b, c); }

// =============================================================================
// Op Method Definitions — Ternary SFPU (4 ops)
// =============================================================================

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Where<df, In0, In1, In2, Out>::init() const { where_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Where<df, In0, In1, In2, Out>::call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const { where_tile<df>(a, b, c, d); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Lerp<df, In0, In1, In2, Out>::init() const { lerp_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Lerp<df, In0, In1, In2, Out>::call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const { lerp_tile<df>(a, b, c, d); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcmul<df, In0, In1, In2, Out>::init() const { addcmul_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcmul<df, In0, In1, In2, Out>::call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const { addcmul_tile<df>(a, b, c, d, value); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcdiv<df, In0, In1, In2, Out>::init() const { addcdiv_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcdiv<df, In0, In1, In2, Out>::call(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const { addcdiv_tile<df>(a, b, c, d, value); }

// =============================================================================
// CompactLoad Method Definitions
// =============================================================================

template <uint32_t CB, bool DoWait, bool DoPop, Dst... Slots>
ALWI void CompactLoad<CB, DoWait, DoPop, Slots...>::init() const {
    // No-op: copy_tile_to_dst_init is handled once by the pipeline before the tile loop.
    // This keeps init() uniform with compute ops but avoids redundant re-initialization.
}

template <uint32_t CB, bool DoWait, bool DoPop, Dst... Slots>
ALWI void CompactLoad<CB, DoWait, DoPop, Slots...>::exec(uint32_t offset) const {
    if constexpr (DoWait) {
        cb_wait_front(CB, 1);
    }
    ((copy_tile(CB, 0, static_cast<uint32_t>(Slots) + offset)), ...);
    if constexpr (DoPop) {
        cb_pop_front(CB, 1);
    }
}

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

constexpr bool sfpu_reconfig_input(SfpuDataFormatReconfig mode) {
    return mode == SfpuDataFormatReconfig::INPUT || mode == SfpuDataFormatReconfig::INPUT_AND_OUTPUT;
}

constexpr bool sfpu_reconfig_output(SfpuDataFormatReconfig mode) {
    return mode == SfpuDataFormatReconfig::OUTPUT || mode == SfpuDataFormatReconfig::INPUT_AND_OUTPUT;
}

/** @brief Get the CB of the first CompactLoad in a chain (for input reconfig) */
template <typename Chain>
struct FirstLoadCB { static constexpr uint32_t value = 0; };
// Non-load first element: recurse
template <typename First, typename... Rest>
struct FirstLoadCB<SfpuChain<First, Rest...>> {
    static constexpr uint32_t value = FirstLoadCB<SfpuChain<Rest...>>::value;
};
// CompactLoad first element: found it
template <uint32_t CB, bool W, bool P, Dst... S, typename... Rest>
struct FirstLoadCB<SfpuChain<CompactLoad<CB, W, P, S...>, Rest...>> {
    static constexpr uint32_t value = CB;
};

}  // namespace detail

// =============================================================================
// Pipeline Implementation
// =============================================================================

/**
 * @brief SFPU pipeline: unified chain execution
 *
 * After sfpu_chain() transformation, all chain elements (CompactLoad and compute ops)
 * have init()/exec()/apply(). The pipeline simply calls:
 * - Non-batched (batch_size=1): chain.apply(0) per tile
 * - Batched: chain.apply_batched(actual, stride) — init once, exec k times per element
 */
template <
    SfpuBatching batching,
    SfpuInputPolicy input_policy,
    SfpuOutputPolicy output_policy,
    SfpuDataFormatReconfig reconfig,
    typename Chain>
ALWI void sfpu_pipeline(
    Chain chain,
    uint32_t ocb,
    uint32_t num_tiles,
    Dst pack_slot) {
    ASSERT(num_tiles > 0);

    constexpr uint32_t chain_stride = Chain::stride;
    constexpr uint32_t batch_size = (batching == SfpuBatching::Disabled)
        ? 1 : (DEST_AUTO_LIMIT / chain_stride);
    static_assert(batch_size >= 1, "chain stride exceeds DEST capacity");

    ASSERT(static_cast<uint32_t>(pack_slot) < chain_stride);

    // Data format reconfiguration (once before the tile loop)
    if constexpr (detail::sfpu_reconfig_input(reconfig)) {
        reconfig_data_format_srca(detail::FirstLoadCB<Chain>::value);
    }
    if constexpr (detail::sfpu_reconfig_output(reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // Initialize unpacker for the first Load's CB (once, before the tile loop)
    copy_tile_to_dst_init_short(detail::FirstLoadCB<Chain>::value);

    // Bulk output: reserve all tiles upfront
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_reserve_back(ocb, num_tiles);
    }

    // Tile loop: init on first tile, then:
    //   - Single compute op: exec_only (safe, no init interference)
    //   - Multi compute op: full apply (inits interfere, must re-init each op)
    // Batched path packs multiple tiles per acquire/release cycle.
    for (uint32_t i = 0; i < num_tiles; i += batch_size) {
        const uint32_t actual = (batch_size == 1) ? 1
            : (((i + batch_size) <= num_tiles) ? batch_size : (num_tiles - i));

        tile_regs_acquire();

        for (uint32_t k = 0; k < actual; ++k) {
            if (i + k == 0) {
                // First tile: init + exec for all elements
                chain.apply(k * chain_stride);
            } else if constexpr (Chain::num_compute_ops <= 1) {
                // Single compute op: safe to skip init on subsequent tiles
                chain.exec_only(k * chain_stride);
            } else {
                // Multi compute op: must re-init each SFPU op (inits interfere)
                chain.apply(k * chain_stride);
            }
        }

        tile_regs_commit();
        tile_regs_wait();

        for (uint32_t k = 0; k < actual; ++k) {
            if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
                cb_reserve_back(ocb, 1);
            }
            pack_tile(static_cast<uint32_t>(pack_slot) + k * chain_stride, ocb);
            if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
                cb_push_back(ocb, 1);
            }
        }

        tile_regs_release();
    }

    // Bulk output: push all tiles at end
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_push_back(ocb, num_tiles);
    }
}

// =============================================================================
// Convenience: Single Unary Op Implementation
// =============================================================================

template <
    uint32_t ICB,
    SfpuBatching batching,
    SfpuInputPolicy input_policy,
    SfpuOutputPolicy output_policy,
    SfpuDataFormatReconfig reconfig,
    typename Op>
ALWI void sfpu_op(uint32_t ocb, uint32_t num_tiles, Op op) {
    auto chain = sfpu_chain(Load<ICB, Dst::D0>{}, op);
    sfpu_pipeline<batching, input_policy, output_policy, reconfig>(chain, ocb, num_tiles);
}

// =============================================================================
// Named Convenience Aliases Implementation
// =============================================================================

// --- Math ---

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Exp<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Log<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Log1p<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Sqrt<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Rsqrt<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Recip<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Abs<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Neg<>{});
}

// --- Activations ---

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sigmoid(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Sigmoid<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_tanh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Tanh<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gelu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Gelu<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_silu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Silu<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_relu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Relu<>{});
}

// --- Trigonometry ---

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sin(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Sin<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_cos(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Cos<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_tan(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Tan<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_asin(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Asin<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_acos(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Acos<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_atan(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Atan<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sinh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Sinh<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_cosh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Cosh<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_asinh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Asinh<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_acosh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Acosh<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_atanh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Atanh<>{});
}

// --- Error / Special Functions ---

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_erf(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Erf<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_erfc(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Erfc<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_erfinv(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Erfinv<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_i0(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, I0<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_i1(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, I1<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_lgamma(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Lgamma<>{});
}

// --- Predicates ---

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isinf(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Isinf<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isnan(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Isnan<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isfinite(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Isfinite<>{});
}

// --- Comparisons ---

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gtz(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Gtz<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_ltz(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Ltz<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_lez(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Lez<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gez(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Gez<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_eqz(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Eqz<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_nez(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Nez<>{});
}

// --- Rounding ---

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_floor(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Floor<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_ceil(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Ceil<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_trunc(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Trunc<>{});
}

template <uint32_t ICB, SfpuBatching B, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_frac(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, B, P, O, R>(ocb, num_tiles, Frac<>{});
}

}  // namespace compute_kernel_lib
