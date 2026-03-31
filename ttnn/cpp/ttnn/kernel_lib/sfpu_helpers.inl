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
ALWI void Exp<approx, fast, Slot>::exec() const { exp_tile<static_cast<bool>(approx), static_cast<bool>(fast)>(dst_idx); }
template <Approx approx, Approx fast, Dst Slot>
ALWI void Exp<approx, fast, Slot>::apply() const { init(); exec(); }

template <Approx approx, Dst Slot>
ALWI void Log<approx, Slot>::init() const { log_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Log<approx, Slot>::exec() const { log_tile<static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Log<approx, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void LogWithBase<Slot>::init() const { log_with_base_tile_init(); }
template <Dst Slot>
ALWI void LogWithBase<Slot>::exec() const { log_with_base_tile(dst_idx, base_scale); }
template <Dst Slot>
ALWI void LogWithBase<Slot>::apply() const { init(); exec(); }

template <Approx approx, Dst Slot>
ALWI void Log1p<approx, Slot>::init() const { log1p_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Log1p<approx, Slot>::exec() const { log1p_tile<static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Log1p<approx, Slot>::apply() const { init(); exec(); }

template <Approx approx, Dst Slot>
ALWI void Sqrt<approx, Slot>::init() const { sqrt_tile_init(); }
template <Approx approx, Dst Slot>
ALWI void Sqrt<approx, Slot>::exec() const { sqrt_tile<static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Sqrt<approx, Slot>::apply() const { init(); exec(); }

template <Legacy legacy, Approx approx, Dst Slot>
ALWI void Rsqrt<legacy, approx, Slot>::init() const { rsqrt_tile_init<static_cast<bool>(legacy)>(); }
template <Legacy legacy, Approx approx, Dst Slot>
ALWI void Rsqrt<legacy, approx, Slot>::exec() const { rsqrt_tile<static_cast<bool>(legacy), static_cast<bool>(approx)>(dst_idx); }
template <Legacy legacy, Approx approx, Dst Slot>
ALWI void Rsqrt<legacy, approx, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Cbrt<Slot>::init() const { cbrt_tile_init(); }
template <Dst Slot>
ALWI void Cbrt<Slot>::exec() const { cbrt_tile(dst_idx); }
template <Dst Slot>
ALWI void Cbrt<Slot>::apply() const { init(); exec(); }

template <Legacy legacy, Dst Slot>
ALWI void Recip<legacy, Slot>::init() const { recip_tile_init<static_cast<bool>(legacy)>(); }
template <Legacy legacy, Dst Slot>
ALWI void Recip<legacy, Slot>::exec() const { recip_tile<static_cast<bool>(legacy)>(dst_idx); }
template <Legacy legacy, Dst Slot>
ALWI void Recip<legacy, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Abs<Slot>::init() const { abs_tile_init(); }
template <Dst Slot>
ALWI void Abs<Slot>::exec() const { abs_tile(dst_idx); }
template <Dst Slot>
ALWI void Abs<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Neg<Slot>::init() const { negative_tile_init(); }
template <Dst Slot>
ALWI void Neg<Slot>::exec() const { negative_tile(dst_idx); }
template <Dst Slot>
ALWI void Neg<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Square<Slot>::init() const { square_tile_init(); }
template <Dst Slot>
ALWI void Square<Slot>::exec() const { square_tile(dst_idx); }
template <Dst Slot>
ALWI void Square<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Sign<Slot>::init() const { sign_tile_init(); }
template <Dst Slot>
ALWI void Sign<Slot>::exec() const { sign_tile(dst_idx); }
template <Dst Slot>
ALWI void Sign<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Signbit<Slot>::init() const { signbit_tile_init(); }
template <Dst Slot>
ALWI void Signbit<Slot>::exec() const { signbit_tile(dst_idx); }
template <Dst Slot>
ALWI void Signbit<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Exp2<Slot>::init() const { exp2_tile_init(); }
template <Dst Slot>
ALWI void Exp2<Slot>::exec() const { exp2_tile(dst_idx); }
template <Dst Slot>
ALWI void Exp2<Slot>::apply() const { init(); exec(); }

template <Approx approx, Dst Slot>
ALWI void Expm1<approx, Slot>::init() const { expm1_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Expm1<approx, Slot>::exec() const { expm1_tile<static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Expm1<approx, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Power<Slot>::init() const { power_tile_init(); }
template <Dst Slot>
ALWI void Power<Slot>::exec() const { power_tile(dst_idx, exponent); }
template <Dst Slot>
ALWI void Power<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void PowerIterative<Slot>::init() const { power_tile_init(); }
template <Dst Slot>
ALWI void PowerIterative<Slot>::exec() const { power_tile(dst_idx, int_exponent); }
template <Dst Slot>
ALWI void PowerIterative<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Rpow<Slot>::init() const { rpow_tile_init(); }
template <Dst Slot>
ALWI void Rpow<Slot>::exec() const { rpow_tile(dst_idx, base_val); }
template <Dst Slot>
ALWI void Rpow<Slot>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Activations (11 ops)
// =============================================================================

template <Approx approx, Dst Slot>
ALWI void Sigmoid<approx, Slot>::init() const { sigmoid_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Sigmoid<approx, Slot>::exec() const { sigmoid_tile<(int)VectorMode::RC, static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Sigmoid<approx, Slot>::apply() const { init(); exec(); }

template <Approx approx, Dst Slot>
ALWI void Tanh<approx, Slot>::init() const { tanh_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Tanh<approx, Slot>::exec() const { tanh_tile<static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Tanh<approx, Slot>::apply() const { init(); exec(); }

template <Approx approx, Dst Slot>
ALWI void Gelu<approx, Slot>::init() const { gelu_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Gelu<approx, Slot>::exec() const { gelu_tile<static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Gelu<approx, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Silu<Slot>::init() const { silu_tile_init(); }
template <Dst Slot>
ALWI void Silu<Slot>::exec() const { silu_tile(dst_idx); }
template <Dst Slot>
ALWI void Silu<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Relu<Slot>::init() const { relu_tile_init(); }
template <Dst Slot>
ALWI void Relu<Slot>::exec() const { relu_tile(dst_idx); }
template <Dst Slot>
ALWI void Relu<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Hardmish<Slot>::init() const { hardmish_tile_init(); }
template <Dst Slot>
ALWI void Hardmish<Slot>::exec() const { hardmish_tile(dst_idx); }
template <Dst Slot>
ALWI void Hardmish<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Hardsigmoid<Slot>::init() const { hardsigmoid_tile_init(); }
template <Dst Slot>
ALWI void Hardsigmoid<Slot>::exec() const { hardsigmoid_tile(dst_idx); }
template <Dst Slot>
ALWI void Hardsigmoid<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Hardtanh<Slot>::init() const { hardtanh_tile_init(); }
template <Dst Slot>
ALWI void Hardtanh<Slot>::exec() const { hardtanh_tile(dst_idx, param_min, param_max); }
template <Dst Slot>
ALWI void Hardtanh<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Softsign<Slot>::init() const { softsign_tile_init(); }
template <Dst Slot>
ALWI void Softsign<Slot>::exec() const { softsign_tile(dst_idx); }
template <Dst Slot>
ALWI void Softsign<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Softplus<Slot>::init() const { softplus_tile_init(); }
template <Dst Slot>
ALWI void Softplus<Slot>::exec() const { softplus_tile(dst_idx, beta, beta_recip, threshold); }
template <Dst Slot>
ALWI void Softplus<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Xielu<Slot>::init() const { xielu_tile_init(); }
template <Dst Slot>
ALWI void Xielu<Slot>::exec() const { xielu_tile(dst_idx, alpha_p, alpha_n); }
template <Dst Slot>
ALWI void Xielu<Slot>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Trigonometry (11 ops)
// =============================================================================

template <Dst Slot>
ALWI void Sin<Slot>::init() const { sin_tile_init(); }
template <Dst Slot>
ALWI void Sin<Slot>::exec() const { sin_tile(dst_idx); }
template <Dst Slot>
ALWI void Sin<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Cos<Slot>::init() const { cos_tile_init(); }
template <Dst Slot>
ALWI void Cos<Slot>::exec() const { cos_tile(dst_idx); }
template <Dst Slot>
ALWI void Cos<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Tan<Slot>::init() const { tan_tile_init(); }
template <Dst Slot>
ALWI void Tan<Slot>::exec() const { tan_tile(dst_idx); }
template <Dst Slot>
ALWI void Tan<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Asin<Slot>::init() const { asin_tile_init(); }
template <Dst Slot>
ALWI void Asin<Slot>::exec() const { asin_tile(dst_idx); }
template <Dst Slot>
ALWI void Asin<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Acos<Slot>::init() const { acos_tile_init(); }
template <Dst Slot>
ALWI void Acos<Slot>::exec() const { acos_tile(dst_idx); }
template <Dst Slot>
ALWI void Acos<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Atan<Slot>::init() const { atan_tile_init(); }
template <Dst Slot>
ALWI void Atan<Slot>::exec() const { atan_tile(dst_idx); }
template <Dst Slot>
ALWI void Atan<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Sinh<Slot>::init() const { sinh_tile_init(); }
template <Dst Slot>
ALWI void Sinh<Slot>::exec() const { sinh_tile(dst_idx); }
template <Dst Slot>
ALWI void Sinh<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Cosh<Slot>::init() const { cosh_tile_init(); }
template <Dst Slot>
ALWI void Cosh<Slot>::exec() const { cosh_tile(dst_idx); }
template <Dst Slot>
ALWI void Cosh<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Asinh<Slot>::init() const { asinh_tile_init(); }
template <Dst Slot>
ALWI void Asinh<Slot>::exec() const { asinh_tile(dst_idx); }
template <Dst Slot>
ALWI void Asinh<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Acosh<Slot>::init() const { acosh_tile_init(); }
template <Dst Slot>
ALWI void Acosh<Slot>::exec() const { acosh_tile(dst_idx); }
template <Dst Slot>
ALWI void Acosh<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Atanh<Slot>::init() const { atanh_tile_init(); }
template <Dst Slot>
ALWI void Atanh<Slot>::exec() const { atanh_tile(dst_idx); }
template <Dst Slot>
ALWI void Atanh<Slot>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Error / Special Functions (6 ops)
// =============================================================================

template <Approx approx, Dst Slot>
ALWI void Erf<approx, Slot>::init() const { erf_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Erf<approx, Slot>::exec() const { erf_tile<static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Erf<approx, Slot>::apply() const { init(); exec(); }

template <Approx approx, Dst Slot>
ALWI void Erfc<approx, Slot>::init() const { erfc_tile_init<static_cast<bool>(approx)>(); }
template <Approx approx, Dst Slot>
ALWI void Erfc<approx, Slot>::exec() const { erfc_tile<static_cast<bool>(approx)>(dst_idx); }
template <Approx approx, Dst Slot>
ALWI void Erfc<approx, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Erfinv<Slot>::init() const { erfinv_tile_init(); }
template <Dst Slot>
ALWI void Erfinv<Slot>::exec() const { erfinv_tile(dst_idx); }
template <Dst Slot>
ALWI void Erfinv<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void I0<Slot>::init() const { i0_tile_init(); }
template <Dst Slot>
ALWI void I0<Slot>::exec() const { i0_tile(dst_idx); }
template <Dst Slot>
ALWI void I0<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void I1<Slot>::init() const { i1_tile_init(); }
template <Dst Slot>
ALWI void I1<Slot>::exec() const { i1_tile(dst_idx); }
template <Dst Slot>
ALWI void I1<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Lgamma<Slot>::init() const { lgamma_stirling_tile_init(); }
template <Dst Slot>
ALWI void Lgamma<Slot>::exec() const { lgamma_stirling_tile(dst_idx); }
template <Dst Slot>
ALWI void Lgamma<Slot>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Predicates and Comparisons (18 ops)
// =============================================================================

template <Dst Slot>
ALWI void Isinf<Slot>::init() const { isinf_tile_init(); }
template <Dst Slot>
ALWI void Isinf<Slot>::exec() const { isinf_tile(dst_idx); }
template <Dst Slot>
ALWI void Isinf<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Isposinf<Slot>::init() const { isposinf_tile_init(); }
template <Dst Slot>
ALWI void Isposinf<Slot>::exec() const { isposinf_tile(dst_idx); }
template <Dst Slot>
ALWI void Isposinf<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Isneginf<Slot>::init() const { isneginf_tile_init(); }
template <Dst Slot>
ALWI void Isneginf<Slot>::exec() const { isneginf_tile(dst_idx); }
template <Dst Slot>
ALWI void Isneginf<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Isnan<Slot>::init() const { isnan_tile_init(); }
template <Dst Slot>
ALWI void Isnan<Slot>::exec() const { isnan_tile(dst_idx); }
template <Dst Slot>
ALWI void Isnan<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Isfinite<Slot>::init() const { isfinite_tile_init(); }
template <Dst Slot>
ALWI void Isfinite<Slot>::exec() const { isfinite_tile(dst_idx); }
template <Dst Slot>
ALWI void Isfinite<Slot>::apply() const { init(); exec(); }

template <DataFormat df, Dst Slot>
ALWI void LogicalNot<df, Slot>::init() const { logical_not_tile_init(); }
template <DataFormat df, Dst Slot>
ALWI void LogicalNot<df, Slot>::exec() const { logical_not_tile<df>(dst_idx); }
template <DataFormat df, Dst Slot>
ALWI void LogicalNot<df, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Gtz<Slot>::init() const { gtz_tile_init(); }
template <Dst Slot>
ALWI void Gtz<Slot>::exec() const { gtz_tile(dst_idx); }
template <Dst Slot>
ALWI void Gtz<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Ltz<Slot>::init() const { ltz_tile_init(); }
template <Dst Slot>
ALWI void Ltz<Slot>::exec() const { ltz_tile(dst_idx); }
template <Dst Slot>
ALWI void Ltz<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Lez<Slot>::init() const { lez_tile_init(); }
template <Dst Slot>
ALWI void Lez<Slot>::exec() const { lez_tile(dst_idx); }
template <Dst Slot>
ALWI void Lez<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Gez<Slot>::init() const { gez_tile_init(); }
template <Dst Slot>
ALWI void Gez<Slot>::exec() const { gez_tile(dst_idx); }
template <Dst Slot>
ALWI void Gez<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Eqz<Slot>::init() const { eqz_tile_init(); }
template <Dst Slot>
ALWI void Eqz<Slot>::exec() const { eqz_tile(dst_idx); }
template <Dst Slot>
ALWI void Eqz<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Nez<Slot>::init() const { nez_tile_init(); }
template <Dst Slot>
ALWI void Nez<Slot>::exec() const { nez_tile(dst_idx); }
template <Dst Slot>
ALWI void Nez<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void UnaryEq<Slot>::init() const { unary_eq_tile_init(); }
template <Dst Slot>
ALWI void UnaryEq<Slot>::exec() const { unary_eq_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void UnaryEq<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void UnaryNe<Slot>::init() const { unary_ne_tile_init(); }
template <Dst Slot>
ALWI void UnaryNe<Slot>::exec() const { unary_ne_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void UnaryNe<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void UnaryGt<Slot>::init() const { unary_gt_tile_init(); }
template <Dst Slot>
ALWI void UnaryGt<Slot>::exec() const { unary_gt_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void UnaryGt<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void UnaryGe<Slot>::init() const { unary_ge_tile_init(); }
template <Dst Slot>
ALWI void UnaryGe<Slot>::exec() const { unary_ge_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void UnaryGe<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void UnaryLt<Slot>::init() const { unary_lt_tile_init(); }
template <Dst Slot>
ALWI void UnaryLt<Slot>::exec() const { unary_lt_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void UnaryLt<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void UnaryLe<Slot>::init() const { unary_le_tile_init(); }
template <Dst Slot>
ALWI void UnaryLe<Slot>::exec() const { unary_le_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void UnaryLe<Slot>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Additional Activations (7 ops)
// =============================================================================

template <Dst Slot>
ALWI void Elu<Slot>::init() const { elu_tile_init(); }
template <Dst Slot>
ALWI void Elu<Slot>::exec() const { elu_tile(dst_idx, alpha); }
template <Dst Slot>
ALWI void Elu<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Selu<Slot>::init() const { selu_tile_init(); }
template <Dst Slot>
ALWI void Selu<Slot>::exec() const { selu_tile(dst_idx, scale, alpha); }
template <Dst Slot>
ALWI void Selu<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Celu<Slot>::init() const { celu_tile_init(); }
template <Dst Slot>
ALWI void Celu<Slot>::exec() const { celu_tile(dst_idx, alpha, alpha_recip); }
template <Dst Slot>
ALWI void Celu<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Softshrink<Slot>::init() const { softshrink_tile_init(); }
template <Dst Slot>
ALWI void Softshrink<Slot>::exec() const { softshrink_tile(dst_idx, lambda); }
template <Dst Slot>
ALWI void Softshrink<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Clamp<Slot>::init() const { clamp_tile_init(); }
template <Dst Slot>
ALWI void Clamp<Slot>::exec() const { clamp_tile(dst_idx, param_min, param_max); }
template <Dst Slot>
ALWI void Clamp<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Threshold<Slot>::init() const { threshold_tile_init(); }
template <Dst Slot>
ALWI void Threshold<Slot>::exec() const { threshold_tile(dst_idx, threshold, value); }
template <Dst Slot>
ALWI void Threshold<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Prelu<Slot>::init() const { prelu_tile_init(); }
template <Dst Slot>
ALWI void Prelu<Slot>::exec() const { prelu_tile(dst_idx, weight); }
template <Dst Slot>
ALWI void Prelu<Slot>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Rounding (6 ops)
// =============================================================================

template <Dst Slot>
ALWI void Floor<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot>
ALWI void Floor<Slot>::exec() const { floor_tile(dst_idx); }
template <Dst Slot>
ALWI void Floor<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Ceil<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot>
ALWI void Ceil<Slot>::exec() const { ceil_tile(dst_idx); }
template <Dst Slot>
ALWI void Ceil<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Trunc<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot>
ALWI void Trunc<Slot>::exec() const { trunc_tile(dst_idx); }
template <Dst Slot>
ALWI void Trunc<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Round<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot>
ALWI void Round<Slot>::exec() const { round_tile(dst_idx, decimals); }
template <Dst Slot>
ALWI void Round<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Frac<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot>
ALWI void Frac<Slot>::exec() const { frac_tile(dst_idx); }
template <Dst Slot>
ALWI void Frac<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void StochasticRound<Slot>::init() const { rounding_op_tile_init(); }
template <Dst Slot>
ALWI void StochasticRound<Slot>::exec() const { stochastic_round_tile(dst_idx); }
template <Dst Slot>
ALWI void StochasticRound<Slot>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Type / Identity / Bitwise (9 ops)
// =============================================================================

template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot>
ALWI void Typecast<in_dtype, out_dtype, Slot>::init() const { typecast_tile_init<in_dtype, out_dtype>(); }
template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot>
ALWI void Typecast<in_dtype, out_dtype, Slot>::exec() const { typecast_tile<in_dtype, out_dtype>(dst_idx); }
template <uint32_t in_dtype, uint32_t out_dtype, Dst Slot>
ALWI void Typecast<in_dtype, out_dtype, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Identity<Slot>::init() const { identity_tile_init(); }
template <Dst Slot>
ALWI void Identity<Slot>::exec() const { identity_tile(dst_idx); }
template <Dst Slot>
ALWI void Identity<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void BitwiseNot<Slot>::init() const { bitwise_not_tile_init(); }
template <Dst Slot>
ALWI void BitwiseNot<Slot>::exec() const { bitwise_not_tile(dst_idx); }
template <Dst Slot>
ALWI void BitwiseNot<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void BitwiseAnd<Slot>::init() const { bitwise_and_tile_init(); }
template <Dst Slot>
ALWI void BitwiseAnd<Slot>::exec() const { bitwise_and_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void BitwiseAnd<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void BitwiseOr<Slot>::init() const { bitwise_or_tile_init(); }
template <Dst Slot>
ALWI void BitwiseOr<Slot>::exec() const { bitwise_or_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void BitwiseOr<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void BitwiseXor<Slot>::init() const { bitwise_xor_tile_init(); }
template <Dst Slot>
ALWI void BitwiseXor<Slot>::exec() const { bitwise_xor_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void BitwiseXor<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void LeftShift<Slot>::init() const { left_shift_tile_init(); }
template <Dst Slot>
ALWI void LeftShift<Slot>::exec() const { left_shift_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void LeftShift<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void RightShift<Slot>::init() const { right_shift_tile_init(); }
template <Dst Slot>
ALWI void RightShift<Slot>::exec() const { right_shift_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void RightShift<Slot>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Scalar Arithmetic (10 ops)
// =============================================================================

template <Dst Slot>
ALWI void AddScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot>
ALWI void AddScalar<Slot>::exec() const { add_unary_tile(dst_idx, scalar); }
template <Dst Slot>
ALWI void AddScalar<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void SubScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot>
ALWI void SubScalar<Slot>::exec() const { sub_unary_tile(dst_idx, scalar); }
template <Dst Slot>
ALWI void SubScalar<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void MulScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot>
ALWI void MulScalar<Slot>::exec() const { mul_unary_tile(dst_idx, scalar); }
template <Dst Slot>
ALWI void MulScalar<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void DivScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot>
ALWI void DivScalar<Slot>::exec() const { div_unary_tile(dst_idx, scalar); }
template <Dst Slot>
ALWI void DivScalar<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void RsubScalar<Slot>::init() const { binop_with_scalar_tile_init(); }
template <Dst Slot>
ALWI void RsubScalar<Slot>::exec() const { rsub_unary_tile(dst_idx, scalar); }
template <Dst Slot>
ALWI void RsubScalar<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Rsub<Slot>::init() const { rsub_tile_init(); }
template <Dst Slot>
ALWI void Rsub<Slot>::exec() const { rsub_tile(dst_idx, param0); }
template <Dst Slot>
ALWI void Rsub<Slot>::apply() const { init(); exec(); }

template <RoundingMode rounding_mode, Dst Slot>
ALWI void Rdiv<rounding_mode, Slot>::init() const { rdiv_tile_init(); }
template <RoundingMode rounding_mode, Dst Slot>
ALWI void Rdiv<rounding_mode, Slot>::exec() const { rdiv_tile<rounding_mode>(dst_idx, value); }
template <RoundingMode rounding_mode, Dst Slot>
ALWI void Rdiv<rounding_mode, Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Fmod<Slot>::init() const { fmod_tile_init(param0, param1); }
template <Dst Slot>
ALWI void Fmod<Slot>::exec() const { fmod_tile(dst_idx, param0, param1); }
template <Dst Slot>
ALWI void Fmod<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Remainder<Slot>::init() const { remainder_tile_init(param0, param1); }
template <Dst Slot>
ALWI void Remainder<Slot>::exec() const { remainder_tile(dst_idx, param0, param1); }
template <Dst Slot>
ALWI void Remainder<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void Dropout<Slot>::init() const {}
template <Dst Slot>
ALWI void Dropout<Slot>::exec() const { dropout_tile(dst_idx, probability, scale_factor); }
template <Dst Slot>
ALWI void Dropout<Slot>::apply() const { exec(); }

// =============================================================================
// Op Method Definitions — Fill and Random (3 ops)
// =============================================================================

template <Dst Slot>
ALWI void FillTile<Slot>::init() const { fill_tile_init(); }
template <Dst Slot>
ALWI void FillTile<Slot>::exec() const { fill_tile(dst_idx, fill_val); }
template <Dst Slot>
ALWI void FillTile<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void FillTileBitcast<Slot>::init() const { fill_tile_init(); }
template <Dst Slot>
ALWI void FillTileBitcast<Slot>::exec() const { fill_tile_bitcast(dst_idx, param0); }
template <Dst Slot>
ALWI void FillTileBitcast<Slot>::apply() const { init(); exec(); }

template <Dst Slot>
ALWI void RandTile<Slot>::init() const {}
template <Dst Slot>
ALWI void RandTile<Slot>::exec() const { rand_tile(dst_idx, from, scale); }
template <Dst Slot>
ALWI void RandTile<Slot>::apply() const { exec(); }

// =============================================================================
// Op Method Definitions — Binary SFPU (7 ops)
// =============================================================================

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::init() const { add_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::exec() const { add_binary_tile(in0, in1, out); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuAdd<In0, In1, Out>::apply() const { init(); exec(); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::init() const { sub_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::exec() const { sub_binary_tile(in0, in1, out); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuSub<In0, In1, Out>::apply() const { init(); exec(); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::init() const { mul_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::exec() const { mul_binary_tile(in0, in1, out); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuMul<In0, In1, Out>::apply() const { init(); exec(); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::init() const { div_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::exec() const { div_binary_tile(in0, in1, out); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuDiv<In0, In1, Out>::apply() const { init(); exec(); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuRsub<In0, In1, Out>::init() const { rsub_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuRsub<In0, In1, Out>::exec() const { rsub_binary_tile(in0, in1, out); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuRsub<In0, In1, Out>::apply() const { init(); exec(); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuPow<In0, In1, Out>::init() const { power_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuPow<In0, In1, Out>::exec() const { power_binary_tile(in0, in1, out); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuPow<In0, In1, Out>::apply() const { init(); exec(); }

template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::init() const { eq_binary_tile_init(); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::exec() const { eq_binary_tile(in0, in1, out); }
template <Dst In0, Dst In1, Dst Out>
ALWI void SfpuEq<In0, In1, Out>::apply() const { init(); exec(); }

// =============================================================================
// Op Method Definitions — Ternary SFPU (4 ops)
// =============================================================================

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Where<df, In0, In1, In2, Out>::init() const { where_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Where<df, In0, In1, In2, Out>::exec() const { where_tile<df>(in0, in1, in2, out); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Where<df, In0, In1, In2, Out>::apply() const { init(); exec(); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Lerp<df, In0, In1, In2, Out>::init() const { lerp_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Lerp<df, In0, In1, In2, Out>::exec() const { lerp_tile<df>(in0, in1, in2, out); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Lerp<df, In0, In1, In2, Out>::apply() const { init(); exec(); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcmul<df, In0, In1, In2, Out>::init() const { addcmul_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcmul<df, In0, In1, In2, Out>::exec() const { addcmul_tile<df>(in0, in1, in2, out, value); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcmul<df, In0, In1, In2, Out>::apply() const { init(); exec(); }

template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcdiv<df, In0, In1, In2, Out>::init() const { addcdiv_tile_init(); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcdiv<df, In0, In1, In2, Out>::exec() const { addcdiv_tile<df>(in0, in1, in2, out, value); }
template <DataFormat df, Dst In0, Dst In1, Dst In2, Dst Out>
ALWI void Addcdiv<df, In0, In1, In2, Out>::apply() const { init(); exec(); }

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

/**
 * @brief Loader functor used by sfpu_pipeline to handle Load ops
 *
 * Tracks the last CB used for _with_dt reconfiguration. On first call,
 * initializes with copy_tile_to_dst_init_short. On subsequent calls with
 * a different CB, uses copy_tile_to_dst_init_short_with_dt.
 *
 * For WaitAndPopPerTile: deduplicates wait/pop per unique CB. When two Loads
 * reference the same CB (e.g., Load<cb, D0>, Load<cb, D1>), waits once before
 * the first copy and defers pop until all copies from that CB are complete.
 */
template <SfpuInputPolicy input_policy>
struct TileLoader {
    uint32_t last_cb;
    bool initialized;
    uint32_t tile_idx;  // 0 for streaming (WaitAndPopPerTile), i for indexed

    // CB dedup tracking for wait/pop (max 8 unique CBs)
    static constexpr uint32_t MAX_CBS = 8;
    uint32_t waited_cbs[MAX_CBS];
    uint32_t num_waited;

    template <typename LoadOp>
    ALWI void operator()(const LoadOp& load) {
        constexpr uint32_t cb = LoadOp::cb;
        constexpr uint32_t dst = LoadOp::dst_idx;

        // _with_dt tracking: reconfigure unpacker when CB changes
        if (!initialized) {
            copy_tile_to_dst_init_short(cb);
            last_cb = cb;
            initialized = true;
        } else if (cb != last_cb) {
            copy_tile_to_dst_init_short_with_dt(last_cb, cb);
            last_cb = cb;
        }

        // CB synchronization: wait once per unique CB
        if constexpr (input_policy == SfpuInputPolicy::WaitAndPopPerTile) {
            bool already_waited = false;
            for (uint32_t i = 0; i < num_waited; ++i) {
                if (waited_cbs[i] == cb) { already_waited = true; break; }
            }
            if (!already_waited) {
                cb_wait_front(cb, 1);
                waited_cbs[num_waited++] = cb;
            }
        }

        // Copy tile from CB to DEST slot
        // For streaming: tile_idx is always 0 (tile stays at CB front until all copies done)
        // For upfront/no-wait: tile_idx increments through the CB
        copy_tile(cb, tile_idx, dst);
    }

    // Pop all waited CBs (called after all Loads are complete)
    ALWI void pop_all() {
        if constexpr (input_policy == SfpuInputPolicy::WaitAndPopPerTile) {
            for (uint32_t i = 0; i < num_waited; ++i) {
                cb_pop_front(waited_cbs[i], 1);
            }
        }
    }
};

/**
 * @brief Functor that reconfigures unpacker format to the first Load's CB
 *
 * Only reconfigures once — on the first Load's CB. This is a safety net for
 * when sfpu_pipeline is called after a different operation (reduce, binary FPU)
 * that may have left the unpacker in an unknown state. Within the tile loop,
 * TileLoader's copy_tile_to_dst_init_short handles the full unpacker init,
 * and _with_dt handles subsequent CB switches conditionally.
 */
struct InputReconfigFunctor {
    bool done;
    template <typename LoadOp>
    ALWI void operator()(const LoadOp&) {
        if (!done) {
            reconfig_data_format_srca(LoadOp::cb);
            done = true;
        }
    }
};

/** @brief Functor that waits for num_tiles on each Load's CB */
struct UpfrontWaitFunctor {
    uint32_t num_tiles;
    template <typename LoadOp>
    ALWI void operator()(const LoadOp&) {
        cb_wait_front(LoadOp::cb, num_tiles);
    }
};

}  // namespace detail

// =============================================================================
// Pipeline Implementation
// =============================================================================

template <
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

    // Runtime DEST capacity validation
    ASSERT(static_cast<uint32_t>(pack_slot) < DEST_AUTO_LIMIT);

    // Data format reconfiguration (once before the tile loop)
    // Input: reconfig unpacker to first Load's CB format. This is a safety net
    // for when the pipeline follows a different operation type. Within the tile
    // loop, TileLoader handles CB switches via copy_tile_to_dst_init_short/_with_dt.
    if constexpr (detail::sfpu_reconfig_input(reconfig)) {
        detail::InputReconfigFunctor reconfig_fn{false};
        chain.for_each_load(reconfig_fn);
    }
    // Output: reconfig packer to output CB format.
    if constexpr (detail::sfpu_reconfig_output(reconfig)) {
        pack_reconfig_data_format(ocb);
    }

    // Upfront waits for non-streaming input policies
    if constexpr (input_policy == SfpuInputPolicy::WaitUpfrontNoPop) {
        // Wait for num_tiles on each Load's CB. If multiple Loads reference
        // the same CB, the redundant cb_wait_front calls are harmless (idempotent).
        detail::UpfrontWaitFunctor wait_fn{num_tiles};
        chain.for_each_load(wait_fn);
    }

    // Bulk output: reserve all tiles upfront
    if constexpr (output_policy == SfpuOutputPolicy::Bulk) {
        cb_reserve_back(ocb, num_tiles);
    }

    // Per-tile streaming loop
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        // --- Load phase ---
        // Tile index: 0 for streaming (WaitAndPopPerTile pops after each),
        // i for indexed access (WaitUpfrontNoPop/NoWaitNoPop)
        constexpr bool streaming = (input_policy == SfpuInputPolicy::WaitAndPopPerTile);
        const uint32_t tile_idx = streaming ? 0 : i;

        detail::TileLoader<input_policy> loader{0, false, tile_idx, {}, 0};
        chain.for_each_load(loader);
        loader.pop_all();

        // --- Compute phase ---
        chain.apply();

        // --- Pack phase ---
        tile_regs_commit();
        tile_regs_wait();

        if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
            cb_reserve_back(ocb, 1);
        }

        pack_tile(static_cast<uint32_t>(pack_slot), ocb);

        if constexpr (output_policy == SfpuOutputPolicy::PerTile) {
            cb_push_back(ocb, 1);
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
    SfpuInputPolicy input_policy,
    SfpuOutputPolicy output_policy,
    SfpuDataFormatReconfig reconfig,
    typename Op>
ALWI void sfpu_op(uint32_t ocb, uint32_t num_tiles, Op op) {
    auto chain = sfpu_chain(Load<ICB, Dst::D0>{}, op);
    sfpu_pipeline<input_policy, output_policy, reconfig>(chain, ocb, num_tiles);
}

// =============================================================================
// Named Convenience Aliases Implementation
// =============================================================================

// --- Math ---

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_exp(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Exp<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_log(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Log<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_log1p(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Log1p<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sqrt(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Sqrt<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_rsqrt(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Rsqrt<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_recip(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Recip<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_abs(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Abs<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_neg(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Neg<>{});
}

// --- Activations ---

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sigmoid(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Sigmoid<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_tanh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Tanh<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gelu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Gelu<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_silu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Silu<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_relu(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Relu<>{});
}

// --- Trigonometry ---

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sin(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Sin<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_cos(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Cos<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_tan(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Tan<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_asin(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Asin<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_acos(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Acos<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_atan(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Atan<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_sinh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Sinh<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_cosh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Cosh<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_asinh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Asinh<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_acosh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Acosh<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_atanh(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Atanh<>{});
}

// --- Error / Special Functions ---

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_erf(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Erf<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_erfc(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Erfc<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_erfinv(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Erfinv<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_i0(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, I0<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_i1(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, I1<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_lgamma(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Lgamma<>{});
}

// --- Predicates ---

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isinf(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Isinf<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isnan(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Isnan<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_isfinite(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Isfinite<>{});
}

// --- Comparisons ---

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gtz(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Gtz<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_ltz(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Ltz<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_lez(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Lez<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_gez(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Gez<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_eqz(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Eqz<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_nez(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Nez<>{});
}

// --- Rounding ---

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_floor(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Floor<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_ceil(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Ceil<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_trunc(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Trunc<>{});
}

template <uint32_t ICB, SfpuInputPolicy P, SfpuOutputPolicy O, SfpuDataFormatReconfig R>
ALWI void sfpu_frac(uint32_t ocb, uint32_t num_tiles) {
    sfpu_op<ICB, P, O, R>(ocb, num_tiles, Frac<>{});
}

}  // namespace compute_kernel_lib
