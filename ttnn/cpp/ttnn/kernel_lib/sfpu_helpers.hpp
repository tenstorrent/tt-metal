// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Umbrella header: includes all SFPU operation categories.
// For minimal includes, use the per-category headers directly:
//   sfpu_chain.hpp      — chain/pipeline infrastructure only
//   sfpu_math.hpp       — exp, log, sqrt, recip, abs, neg, power, ...
//   sfpu_activations.hpp — sigmoid, tanh, gelu, relu, silu, hardmish, ...
//   sfpu_trig.hpp       — sin, cos, tan, asin, acos, atan, sinh, cosh, ...
//   sfpu_special.hpp    — erf, erfc, erfinv, i0, i1, lgamma
//   sfpu_predicates.hpp — isinf, isnan, gtz, ltz, unary_eq, ...
//   sfpu_rounding.hpp   — floor, ceil, trunc, round, frac
//   sfpu_scalar.hpp     — add/sub/mul/div scalar, rsub, rdiv, fmod, dropout
//   sfpu_misc.hpp       — typecast, identity, fill, rand
//   sfpu_binary.hpp     — SfpuAdd, SfpuSub, SfpuMul, SfpuDiv, SfpuPow, SfpuEq
//   sfpu_ternary.hpp    — Where, Lerp, Addcmul, Addcdiv

#include "ttnn/cpp/ttnn/kernel_lib/sfpu_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activations.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_trig.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_special.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_predicates.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_rounding.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_binary.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_ternary.hpp"
