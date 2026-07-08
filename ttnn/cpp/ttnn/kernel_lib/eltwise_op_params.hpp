// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eltwise_op_params.hpp
 * @brief Self-documenting template-param enums shared by the eltwise op-helper structs.
 *
 * These are an op-helper concern, not part of the chain machinery, so they live here rather
 * than in eltwise_chain.hpp. The op-helper headers (eltwise_math.hpp, eltwise_special.hpp,
 * eltwise_activations.hpp) include this; kernels pick the values up transitively through them.
 */

namespace compute_kernel_lib {

/// SFPU approximation mode for transcendental ops (Exp, Log, Sqrt, Rsqrt, Erf, the Gelu/Tanh
/// derivatives, …). Lowers to the LLK op's `APPROXIMATE` template bool: `Exact` keeps the
/// precise path, `Fast` selects the lower-precision fast approximation.
enum class Approx : bool { Exact = false, Fast = true };

/// Legacy code-path toggle. Currently used only by `Rsqrt`, whose `rsqrt_tile<...>` takes a
/// leading `legacy` template bool selecting the older implementation. `Off` = modern path.
enum class Legacy : bool { Off = false, On = true };

}  // namespace compute_kernel_lib
