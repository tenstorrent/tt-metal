// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cstdint>
#include <optional>

#include <tt_stl/assert.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/activation_types.hpp"

namespace compute_kernel_lib {

/**
 * @brief Consolidated activation parameters structure
 *
 * Contains the activation type and its associated parameters in a single struct
 * These values are passed as compile time arguments to the kernel
 */
struct ActivationParams {
    KernelActivation type = KernelActivation::NONE;
    uint32_t param0 = 0;
    uint32_t param1 = 0;
    uint32_t param2 = 0;
    bool pack_relu = false;
};

/**
 * @brief Extract activation parameters
 *
 * Extracts the activation type and both parameters. Prepares parameter values for the kernel.
 *
 * @param activation The UnaryWithParam containing the activation operation and parameters
 * @return ActivationParams struct with type and activation specific param0, param1, param2
 */
inline ActivationParams get_activation_params(
    const ttnn::operations::unary::UnaryWithParam& activation, bool default_fast_gelu = false) {
    using ttnn::operations::unary::UnaryOpType;

    // Activation parameters provided by the ttnn op.
    std::span<const float> params = activation.get_params();
    TT_FATAL(
        params.size() <= 2, "Invalid number of activation parameters: {}. Expected no more than 2.", params.size());
    const bool has_first = !params.empty();
    const bool has_second = params.size() > 1;

    // Activation parameters to be given to the kernel
    ActivationParams result;

    switch (activation.op_type) {
        case UnaryOpType::RELU: result.pack_relu = true; break;
        case UnaryOpType::MISH:
            result.type = KernelActivation::MISH;
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            break;
        case UnaryOpType::SQRT:
            result.type = KernelActivation::SQRT;
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            break;
        case UnaryOpType::LEAKY_RELU:
            TT_FATAL(has_first, "LEAKY_RELU requires a slope parameter");
            result.type = KernelActivation::LEAKY_RELU;
            result.param0 = std::bit_cast<uint32_t>(params[0]);
            break;
        case UnaryOpType::ELU:
            TT_FATAL(has_first, "ELU requires an alpha parameter");
            result.type = KernelActivation::ELU;
            result.param0 = std::bit_cast<uint32_t>(params[0]);
            break;
        case UnaryOpType::EXP:
            result.type = KernelActivation::EXP;
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            break;
        case UnaryOpType::RECIP: result.type = KernelActivation::RECIP; break;
        case UnaryOpType::GELU:
            result.type = KernelActivation::GELU;
            // param0 is vector mode (0=RC, 1=R, 2=C) or fast mode
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : static_cast<uint32_t>(default_fast_gelu);
            break;

        case UnaryOpType::GELU_TANH:
            result.type = KernelActivation::GELU_TANH;
            // No parameters
            break;

        case UnaryOpType::TANH:
            result.type = KernelActivation::TANH;
            // param0 is vector mode or fast mode
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            break;

        case UnaryOpType::SILU:
            result.type = KernelActivation::SILU;
            // No parameters currently
            break;

        case UnaryOpType::RELU6:
            result.type = KernelActivation::RELU6;
            // param0 is max value (default 6.0)
            result.param0 = has_first ? std::bit_cast<uint32_t>(params[0]) : 0x40c00000u;
            break;
        case UnaryOpType::SIGMOID: {
            result.type = KernelActivation::SIGMOID;
            const uint32_t param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            TT_FATAL(param0 <= 4, "Invalid Vector mode value: {}", param0);
            result.param0 = param0;
            // param1 is fast_approximate flag
            result.param1 = has_second ? static_cast<uint32_t>(params[1]) : 0;
            break;
        }

        case UnaryOpType::HARDSIGMOID:
            result.type = KernelActivation::HARDSIGMOID;
            // param0 could support approximation mode
            result.param0 = has_first ? static_cast<uint32_t>(params[0]) : 0;
            break;

        case UnaryOpType::HARDTANH:
            result.type = KernelActivation::HARDTANH;
            // param0 is min value (default -1.0)
            result.param0 = has_first ? std::bit_cast<uint32_t>(params[0]) : 0xbf800000u;
            // param1 is max value (default 1.0)
            result.param1 = has_second ? std::bit_cast<uint32_t>(params[1]) : 0x3f800000u;
            break;

        case UnaryOpType::SELU:
            result.type = KernelActivation::SELU;
            // selu(x) is scale * x for x >= 0, and scale * alpha * (exp(x) - 1) for x < 0.
            // selu_tile_pack takes the scale first and alpha second.
            //
            // Each default is the nearest float to the published SELU constant. Shown below as
            // the published value and the exact value of the float it rounds to, which is what
            // the bit patterns hold:
            //   scale  1.0507009873554804934193349852946 -> 1.05070102214813232421875
            //   alpha  1.6732632423543772848170429916717 -> 1.67326319217681884765625
            // param0 is scale
            result.param0 = has_first ? std::bit_cast<uint32_t>(params[0]) : 0x3f867d5fu;
            // param1 is alpha
            result.param1 = has_second ? std::bit_cast<uint32_t>(params[1]) : 0x3fd62d7du;
            break;

        case UnaryOpType::SOFTPLUS:
            result.type = KernelActivation::SOFTPLUS;
            // param0 is beta (default 1.0)
            // param1 is threshold (default 20.0)
            // we also prepare beta reciprocal as a kernel compile arg,
            // which is passed as param2
            if (has_first) {
                float beta = params[0];
                TT_FATAL(beta != 0, "SOFTPLUS activation beta parameter cannot be zero");
                float beta_reciprocal = 1.0f / params[0];
                result.param0 = std::bit_cast<uint32_t>(beta);
                result.param2 = std::bit_cast<uint32_t>(beta_reciprocal);
            } else {
                result.param0 = 0x3f800000u;
                result.param2 = 0x3f800000u;
            }
            result.param1 = has_second ? std::bit_cast<uint32_t>(params[1]) : 0x41a00000u;
            break;

        default: TT_THROW("Unsupported UnaryOpType for fused activation: {}", activation.op_type);
    }

    return result;
}

// Conv's unary API defaults parameterless GELU to fast mode; matmul defaults to exact.
// All other parameter encoding and supported operations are shared.
inline ActivationParams get_activation_kernel_config(
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation) {
    return activation ? get_activation_params(*activation, /*default_fast_gelu=*/true) : ActivationParams{};
}

}  // namespace compute_kernel_lib
