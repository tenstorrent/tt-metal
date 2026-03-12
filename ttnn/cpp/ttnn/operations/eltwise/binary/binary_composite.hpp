// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_composite_op.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {

namespace operations::binary {

/**
 * @brief Performs element-wise power operation on the input with the exponent.
 * When exponent is Tensor, the supported dtypes are float32 and bfloat16.
 * The tested range for the input is (-30,30) and for the exponent is (-20, 20).
 *
 * @param input The input tensor, i.e the base.
 * @param exponent The exponent
 * @return The result tensor
 */
struct ExecutePower {
    static Tensor invoke(
        const Tensor& input_tensor,
        int32_t exponent,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& input_a,
        float exponent,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static Tensor invoke(
        float input_a,
        const Tensor& exponent,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const Tensor& exponent,
        const std::optional<const DataType>& dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOps {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, memory_config);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsIsClose {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        float rtol,
        float atol,
        const bool equal_nan,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(
            input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteDivLikeOps {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, memory_config);
    }
    static Tensor invoke(
        const Tensor& input_tensor_a, float value, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, value, memory_config);
    }
};

struct ExecuteDiv {
    static Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        bool fast_and_approximate_mode = false,
        const std::optional<std::string>& rounding_mode = std::nullopt,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<Tensor>& output_tensor = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const Tensor& input,
        float value,
        bool fast_and_approximate_mode = false,
        const std::optional<std::string>& rounding_mode = std::nullopt,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        std::optional<Tensor> output = std::nullopt,
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const ttnn::operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
        const std::optional<bool>& use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct ExecuteBiasGelu {
    static Tensor invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
        return BinaryOperation<binary_op_type>::invoke(
            input_tensor_a_arg,
            input_tensor_b_arg,
            output_dtype,
            memory_config,
            optional_output_tensor,
            post_activations,
            lhs_activations,
            rhs_activations,
            use_legacy,
            sub_core_grids);
    }

    static Tensor invoke(
        const ttnn::Tensor& input_tensor_a,
        const float bias,
        const std::optional<const DataType>& /*dtype*/ = std::nullopt,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> /*post_activations*/ = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> /*lhs_activations*/ = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> /*rhs_activations*/ = {},
        std::optional<bool> /*use_legacy*/ = std::nullopt,
        const std::optional<CoreRangeSet>& /*sub_core_grids*/ = std::nullopt) {
        return ttnn::gelu(
            ttnn::add(input_tensor_a, bias, std::nullopt, memory_config, optional_output_tensor),
            true,
            memory_config,
            optional_output_tensor);
    }

    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
        return BinaryOperation<binary_op_type>::invoke(
            input_tensor_a,
            input_tensor_b,
            output_dtype,
            memory_config,
            output,
            lhs_activations,
            rhs_activations,
            post_activations,
            sub_core_grids);
    }

    static Tensor invoke(
        const Tensor& input_tensor_a,
        float scalar,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& output = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {}) {
        return BinaryOperation<binary_op_type>::invoke(
            input_tensor_a,
            scalar,
            output_dtype,
            memory_config,
            output,
            lhs_activations,
            rhs_activations,
            post_activations);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsPolyval {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const std::vector<float>& coeffs,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, coeffs, memory_config);
    }
};

struct ExecuteBinaryFmod {
    static Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        float scalar,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

struct ExecuteBinaryRemainder {
    static Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        float scalar,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

struct ExecuteLCM {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteGCD {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteMaximum {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_a,
        unary::ScalarVariant value,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteMinimum {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_a,
        unary::ScalarVariant value,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecutePrelu {
    static Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const std::array<float, 1>& weight,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor, float weight, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);
};

struct ExecuteRsub {
    static Tensor invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        float input_b,
        const std::optional<const DataType>& output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteBitwiseAnd {
    static Tensor invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        int32_t input_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteBitwiseOr {
    static Tensor invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        int32_t input_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteBitwiseXor {
    static Tensor invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        int32_t input_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteBitwiseLeftShift {
    static Tensor invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        int32_t input_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteBitwiseRightShift {
    static Tensor invoke(
        const Tensor& input_tensor_a_arg,
        const Tensor& input_tensor_b_arg,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        int32_t input_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        tt::stl::Span<const unary::EltwiseUnaryWithParam> post_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> lhs_activations = {},
        tt::stl::Span<const unary::EltwiseUnaryWithParam> rhs_activations = {},
        std::optional<bool> use_legacy = std::nullopt);
};

struct ExecuteLogicalLeftShift : ExecuteBitwiseLeftShift {
    // Inherits all functionality from ExecuteBitwiseLeftShift
    // but creates a distinct type for registration
};

}  // namespace operations::binary

inline Tensor bias_gelu(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::ExecuteBiasGelu<operations::binary::BinaryOpType::BIAS_GELU>::invoke(
        input_tensor_a_arg,
        input_tensor_b_arg,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}
inline Tensor bias_gelu(
    const Tensor& input_tensor_a,
    float bias,
    const std::optional<const DataType>& /*dtype*/ = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> /*post_activations*/ = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> /*lhs_activations*/ = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> /*rhs_activations*/ = {},
    std::optional<bool> /*use_legacy*/ = std::nullopt,
    const std::optional<CoreRangeSet>& /*sub_core_grids*/ = std::nullopt) {
    return ttnn::gelu(
        ttnn::add(input_tensor_a, bias, std::nullopt, memory_config, optional_output_tensor),
        true,
        memory_config,
        optional_output_tensor);
}
inline Tensor bias_gelu(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::ExecuteBiasGelu<operations::binary::BinaryOpType::BIAS_GELU>::invoke(
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        std::nullopt,
        sub_core_grids);
}

inline Tensor minimum(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteMinimum::invoke(
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor minimum(
    const Tensor& input_a,
    operations::unary::ScalarVariant value,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteMinimum::invoke(
        input_a,
        value,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor maximum(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteMaximum::invoke(
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor maximum(
    const Tensor& input_a,
    operations::unary::ScalarVariant value,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteMaximum::invoke(
        input_a,
        value,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor gcd(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteGCD::invoke(
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor lcm(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteLCM::invoke(
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor atan2(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::ATAN2>::invoke(
        input_tensor_a, input_tensor_b, memory_config);
}

inline Tensor nextafter(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::NEXTAFTER>::invoke(
        input_tensor_a, input_tensor_b, memory_config);
}

inline Tensor isclose(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    float rtol,
    float atol,
    const bool equal_nan,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteBinaryCompositeOpsIsClose<operations::binary::BinaryCompositeOpType::ISCLOSE>::
        invoke(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
}

inline Tensor div_no_nan(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteDivLikeOps<operations::binary::BinaryCompositeOpType::DIV_NO_NAN>::invoke(
        input_tensor_a, input_tensor_b, memory_config);
}

inline Tensor div_no_nan(
    const Tensor& input_tensor_a, float value, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteDivLikeOps<operations::binary::BinaryCompositeOpType::DIV_NO_NAN>::invoke(
        input_tensor_a, value, memory_config);
}

inline Tensor floor_div(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteDivLikeOps<operations::binary::BinaryCompositeOpType::FLOOR_DIV>::invoke(
        input_tensor_a, input_tensor_b, memory_config);
}

inline Tensor floor_div(
    const Tensor& input_tensor_a, float value, const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteDivLikeOps<operations::binary::BinaryCompositeOpType::FLOOR_DIV>::invoke(
        input_tensor_a, value, memory_config);
}

inline Tensor outer(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::OUTER>::invoke(
        input_tensor_a, input_tensor_b, memory_config);
}

inline Tensor div(
    const Tensor& input_a,
    const Tensor& input_b,
    bool fast_and_approximate_mode = false,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::ExecuteDiv::invoke(
        input_a,
        input_b,
        fast_and_approximate_mode,
        rounding_mode,
        output_dtype,
        output_mem_config,
        output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}

inline Tensor div(
    const Tensor& input,
    float value,
    bool fast_and_approximate_mode = false,
    const std::optional<std::string>& rounding_mode = std::nullopt,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    std::optional<Tensor> output = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    const std::optional<bool>& use_legacy = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::ExecuteDiv::invoke(
        input,
        value,
        fast_and_approximate_mode,
        rounding_mode,
        output_dtype,
        output_mem_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy,
        sub_core_grids);
}

inline Tensor remainder(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::ExecuteBinaryRemainder::invoke(input_a, input_b, output_mem_config, sub_core_grids);
}

inline Tensor remainder(
    const Tensor& input_tensor,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::ExecuteBinaryRemainder::invoke(input_tensor, scalar, output_mem_config, sub_core_grids);
}

inline Tensor fmod(
    const Tensor& input_a,
    const Tensor& input_b,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::ExecuteBinaryFmod::invoke(input_a, input_b, output_mem_config, sub_core_grids);
}

inline Tensor fmod(
    const Tensor& input_tensor,
    float scalar,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    return operations::binary::ExecuteBinaryFmod::invoke(input_tensor, scalar, output_mem_config, sub_core_grids);
}

inline Tensor polyval(
    const Tensor& input_tensor_a,
    const std::vector<float>& coeffs,
    const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    return operations::binary::ExecuteBinaryCompositeOpsPolyval<
        operations::binary::BinaryCompositeOpType::POLYVAL>::invoke(input_tensor_a, coeffs, memory_config);
}

inline Tensor prelu(
    const Tensor& input_a, const Tensor& input_b, const std::optional<MemoryConfig>& output_mem_config = std::nullopt) {
    return operations::binary::ExecutePrelu::invoke(input_a, input_b, output_mem_config);
}

inline Tensor prelu(
    const Tensor& input_tensor,
    const std::array<float, 1>& weight,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt) {
    return operations::binary::ExecutePrelu::invoke(input_tensor, weight, output_mem_config);
}

inline Tensor prelu(
    const Tensor& input_tensor, float weight, const std::optional<MemoryConfig>& output_mem_config = std::nullopt) {
    return operations::binary::ExecutePrelu::invoke(input_tensor, weight, output_mem_config);
}

inline Tensor rsub(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteRsub::invoke(
        input_tensor_a_arg,
        input_tensor_b_arg,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor rsub(
    const Tensor& input_tensor,
    float input_b,
    const std::optional<const DataType>& output_dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteRsub::invoke(
        input_tensor,
        input_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_and(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseAnd::invoke(
        input_tensor_a_arg,
        input_tensor_b_arg,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_and(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseAnd::invoke(
        input_tensor,
        input_b,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_or(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseOr::invoke(
        input_tensor_a_arg,
        input_tensor_b_arg,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_or(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseOr::invoke(
        input_tensor,
        input_b,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_xor(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseXor::invoke(
        input_tensor_a_arg,
        input_tensor_b_arg,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_xor(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseXor::invoke(
        input_tensor,
        input_b,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_left_shift(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseLeftShift::invoke(
        input_tensor_a_arg,
        input_tensor_b_arg,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_left_shift(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseLeftShift::invoke(
        input_tensor,
        input_b,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor logical_left_shift(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteLogicalLeftShift::invoke(
        input_tensor_a_arg,
        input_tensor_b_arg,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor logical_left_shift(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteLogicalLeftShift::invoke(
        input_tensor,
        input_b,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_right_shift(
    const Tensor& input_tensor_a_arg,
    const Tensor& input_tensor_b_arg,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseRightShift::invoke(
        input_tensor_a_arg,
        input_tensor_b_arg,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor bitwise_right_shift(
    const Tensor& input_tensor,
    int32_t input_b,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecuteBitwiseRightShift::invoke(
        input_tensor,
        input_b,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor pow(
    const Tensor& input_tensor,
    int32_t exponent,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return operations::binary::ExecutePower::invoke(input_tensor, exponent, output_mem_config, optional_output_tensor);
}

inline Tensor pow(
    const Tensor& input_a,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
    return operations::binary::ExecutePower::invoke(input_a, exponent, output_mem_config, optional_output_tensor);
}

inline Tensor pow(
    float input_a,
    const Tensor& exponent,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecutePower::invoke(
        input_a,
        exponent,
        dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

inline Tensor pow(
    const Tensor& input_tensor,
    const Tensor& exponent,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> post_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> lhs_activations = {},
    tt::stl::Span<const operations::unary::EltwiseUnaryWithParam> rhs_activations = {},
    std::optional<bool> use_legacy = std::nullopt) {
    return operations::binary::ExecutePower::invoke(
        input_tensor,
        exponent,
        dtype,
        memory_config,
        optional_output_tensor,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

}  // namespace ttnn
