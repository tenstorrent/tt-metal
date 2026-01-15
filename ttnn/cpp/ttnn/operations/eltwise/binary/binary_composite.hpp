// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

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
        uint32_t exponent,
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
        std::optional<Tensor> output = std::nullopt,
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

constexpr auto minimum = ttnn::register_operation<"ttnn::minimum", operations::binary::ExecuteMinimum>();
constexpr auto maximum = ttnn::register_operation<"ttnn::maximum", operations::binary::ExecuteMaximum>();
constexpr auto atan2 = ttnn::register_operation<
    "ttnn::atan2",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::ATAN2>>();
constexpr auto nextafter = ttnn::register_operation<
    "ttnn::nextafter",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::NEXTAFTER>>();
constexpr auto isclose = ttnn::register_operation<
    "ttnn::isclose",
    operations::binary::ExecuteBinaryCompositeOpsIsClose<operations::binary::BinaryCompositeOpType::ISCLOSE>>();
constexpr auto remainder = ttnn::register_operation<"ttnn::remainder", operations::binary::ExecuteBinaryRemainder>();
constexpr auto fmod = ttnn::register_operation<"ttnn::fmod", operations::binary::ExecuteBinaryFmod>();
constexpr auto div = ttnn::register_operation<"ttnn::div", operations::binary::ExecuteDiv>();
constexpr auto div_no_nan = ttnn::register_operation<
    "ttnn::div_no_nan",
    operations::binary::ExecuteDivLikeOps<operations::binary::BinaryCompositeOpType::DIV_NO_NAN>>();
constexpr auto floor_div = ttnn::register_operation<
    "ttnn::floor_div",
    operations::binary::ExecuteDivLikeOps<operations::binary::BinaryCompositeOpType::FLOOR_DIV>>();
constexpr auto bias_gelu = ttnn::register_operation<
    "ttnn::bias_gelu",
    operations::binary::ExecuteBiasGelu<operations::binary::BinaryOpType::BIAS_GELU>>();
constexpr auto outer = ttnn::register_operation<
    "ttnn::outer",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::OUTER>>();
constexpr auto polyval = ttnn::register_operation<
    "ttnn::polyval",
    operations::binary::ExecuteBinaryCompositeOpsPolyval<operations::binary::BinaryCompositeOpType::POLYVAL>>();
constexpr auto gcd = ttnn::register_operation<"ttnn::gcd", operations::binary::ExecuteGCD>();
constexpr auto lcm = ttnn::register_operation<"ttnn::lcm", operations::binary::ExecuteLCM>();
constexpr auto prelu = ttnn::register_operation<"ttnn::prelu", operations::binary::ExecutePrelu>();
constexpr auto rsub = ttnn::register_operation<"ttnn::rsub", operations::binary::ExecuteRsub>();
constexpr auto bitwise_and = ttnn::register_operation<"ttnn::bitwise_and", operations::binary::ExecuteBitwiseAnd>();
constexpr auto bitwise_or = ttnn::register_operation<"ttnn::bitwise_or", operations::binary::ExecuteBitwiseOr>();
constexpr auto bitwise_xor = ttnn::register_operation<"ttnn::bitwise_xor", operations::binary::ExecuteBitwiseXor>();
constexpr auto bitwise_left_shift =
    ttnn::register_operation<"ttnn::bitwise_left_shift", operations::binary::ExecuteBitwiseLeftShift>();
constexpr auto logical_left_shift =
    ttnn::register_operation<"ttnn::logical_left_shift", operations::binary::ExecuteLogicalLeftShift>();
constexpr auto bitwise_right_shift =
    ttnn::register_operation<"ttnn::bitwise_right_shift", operations::binary::ExecuteBitwiseRightShift>();
constexpr auto pow = ttnn::register_operation<"ttnn::pow", operations::binary::ExecutePower>();

}  // namespace ttnn
