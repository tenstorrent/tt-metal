// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_composite_op.hpp"
#include "ttnn/operations/eltwise/binary/device/binary_device_operation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn {

namespace operations {

namespace binary {

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOps
{
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, memory_config);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsFloat
{
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        float alpha,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, alpha, memory_config);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsIsClose
{
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        float rtol,
        float atol,
        const bool equal_nan,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, rtol, atol, equal_nan, memory_config);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteDivLikeOps
{
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, input_tensor_b, memory_config);
    }
    static Tensor invoke(
        const Tensor& input_tensor_a,
        float value,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, value, memory_config);
    }
};

struct ExecuteDiv
{
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        bool accurate_mode = false,
        const std::string& round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        float value,
        bool accurate_mode = false,
        const std::string& round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        bool accurate_mode = false,
        const std::string& round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        float value,
        bool accurate_mode = false,
        const std::string& round_mode = "None",
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct ExecuteBiasGelu {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt) {

            return BinaryOperation<binary_op_type>::invoke(
                queue_id, input_tensor_a_arg, input_tensor_b_arg, output_dtype, memory_config, optional_output_tensor, activations, input_tensor_a_activation);
    }

    static Tensor invoke(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt) {

            return BinaryOperation<binary_op_type>::invoke(
                DefaultQueueId, input_tensor_a_arg, input_tensor_b_arg, output_dtype, memory_config, optional_output_tensor, activations, input_tensor_a_activation);
    }

    static Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_a,
        const float bias,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt) {

            return ttnn::gelu(queue_id, ttnn::add(queue_id, input_tensor_a, bias, std::nullopt, memory_config, optional_output_tensor), true, memory_config, optional_output_tensor);
    }

    static Tensor invoke(
        const ttnn::Tensor &input_tensor_a,
        const float bias,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt) {

            return invoke(DefaultQueueId, input_tensor_a, bias, dtype, memory_config, optional_output_tensor, activations, input_tensor_a_activation);
    }
};

template <BinaryCompositeOpType binary_comp_op_type>
struct ExecuteBinaryCompositeOpsPolyval
{
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const std::vector<float>& coeffs,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return OpHandler<binary_comp_op_type>::handle(input_tensor_a, coeffs, memory_config);
    }
};

struct ExecuteBinaryFmod
{
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        float scalar,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

struct ExecuteBinaryRemainder
{
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        float scalar,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace binary
}  // namespace operations

constexpr auto hypot = ttnn::register_operation_with_auto_launch_op<
    "ttnn::hypot",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::HYPOT>>();
constexpr auto xlogy = ttnn::register_operation_with_auto_launch_op<
    "ttnn::xlogy",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::XLOGY>>();
constexpr auto minimum = ttnn::register_operation_with_auto_launch_op<
    "ttnn::minimum",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::MINIMUM>>();
constexpr auto maximum = ttnn::register_operation_with_auto_launch_op<
    "ttnn::maximum",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::MAXIMUM>>();
constexpr auto atan2 = ttnn::register_operation_with_auto_launch_op<
    "ttnn::atan2",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::ATAN2>>();
constexpr auto logical_xor = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_xor",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::LOGICAL_XOR>>();
constexpr auto nextafter = ttnn::register_operation_with_auto_launch_op<
    "ttnn::nextafter",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::NEXTAFTER>>();
constexpr auto addalpha = ttnn::register_operation_with_auto_launch_op<
    "ttnn::addalpha",
    operations::binary::ExecuteBinaryCompositeOpsFloat<operations::binary::BinaryCompositeOpType::ADDALPHA>>();
constexpr auto subalpha = ttnn::register_operation_with_auto_launch_op<
    "ttnn::subalpha",
    operations::binary::ExecuteBinaryCompositeOpsFloat<operations::binary::BinaryCompositeOpType::SUBALPHA>>();
constexpr auto isclose = ttnn::register_operation_with_auto_launch_op<
    "ttnn::isclose",
    operations::binary::ExecuteBinaryCompositeOpsIsClose<operations::binary::BinaryCompositeOpType::ISCLOSE>>();
constexpr auto remainder = ttnn::register_operation_with_auto_launch_op<
    "ttnn::remainder",
    operations::binary::ExecuteBinaryRemainder>();
constexpr auto fmod = ttnn::register_operation_with_auto_launch_op<
    "ttnn::fmod",
    operations::binary::ExecuteBinaryFmod>();
constexpr auto div = ttnn::register_operation_with_auto_launch_op<
    "ttnn::div",
    operations::binary::ExecuteDiv>();
constexpr auto div_no_nan = ttnn::register_operation_with_auto_launch_op<
    "ttnn::div_no_nan",
    operations::binary::ExecuteDivLikeOps<operations::binary::BinaryCompositeOpType::DIV_NO_NAN>>();
constexpr auto floor_div = ttnn::register_operation_with_auto_launch_op<
    "ttnn::floor_div",
    operations::binary::ExecuteDivLikeOps<operations::binary::BinaryCompositeOpType::FLOOR_DIV>>();
constexpr auto logical_xor_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_xor_",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::LOGICAL_XOR_>>();
constexpr auto bias_gelu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::bias_gelu",
    operations::binary::ExecuteBiasGelu<operations::binary::BinaryOpType::BIAS_GELU>>();
constexpr auto scatter = ttnn::register_operation_with_auto_launch_op<
    "ttnn::scatter",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::SCATTER>>();
constexpr auto outer = ttnn::register_operation_with_auto_launch_op<
    "ttnn::outer",
    operations::binary::ExecuteBinaryCompositeOps<operations::binary::BinaryCompositeOpType::OUTER>>();
constexpr auto polyval = ttnn::register_operation_with_auto_launch_op<
    "ttnn::polyval",
    operations::binary::ExecuteBinaryCompositeOpsPolyval<operations::binary::BinaryCompositeOpType::POLYVAL>>();

}  // namespace ttnn
