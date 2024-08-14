
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "device/binary_device_operation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"

namespace ttnn {

namespace operations {
namespace unary {
struct UnaryWithParam;
}
namespace binary {

template <BinaryOpType binary_op_type, bool in_place>
struct BinaryOperation {
    static Tensor operator()(
        uint8_t queue_id,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor operator()(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    // TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
    // Currently, this is exactly how tt::tt_metal::add_unary works
    static Tensor operator()(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor operator()(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);
};

template <BinaryOpType binary_op_type, bool in_place>
struct BinaryOperationOverload {
    static Tensor operator()(
        uint8_t queue_id,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor operator()(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    // TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
    // Currently, this is exactly how tt::tt_metal::add_unary works
    static Tensor operator()(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor operator()(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static ComplexTensor operator()(
        const ComplexTensor &input_tensor_a_arg,
        const ComplexTensor &input_tensor_b_arg,
        const MemoryConfig &memory_config);
};

template <BinaryOpType binary_op_type>
struct RelationalBinary {
    static Tensor operator()(
        uint8_t queue_id,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor operator()(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor operator()(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor operator()(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    // scalar - tensor combination not available on Pytorch for this op
    static Tensor operator()(
        uint8_t queue_id,
        const float scalar,
        const ttnn::Tensor &input_tensor_a,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceRelationalBinary {
    static Tensor operator()(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b);

    static Tensor operator()(
        const Tensor& input_tensor,
        const float scalar);
};

}  // binary
}  // operations

constexpr auto add = ttnn::register_operation<
    "ttnn::add",
    operations::binary::BinaryOperationOverload<operations::binary::BinaryOpType::ADD, false>>();
constexpr auto add_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::add_",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::ADD, true>>();
constexpr auto subtract = ttnn::register_operation<
    "ttnn::subtract",
    operations::binary::BinaryOperationOverload<operations::binary::BinaryOpType::SUB, false>>();
constexpr auto subtract_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::subtract_",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::SUB, true>>();
constexpr auto multiply = ttnn::register_operation_with_auto_launch_op<
    "ttnn::multiply",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::MUL, false>>();
constexpr auto multiply_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::multiply_",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::MUL, true>>();
constexpr auto eq = ttnn::register_operation_with_auto_launch_op<
    "ttnn::eq",
    operations::binary::RelationalBinary<operations::binary::BinaryOpType::EQ>>();
constexpr auto ne = ttnn::register_operation_with_auto_launch_op<
    "ttnn::ne",
    operations::binary::RelationalBinary<operations::binary::BinaryOpType::NE>>();
constexpr auto ge = ttnn::register_operation_with_auto_launch_op<
    "ttnn::ge",
    operations::binary::RelationalBinary<operations::binary::BinaryOpType::GTE>>();
constexpr auto gt = ttnn::register_operation_with_auto_launch_op<
    "ttnn::gt",
    operations::binary::RelationalBinary<operations::binary::BinaryOpType::GT>>();
constexpr auto le = ttnn::register_operation_with_auto_launch_op<
    "ttnn::le",
    operations::binary::RelationalBinary<operations::binary::BinaryOpType::LTE>>();
constexpr auto lt = ttnn::register_operation_with_auto_launch_op<
    "ttnn::lt",
    operations::binary::RelationalBinary<operations::binary::BinaryOpType::LT>>();
constexpr auto logical_and = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_and",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGICAL_AND, false>>();
constexpr auto logical_or = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_or",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGICAL_OR, false>>();
constexpr auto ldexp = ttnn::register_operation_with_auto_launch_op<
    "ttnn::ldexp",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LDEXP, false>>();
constexpr auto logaddexp = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logaddexp",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGADDEXP, false>>();
constexpr auto logaddexp2 = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logaddexp2",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGADDEXP2, false>>();
constexpr auto squared_difference = ttnn::register_operation_with_auto_launch_op<
    "ttnn::squared_difference",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::SQUARED_DIFFERENCE, false>>();
constexpr auto divide = ttnn::register_operation_with_auto_launch_op<
    "ttnn::divide",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::DIV_FAST, false>>();
constexpr auto gt_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::gt_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::GT>>();
constexpr auto ge_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::ge_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::GTE>>();
constexpr auto le_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::le_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::LTE>>();
constexpr auto lt_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::lt_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::LT>>();

template <typename InputBType>
ttnn::Tensor operator+(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return add(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator-(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return subtract(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator*(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return multiply(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator==(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return eq(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator!=(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return ne(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator>(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return gt(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator>=(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return ge(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator<(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return lt(input_tensor_a, scalar);
}

template <typename InputBType>
ttnn::Tensor operator<=(const ttnn::Tensor &input_tensor_a, InputBType scalar) {
    return le(input_tensor_a, scalar);
}

}  // namespace ttnn
