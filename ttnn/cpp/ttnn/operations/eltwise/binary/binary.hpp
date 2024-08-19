
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/eltwise/binary/common/binary_op_types.hpp"
#include "device/binary_device_operation.hpp"

namespace ttnn {

namespace operations {
namespace unary {
struct UnaryWithParam;
}
namespace binary {

template <BinaryOpType binary_op_type>
struct BinaryOperation {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    // TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
    // Currently, this is exactly how tt::tt_metal::add_unary works
    static Tensor invoke(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct RelationalBinary {
    static Tensor invoke(
        uint8_t queue_id,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    // scalar - tensor combination not available on Pytorch for this op
    static Tensor invoke(
        uint8_t queue_id,
        const float scalar,
        const ttnn::Tensor &input_tensor_a,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt);
};

template <BinaryOpType binary_op_type>
struct InplaceRelationalBinary {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b);

    static Tensor invoke(
        const Tensor& input_tensor,
        const float scalar);
};

template <BinaryOpType binary_op_type>
struct InplaceLogicalBinary {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b);
};

template <BinaryOpType binary_op_type>
struct InplaceBinaryOperation {
    static Tensor invoke(
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);

    static Tensor invoke(
        const Tensor& input_tensor,
        const float scalar,
        std::optional<unary::FusedActivations> activations = std::nullopt,
        std::optional<unary::UnaryWithParam> input_tensor_a_activation = std::nullopt);
};

}  // binary
}  // operations

constexpr auto add = ttnn::register_operation_with_auto_launch_op<
    "ttnn::add",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::ADD>>();
constexpr auto add_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::add_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::ADD>>();
constexpr auto subtract = ttnn::register_operation<
    "ttnn::subtract",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::SUB>>();
constexpr auto subtract_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::subtract_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::SUB>>();
constexpr auto multiply = ttnn::register_operation<
    "ttnn::multiply",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::MUL>>();
constexpr auto multiply_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::multiply_",
    operations::binary::InplaceBinaryOperation<operations::binary::BinaryOpType::MUL>>();
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
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGICAL_AND>>();
constexpr auto logical_or = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_or",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGICAL_OR>>();
constexpr auto ldexp = ttnn::register_operation_with_auto_launch_op<
    "ttnn::ldexp",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LDEXP>>();
constexpr auto logaddexp = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logaddexp",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGADDEXP>>();
constexpr auto logaddexp2 = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logaddexp2",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::LOGADDEXP2>>();
constexpr auto squared_difference = ttnn::register_operation_with_auto_launch_op<
    "ttnn::squared_difference",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::SQUARED_DIFFERENCE>>();
constexpr auto divide = ttnn::register_operation<
    "ttnn::divide",
    operations::binary::BinaryOperation<operations::binary::BinaryOpType::DIV_FAST>>();
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
constexpr auto logical_and_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_and_",
    operations::binary::InplaceLogicalBinary<operations::binary::BinaryOpType::LOGICAL_AND>>();
constexpr auto logical_or_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::logical_or_",
    operations::binary::InplaceLogicalBinary<operations::binary::BinaryOpType::LOGICAL_OR>>();
constexpr auto eq_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::eq_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::EQ>>();
constexpr auto ne_ = ttnn::register_operation_with_auto_launch_op<
    "ttnn::ne_",
    operations::binary::InplaceRelationalBinary<operations::binary::BinaryOpType::NE>>();

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
