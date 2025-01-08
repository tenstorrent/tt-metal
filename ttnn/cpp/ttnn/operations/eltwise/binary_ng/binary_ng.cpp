
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng.hpp"
#include "device/binary_ng_device_operation.hpp"

namespace ttnn::operations::binary_ng {

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    auto input_a = typecast_to(DataType::BFLOAT16, input_tensor_a);
    auto input_b = typecast_to(DataType::BFLOAT16, input_tensor_b);

    return ttnn::prim::binary_ng(
        queue_id,
        input_a,
        input_b,
        binary_op_type,
        output_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    return invoke(
        DefaultQueueId,
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    float scalar,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    auto input_a = typecast_to(DataType::BFLOAT16, input_tensor_a);

    return ttnn::prim::binary_ng(
        queue_id,
        input_a,
        scalar,
        binary_op_type,
        output_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    const Tensor& input_tensor_a,
    float scalar,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    return invoke(
        DefaultQueueId,
        input_tensor_a,
        scalar,
        output_dtype,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    auto input_a = typecast_to(DataType::BFLOAT16, input_tensor_a);
    auto input_b = typecast_to(DataType::BFLOAT16, input_tensor_b);

    return ttnn::prim::binary_ng(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        binary_op_type,
        output_dtype,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryNg<binary_op_type>::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    return invoke(
        DefaultQueueId,
        input_tensor_a,
        input_tensor_b,
        output_dtype,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const float scalar,
    const std::optional<const DataType>& output_dtype,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    auto input_a = typecast_to(DataType::BFLOAT16, input_tensor_a);

    return ttnn::prim::binary_ng(
        queue_id,
        input_tensor_a,
        scalar,
        binary_op_type,
        output_dtype,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryNg<binary_op_type>::invoke(
    const Tensor& input_tensor_a,
    const float scalar,
    const std::optional<const DataType>& output_dtype,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryOpType> post_activations) {
    return invoke(
        DefaultQueueId, input_tensor_a, scalar, output_dtype, lhs_activations, rhs_activations, post_activations);
}

template struct BinaryNg<BinaryOpType::ADD>;
template struct BinaryNg<BinaryOpType::SUB>;
template struct BinaryNg<BinaryOpType::MUL>;
template struct BinaryNg<BinaryOpType::DIV>;
template struct BinaryNg<BinaryOpType::GT>;
template struct BinaryNg<BinaryOpType::LT>;
template struct BinaryNg<BinaryOpType::LTE>;
template struct BinaryNg<BinaryOpType::GTE>;
template struct BinaryNg<BinaryOpType::EQ>;
template struct BinaryNg<BinaryOpType::NE>;
template struct BinaryNg<BinaryOpType::SQUARED_DIFFERENCE>;
template struct BinaryNg<BinaryOpType::BIAS_GELU>;
template struct BinaryNg<BinaryOpType::LOGICAL_AND>;
template struct BinaryNg<BinaryOpType::LOGICAL_OR>;
template struct BinaryNg<BinaryOpType::LOGICAL_XOR>;
template struct BinaryNg<BinaryOpType::LDEXP>;
template struct BinaryNg<BinaryOpType::LOGADDEXP>;
template struct BinaryNg<BinaryOpType::LOGADDEXP2>;
template struct InplaceBinaryNg<BinaryOpType::ADD>;
template struct InplaceBinaryNg<BinaryOpType::SUB>;
template struct InplaceBinaryNg<BinaryOpType::MUL>;
template struct InplaceBinaryNg<BinaryOpType::GT>;
template struct InplaceBinaryNg<BinaryOpType::LT>;
template struct InplaceBinaryNg<BinaryOpType::LTE>;
template struct InplaceBinaryNg<BinaryOpType::GTE>;
template struct InplaceBinaryNg<BinaryOpType::EQ>;
template struct InplaceBinaryNg<BinaryOpType::NE>;
template struct InplaceBinaryNg<BinaryOpType::LOGICAL_AND>;
template struct InplaceBinaryNg<BinaryOpType::LOGICAL_OR>;
template struct InplaceBinaryNg<BinaryOpType::LOGICAL_XOR>;

}  // namespace ttnn::operations::binary_ng
