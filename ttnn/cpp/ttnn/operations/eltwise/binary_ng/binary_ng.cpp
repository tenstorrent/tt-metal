
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary_ng.hpp"
#include "device/binary_ng_device_operation.hpp"

inline Tensor typecast_to(DataType dtype, const Tensor& input) {
    return input.get_dtype() == dtype ? input : ttnn::typecast(input, dtype);
}

inline bool needs_typecast_to_bfloat16(const Tensor& input) {
    return (input.get_dtype() == DataType::BFLOAT8_B || input.get_dtype() == DataType::BFLOAT4_B);
}

namespace ttnn::operations::binary_ng {

template <BinaryOpType binary_op_type>
Tensor BinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const DataType>& output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
    bool typecast_a = needs_typecast_to_bfloat16(input_tensor_a);
    bool typecast_b = needs_typecast_to_bfloat16(input_tensor_b);
    Tensor input_a = typecast_a ? typecast_to(DataType::BFLOAT16, input_tensor_a) : input_tensor_a;
    Tensor input_b = typecast_b ? typecast_to(DataType::BFLOAT16, input_tensor_b) : input_tensor_b;

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
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
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
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
    bool typecast_a = needs_typecast_to_bfloat16(input_tensor_a);
    Tensor input_a = typecast_a ? typecast_to(DataType::BFLOAT16, input_tensor_a) : input_tensor_a;

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
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
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
Tensor BinaryNgBitwise<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    TT_FATAL(
        input_tensor_a.get_dtype() == DataType::INT32 && input_tensor_b.get_dtype() == DataType::INT32,
        "Bitwise ops require input tensors to be of INT32 datatype ");

    tt::stl::Span<const unary::UnaryWithParam> lhs_activations = {};
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations = {};
    tt::stl::Span<const unary::UnaryWithParam> post_activations = {};

    return ttnn::prim::binary_ng(
        queue_id,
        input_tensor_a,
        input_tensor_b,
        binary_op_type,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNgBitwise<binary_op_type>::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return BinaryNgBitwise<binary_op_type>::invoke(
        DefaultQueueId, input_tensor_a, input_tensor_b, memory_config, optional_output_tensor);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNgBitwise<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    float scalar,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    TT_FATAL(
        input_tensor_a.get_dtype() == DataType::INT32, "Bitwise ops require input tensor to be of INT32 datatype ");

    tt::stl::Span<const unary::UnaryWithParam> lhs_activations = {};
    tt::stl::Span<const unary::UnaryWithParam> rhs_activations = {};
    tt::stl::Span<const unary::UnaryWithParam> post_activations = {};

    return ttnn::prim::binary_ng(
        queue_id,
        input_tensor_a,
        scalar,
        binary_op_type,
        std::nullopt,
        memory_config,
        optional_output_tensor,
        lhs_activations,
        rhs_activations,
        post_activations);
}

template <BinaryOpType binary_op_type>
Tensor BinaryNgBitwise<binary_op_type>::invoke(
    const Tensor& input_tensor_a,
    float scalar,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    return BinaryNgBitwise<binary_op_type>::invoke(
        DefaultQueueId, input_tensor_a, scalar, memory_config, optional_output_tensor);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
    bool typecast_a = needs_typecast_to_bfloat16(input_tensor_a);
    bool typecast_b = needs_typecast_to_bfloat16(input_tensor_b);

    if (!typecast_a && !typecast_b) {
        return ttnn::prim::binary_ng(
            queue_id,
            input_tensor_a,
            input_tensor_b,
            binary_op_type,
            input_tensor_a.get_dtype(),
            input_tensor_a.memory_config(),
            input_tensor_a,
            lhs_activations,
            rhs_activations,
            post_activations);
    } else {
        Tensor input_a = typecast_to(DataType::BFLOAT16, input_tensor_a);
        Tensor input_b = typecast_to(DataType::BFLOAT16, input_tensor_b);

        ttnn::prim::binary_ng(
            queue_id,
            input_a,
            input_b,
            binary_op_type,
            input_a.get_dtype(),
            input_a.memory_config(),
            input_a,
            lhs_activations,
            rhs_activations,
            post_activations);

        if (typecast_a) {
            copy::detail::copy_impl(
                queue_id,
                input_a,
                {ttnn::operations::unary::UnaryWithParam(
                    ttnn::operations::unary::UnaryOpType::TYPECAST,
                    {static_cast<float>(input_a.get_dtype()), static_cast<float>(input_tensor_a.get_dtype())})},
                input_tensor_a.memory_config(),
                input_tensor_a);

            return input_tensor_a;
        }
        return input_tensor_a;
    }
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryNg<binary_op_type>::invoke(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
    return InplaceBinaryNg<binary_op_type>::invoke(
        DefaultQueueId, input_tensor_a, input_tensor_b, lhs_activations, rhs_activations, post_activations);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryNg<binary_op_type>::invoke(
    uint8_t queue_id,
    const Tensor& input_tensor_a,
    float scalar,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
    bool typecast_a = needs_typecast_to_bfloat16(input_tensor_a);

    if (!typecast_a) {
        return ttnn::prim::binary_ng(
            queue_id,
            input_tensor_a,
            scalar,
            binary_op_type,
            input_tensor_a.get_dtype(),
            input_tensor_a.memory_config(),
            input_tensor_a,
            lhs_activations,
            rhs_activations,
            post_activations);
    } else {
        Tensor input_a = typecast_to(DataType::BFLOAT16, input_tensor_a);
        ttnn::prim::binary_ng(
            queue_id,
            input_a,
            scalar,
            binary_op_type,
            input_a.get_dtype(),
            input_a.memory_config(),
            input_a,
            lhs_activations,
            rhs_activations,
            post_activations);

        copy::detail::copy_impl(
            queue_id,
            input_a,
            {ttnn::operations::unary::UnaryWithParam(
                ttnn::operations::unary::UnaryOpType::TYPECAST,
                {static_cast<float>(input_a.get_dtype()), static_cast<float>(input_tensor_a.get_dtype())})},
            input_tensor_a.memory_config(),
            input_tensor_a);

        return input_tensor_a;
    }
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryNg<binary_op_type>::invoke(
    const Tensor& input_tensor_a,
    float scalar,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations) {
    return InplaceBinaryNg<binary_op_type>::invoke(
        DefaultQueueId, input_tensor_a, scalar, lhs_activations, rhs_activations, post_activations);
}

template struct BinaryNg<BinaryOpType::ADD>;
template struct BinaryNg<BinaryOpType::SUB>;
template struct BinaryNg<BinaryOpType::MUL>;
template struct BinaryNg<BinaryOpType::DIV>;
template struct BinaryNg<BinaryOpType::RSUB>;
template struct BinaryNg<BinaryOpType::POWER>;
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

template struct BinaryNgBitwise<BinaryOpType::BITWISE_AND>;
template struct BinaryNgBitwise<BinaryOpType::BITWISE_OR>;
template struct BinaryNgBitwise<BinaryOpType::BITWISE_XOR>;
template struct BinaryNgBitwise<BinaryOpType::LEFT_SHIFT>;
template struct BinaryNgBitwise<BinaryOpType::RIGHT_SHIFT>;

template struct InplaceBinaryNg<BinaryOpType::ADD>;
template struct InplaceBinaryNg<BinaryOpType::SUB>;
template struct InplaceBinaryNg<BinaryOpType::MUL>;
template struct InplaceBinaryNg<BinaryOpType::DIV>;
template struct InplaceBinaryNg<BinaryOpType::RSUB>;
template struct InplaceBinaryNg<BinaryOpType::POWER>;
template struct InplaceBinaryNg<BinaryOpType::GT>;
template struct InplaceBinaryNg<BinaryOpType::LT>;
template struct InplaceBinaryNg<BinaryOpType::LTE>;
template struct InplaceBinaryNg<BinaryOpType::GTE>;
template struct InplaceBinaryNg<BinaryOpType::EQ>;
template struct InplaceBinaryNg<BinaryOpType::NE>;
template struct InplaceBinaryNg<BinaryOpType::SQUARED_DIFFERENCE>;
template struct InplaceBinaryNg<BinaryOpType::BIAS_GELU>;
template struct InplaceBinaryNg<BinaryOpType::LOGICAL_AND>;
template struct InplaceBinaryNg<BinaryOpType::LOGICAL_OR>;
template struct InplaceBinaryNg<BinaryOpType::LOGICAL_XOR>;
template struct InplaceBinaryNg<BinaryOpType::LDEXP>;
template struct InplaceBinaryNg<BinaryOpType::LOGADDEXP>;
template struct InplaceBinaryNg<BinaryOpType::LOGADDEXP2>;

}  // namespace ttnn::operations::binary_ng
