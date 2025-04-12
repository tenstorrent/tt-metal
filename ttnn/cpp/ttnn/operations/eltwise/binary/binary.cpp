
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary.hpp"

#include "device/binary_device_operation.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/binary_ng_device_operation.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::binary {

ttnn::Tensor typecast_to(ttnn::DataType dtype, const ttnn::Tensor& input) {
    return input.get_dtype() == dtype ? input : ttnn::typecast(input, dtype);
}

bool needs_typecast_to_bfloat16(const ttnn::DataType input) {
    return (input == ttnn::DataType::BFLOAT8_B || input == ttnn::DataType::BFLOAT4_B);
}
namespace detail {

constexpr bool is_associative(BinaryOpType op) {
    return op == BinaryOpType::ADD || op == BinaryOpType::MUL || op == BinaryOpType::EQ || op == BinaryOpType::NE ||
           op == BinaryOpType::LOGICAL_AND || op == BinaryOpType::LOGICAL_OR || op == BinaryOpType::LOGADDEXP ||
           op == BinaryOpType::LOGADDEXP2 || op == BinaryOpType::LOGICAL_XOR || op == BinaryOpType::MAXIMUM;
}

// Tensor - Scalar
inline Tensor binary_impl(
    QueueId queue_id,
    BinaryOpType binary_op_type,
    const ttnn::Tensor& lhs,
    const float rhs,
    const std::optional<const DataType>& dtype = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    auto output_tensor = lhs;
    if (binary_op_type == BinaryOpType::GT) {
        output_tensor = ttnn::gt_unary(queue_id, lhs, rhs, memory_config, output);
    } else if (binary_op_type == BinaryOpType::LT) {
        output_tensor = ttnn::lt_unary(queue_id, lhs, rhs, memory_config, output);
    } else if (binary_op_type == BinaryOpType::NE) {
        output_tensor = ttnn::ne_unary(queue_id, lhs, rhs, memory_config, output);
    } else if (binary_op_type == BinaryOpType::GTE) {
        output_tensor = ttnn::gez(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, memory_config), memory_config, output);
    } else if (binary_op_type == BinaryOpType::LTE) {
        output_tensor = ttnn::lez(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, memory_config), memory_config, output);
    } else if (binary_op_type == BinaryOpType::EQ) {
        output_tensor = ttnn::eqz(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, memory_config), memory_config, output);
    } else {
        TT_THROW("Unsupported operation");
    }
    if (dtype.has_value()) {
        output_tensor = ttnn::typecast(queue_id, output_tensor, *dtype, std::nullopt, output);
    }
    return output_tensor;
}

// Scalar - Tensor
inline Tensor binary_impl(
    QueueId queue_id,
    BinaryOpType binary_op_type,
    const float lhs,
    const ttnn::Tensor& rhs,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt) {
    if (binary_op_type == BinaryOpType::GTE) {
        return ttnn::gez(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, memory_config), memory_config, output);
    }
    if (binary_op_type == BinaryOpType::LTE) {
        return ttnn::lez(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, memory_config), memory_config, output);
    }
    if (binary_op_type == BinaryOpType::EQ) {
        return ttnn::eqz(queue_id, ttnn::sub_sfpu(queue_id, lhs, rhs, memory_config), memory_config, output);
    }

    TT_THROW("Unsupported operation");
}

inline auto preprocess_inputs(BinaryOpType binary_op_type, Tensor a, Tensor b) {
    // TODO: #7731 (Remove calls to repeat)
    constexpr auto repeat_smaller = [](const Tensor& first, Tensor& second) {
        const auto& first_shape = first.get_logical_shape();
        const auto& second_shape = second.get_logical_shape();
        // repeats second if it is smaller
        if (first_shape.rank() == 4 and second_shape.rank() == 4 and first_shape[0] > second_shape[0]) {
            TT_FATAL(second_shape[0] == 1, "Dimension trying to broadcast is not equal to 1");
            Shape repeats(std::array<uint32_t, 4>{first_shape[0], 1, 1, 1});
            second = ttnn::repeat(second, repeats);
        }
        // repeats second if it is smaller
        if (first_shape.rank() >= 3 and second_shape.rank() >= 3 and first_shape[-3] > second_shape[-3]) {
            TT_FATAL(second_shape[-3] == 1, "Dimension trying to broadcast is not equal to 1");
            int rank_a = first_shape.rank();
            std::vector<uint32_t> repeat_dim(rank_a, 1);
            repeat_dim[rank_a - 3] = first_shape[rank_a - 3];
            Shape repeats(repeat_dim);
            second = ttnn::repeat(second, repeats);
        }
    };

    repeat_smaller(a, b);
    repeat_smaller(b, a);

    // Swap tensors if a needs to be broadcasted to b
    if (detail::is_associative(binary_op_type) and a.get_logical_volume() < b.get_logical_volume()) {
        return std::make_tuple(b, a);
    }

    return std::make_tuple(a, b);
}

inline auto invoke_binary_ng(
    QueueId queue_id,
    const Tensor& lhs,
    const auto& rhs,
    BinaryOpType binary_op_type,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy) {
    const auto& output_mem_cfg = memory_config ? *memory_config : output ? output->memory_config() : MemoryConfig{};

    if (use_legacy.value_or(true)) {
        const std::vector activations(post_activations.begin(), post_activations.end());
        const std::optional lhs_activation =
            lhs_activations.empty() ? std::nullopt : std::optional{lhs_activations.front()};

        if constexpr (requires { detail::preprocess_inputs(binary_op_type, lhs, rhs); }) {
            auto [a, b] = detail::preprocess_inputs(binary_op_type, lhs, rhs);

            return ttnn::prim::binary(
                queue_id, a, b, binary_op_type, dtype, memory_config, output, activations, lhs_activation);
        } else {
            return ttnn::prim::binary(
                queue_id, lhs, rhs, binary_op_type, dtype, memory_config, output, activations, lhs_activation);
        }
    }

    const ttnn::DataType a_dtype = lhs.get_dtype();
    const bool output_preallocated = output.has_value();
    const ttnn::DataType out_dtype = output_preallocated ? output->get_dtype() : dtype.value_or(a_dtype);

    const auto mem_config = output_preallocated ? output->memory_config() : memory_config.value_or(lhs.memory_config());

    if (dtype.has_value() && output_preallocated) {
        TT_FATAL(*dtype == out_dtype, "If both output dtype and output tensor are provided, their dtypes should match");
    }

    bool typecast_a = binary::needs_typecast_to_bfloat16(a_dtype);
    bool typecast_b = [&] {
        if constexpr (requires { rhs.get_dtype(); }) {
            return binary::needs_typecast_to_bfloat16(rhs.get_dtype());
        } else {
            return false;
        }
    }();
    bool typecast_out = binary::needs_typecast_to_bfloat16(out_dtype);

    // RM is never BFLOAT8 or BFLOAT4 so we can assume it goes in here.
    if (!typecast_a && !typecast_b) {
        bool input_a_rm = lhs.get_layout() == Layout::ROW_MAJOR;
        bool input_b_rm = [&] {
            if constexpr (requires { rhs.get_layout(); }) {
                return rhs.get_layout() == Layout::ROW_MAJOR;
            } else {
                return true;
            }
        }();
        Tensor input_a =
            input_a_rm ? ttnn::to_layout(lhs, Layout::TILE, std::nullopt, std::nullopt, (IDevice*)nullptr) : lhs;
        auto input_b = [&] {
            if constexpr (requires { rhs.get_layout(); }) {
                return input_b_rm ? ttnn::to_layout(rhs, Layout::TILE, std::nullopt, std::nullopt, (IDevice*)nullptr)
                                  : rhs;
            } else {
                return rhs;
            }
        }();

        if (input_a_rm && input_b_rm) {
            // we don't support to_layout with optional output tensor
            TT_FATAL(
                !output_preallocated,
                "Optional output tensor with Row Major input is not supported right now for Elementwise operations");
        }

        Tensor result = ttnn::prim::binary_ng(
            queue_id,
            input_a,
            input_b,
            binary_op_type,
            out_dtype,
            mem_config,
            output,
            lhs_activations,
            rhs_activations,
            post_activations);

        // if both inputs are in row major, convert the output to row major
        // since there's no consensus here, avoiding the conversion if we have an excuse to is likely the best option
        // since it leads to better perf
        if (input_a_rm && input_b_rm) {
            result = ttnn::to_layout(result, Layout::ROW_MAJOR, std::nullopt, mem_config, (IDevice*)nullptr);
        }

        return result;
    } else {
        Tensor input_a = binary::typecast_to(DataType::BFLOAT16, lhs);
        auto input_b = [&] {
            if constexpr (requires { binary::typecast_to(DataType::BFLOAT16, rhs); }) {
                return binary::typecast_to(DataType::BFLOAT16, rhs);
            } else {
                return rhs;
            }
        }();
        const auto output_tensor =
            output_preallocated and typecast_out ? ttnn::typecast(*output, DataType::BFLOAT16) : output;

        Tensor result = ttnn::prim::binary_ng(
            queue_id,
            input_a,
            input_b,
            binary_op_type,
            input_a.get_dtype(),
            mem_config,
            output_tensor,
            lhs_activations,
            rhs_activations,
            post_activations);

        return typecast_out ? ttnn::typecast(result, out_dtype, mem_config, output) : result;
    }
}

}  // namespace detail

template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy) {
    return detail::invoke_binary_ng(
        queue_id,
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(
    QueueId queue_id,
    const ttnn::Tensor& lhs,
    float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy) {
    return detail::invoke_binary_ng(
        queue_id,
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy) {
    return detail::invoke_binary_ng(
        queue_id,
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(
    QueueId queue_id,
    const ttnn::Tensor& lhs,
    const float rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    const std::optional<bool>& use_legacy) {
    if (use_legacy.value_or(true)) {
        return detail::binary_impl(DefaultQueueId, binary_op_type, lhs, rhs, dtype, memory_config, output);
    }

    return detail::invoke_binary_ng(
        queue_id,
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}
// scalar - tensor combination not available on Pytorch for this op
template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(
    QueueId queue_id,
    const float lhs,
    const ttnn::Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    return detail::binary_impl(DefaultQueueId, binary_op_type, lhs, rhs, memory_config, output);
}

template <BinaryOpType binary_op_type>
Tensor InplaceRelationalBinary<binary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return RelationalBinary<binary_op_type>::invoke(
        queue_id,
        lhs,
        rhs,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template <BinaryOpType binary_op_type>
Tensor InplaceRelationalBinary<binary_op_type>::invoke(
    QueueId queue_id,
    const ttnn::Tensor& lhs,
    const float rhs,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return RelationalBinary<binary_op_type>::invoke(
        queue_id,
        lhs,
        rhs,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template <BinaryOpType binary_op_type>
Tensor InplaceLogicalBinary<binary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperation<binary_op_type>::invoke(
        queue_id,
        lhs,
        rhs,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryOperation<binary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperation<binary_op_type>::invoke(
        queue_id,
        lhs,
        rhs,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryOperation<binary_op_type>::invoke(
    QueueId queue_id,
    const ttnn::Tensor& lhs,
    const float rhs,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return BinaryOperation<binary_op_type>::invoke(
        queue_id,
        lhs,
        rhs,
        std::nullopt,
        std::nullopt,
        lhs,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template <BinaryOpType binary_op_type>
Tensor BinaryOperationSfpu<binary_op_type>::invoke(
    QueueId queue_id,
    const Tensor& lhs,
    const Tensor& rhs,
    const std::optional<const DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> post_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> lhs_activations,
    tt::stl::Span<const ttnn::operations::unary::UnaryWithParam> rhs_activations,
    std::optional<bool> use_legacy) {
    return detail::invoke_binary_ng(
        queue_id,
        lhs,
        rhs,
        binary_op_type,
        dtype,
        memory_config,
        output,
        post_activations,
        lhs_activations,
        rhs_activations,
        use_legacy);
}

template struct BinaryOperation<BinaryOpType::ADD>;
template struct InplaceBinaryOperation<BinaryOpType::ADD>;
template struct BinaryOperation<BinaryOpType::SUB>;
template struct InplaceBinaryOperation<BinaryOpType::SUB>;
template struct BinaryOperation<BinaryOpType::MUL>;
template struct InplaceBinaryOperation<BinaryOpType::MUL>;
template struct BinaryOperation<BinaryOpType::LOGICAL_AND>;
template struct BinaryOperation<BinaryOpType::LOGICAL_OR>;
template struct BinaryOperation<BinaryOpType::LOGICAL_XOR>;
template struct BinaryOperation<BinaryOpType::LDEXP>;
template struct InplaceBinaryOperation<BinaryOpType::LDEXP>;
template struct BinaryOperation<BinaryOpType::LOGADDEXP>;
template struct InplaceBinaryOperation<BinaryOpType::LOGADDEXP>;
template struct BinaryOperation<BinaryOpType::LOGADDEXP2>;
template struct InplaceBinaryOperation<BinaryOpType::LOGADDEXP2>;
template struct BinaryOperation<BinaryOpType::SQUARED_DIFFERENCE>;
template struct InplaceBinaryOperation<BinaryOpType::SQUARED_DIFFERENCE>;
template struct BinaryOperation<BinaryOpType::DIV>;
template struct InplaceBinaryOperation<BinaryOpType::DIV>;
template struct BinaryOperation<BinaryOpType::BIAS_GELU>;
template struct InplaceBinaryOperation<BinaryOpType::BIAS_GELU>;
template struct BinaryOperation<BinaryOpType::RSUB>;
template struct InplaceBinaryOperation<BinaryOpType::RSUB>;
template struct BinaryOperation<BinaryOpType::BITWISE_AND>;
template struct BinaryOperation<BinaryOpType::BITWISE_OR>;
template struct BinaryOperation<BinaryOpType::BITWISE_XOR>;
template struct BinaryOperation<BinaryOpType::LEFT_SHIFT>;
template struct BinaryOperation<BinaryOpType::RIGHT_SHIFT>;

template struct RelationalBinary<BinaryOpType::EQ>;
template struct RelationalBinary<BinaryOpType::NE>;
template struct RelationalBinary<BinaryOpType::GTE>;
template struct RelationalBinary<BinaryOpType::GT>;
template struct RelationalBinary<BinaryOpType::LTE>;
template struct RelationalBinary<BinaryOpType::LT>;

template struct InplaceRelationalBinary<BinaryOpType::GT>;
template struct InplaceRelationalBinary<BinaryOpType::LT>;
template struct InplaceRelationalBinary<BinaryOpType::GTE>;
template struct InplaceRelationalBinary<BinaryOpType::LTE>;
template struct InplaceRelationalBinary<BinaryOpType::EQ>;
template struct InplaceRelationalBinary<BinaryOpType::NE>;

template struct InplaceLogicalBinary<BinaryOpType::LOGICAL_AND>;
template struct InplaceLogicalBinary<BinaryOpType::LOGICAL_OR>;
template struct InplaceLogicalBinary<BinaryOpType::LOGICAL_XOR>;

template struct BinaryOperationSfpu<BinaryOpType::POWER>;
template struct BinaryOperationSfpu<BinaryOpType::BITWISE_AND>;
template struct BinaryOperationSfpu<BinaryOpType::BITWISE_XOR>;
template struct BinaryOperationSfpu<BinaryOpType::BITWISE_OR>;
template struct BinaryOperationSfpu<BinaryOpType::LEFT_SHIFT>;
template struct BinaryOperationSfpu<BinaryOpType::RIGHT_SHIFT>;
template struct BinaryOperationSfpu<BinaryOpType::MAXIMUM>;

}  // namespace ttnn::operations::binary
