
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "binary.hpp"

#include "device/binary_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace ttnn::operations::binary {

namespace detail {

constexpr bool is_associative(BinaryOpType op) {
    return op == BinaryOpType::ADD || op == BinaryOpType::MUL || op == BinaryOpType::EQ || op == BinaryOpType::NE ||
           op == BinaryOpType::LOGICAL_AND || op == BinaryOpType::LOGICAL_OR || op == BinaryOpType::LOGADDEXP ||
           op == BinaryOpType::LOGADDEXP2;
}

// Tensor - Scalar
inline Tensor binary_impl(uint8_t queue_id,
                          BinaryOpType binary_op_type,
                          const ttnn::Tensor &input_tensor,
                          const float scalar,
                          const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
                          const std::optional<Tensor> &optional_output_tensor = std::nullopt) {
    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    auto output_tensor = input_tensor;
    if (binary_op_type == BinaryOpType::GT) {
        output_tensor = ttnn::gt_unary(queue_id, input_tensor, scalar, output_memory_config, optional_output_tensor);
    } else if (binary_op_type == BinaryOpType::LT) {
        output_tensor = ttnn::lt_unary(queue_id, input_tensor, scalar, output_memory_config, optional_output_tensor);
    } else if (binary_op_type == BinaryOpType::NE) {
        output_tensor = ttnn::ne_unary(queue_id, input_tensor, scalar, output_memory_config, optional_output_tensor);
    } else if (binary_op_type == BinaryOpType::GTE) {
        output_tensor = ttnn::gez(queue_id,
                                  ttnn::sub_sfpu(queue_id, input_tensor, scalar, output_memory_config),
                                  output_memory_config,
                                  optional_output_tensor);
    } else if (binary_op_type == BinaryOpType::LTE) {
        output_tensor = ttnn::lez(queue_id,
                                  ttnn::sub_sfpu(queue_id, input_tensor, scalar, output_memory_config),
                                  output_memory_config,
                                  optional_output_tensor);
    } else if (binary_op_type == BinaryOpType::EQ) {
        output_tensor = ttnn::eqz(queue_id,
                                  ttnn::sub_sfpu(queue_id, input_tensor, scalar, output_memory_config),
                                  output_memory_config,
                                  optional_output_tensor);
    } else {
        TT_THROW("Unsupported operation");
    }
    return output_tensor;
}

// Scalar - Tensor
inline Tensor binary_impl(uint8_t queue_id,
                          BinaryOpType binary_op_type,
                          const float scalar,
                          const ttnn::Tensor &input_tensor,
                          const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
                          const std::optional<Tensor> &optional_output_tensor = std::nullopt) {
    auto output_memory_config = optional_output_tensor.has_value()
                                    ? optional_output_tensor.value().memory_config()
                                    : memory_config.value_or(input_tensor.memory_config());
    auto output_tensor = input_tensor;
    if (binary_op_type == BinaryOpType::GTE) {
        output_tensor = ttnn::gez(queue_id,
                                  ttnn::sub_sfpu(queue_id, scalar, input_tensor, output_memory_config),
                                  output_memory_config,
                                  optional_output_tensor);
    } else if (binary_op_type == BinaryOpType::LTE) {
        output_tensor = ttnn::lez(queue_id,
                                  ttnn::sub_sfpu(queue_id, scalar, input_tensor, output_memory_config),
                                  output_memory_config,
                                  optional_output_tensor);
    } else if (binary_op_type == BinaryOpType::EQ) {
        output_tensor = ttnn::eqz(queue_id,
                                  ttnn::sub_sfpu(queue_id, scalar, input_tensor, output_memory_config),
                                  output_memory_config,
                                  optional_output_tensor);
    } else {
        TT_THROW("Unsupported operation");
    }
    return output_tensor;
}

template <BinaryOpType binary_op_type>
auto preprocess_inputs(const Tensor &input_tensor_a_arg, const Tensor &input_tensor_b_arg) {
    Tensor input_tensor_a = input_tensor_a_arg;
    Tensor input_tensor_b = input_tensor_b_arg;

    // TODO: #7731 (Remove calls to repeat )
    auto repeat_smaller = [](const auto &first, auto &second) {
        const auto first_shape = first.get_shape();
        const auto second_shape = second.get_shape();

        // repeats second if it is smaller
        if (first_shape.rank() == 4 and second_shape.rank() == 4 and first_shape[0] > second_shape[0]) {
            tt::log_warning(tt::LogOp, "Using repeat op to broadcast batch dim");
            Shape repeats(std::array<uint32_t, 4>{first_shape[0], 1, 1, 1});
            second = ttnn::repeat(second, repeats);
        }
    };
    repeat_smaller(input_tensor_a, input_tensor_b);
    repeat_smaller(input_tensor_b, input_tensor_a);

    return [](const auto &input_tensor_a, const auto &input_tensor_b) {
        if constexpr (detail::is_associative(binary_op_type)) {
            // Swap tensors if input_tensor_a needs to be broadcasted to input_tensor_b
            if (input_tensor_a.get_logical_volume() < input_tensor_b.get_logical_volume()) {
                return std::make_tuple(input_tensor_b, input_tensor_a);
            }
        }
        return std::make_tuple(input_tensor_a, input_tensor_b);
    }(input_tensor_a, input_tensor_b);
}

}  // namespace detail

template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(uint8_t queue_id,
                                               const Tensor &input_tensor_a_arg,
                                               const Tensor &input_tensor_b_arg,
                                               const std::optional<const DataType> &output_dtype,
                                               const std::optional<MemoryConfig> &memory_config,
                                               std::optional<Tensor> optional_output_tensor,
                                               std::optional<unary::FusedActivations> activations,
                                               std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    auto [input_tensor_a, input_tensor_b] =
        detail::preprocess_inputs<binary_op_type>(input_tensor_a_arg, input_tensor_b_arg);

    return ttnn::prim::binary(queue_id,
                              input_tensor_a,
                              input_tensor_b,
                              binary_op_type,
                              output_dtype,
                              memory_config,
                              optional_output_tensor,
                              activations,
                              input_tensor_a_activation);
}

template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(const Tensor &input_tensor_a_arg,
                                               const Tensor &input_tensor_b_arg,
                                               const std::optional<const DataType> &output_dtype,
                                               const std::optional<MemoryConfig> &memory_config,
                                               std::optional<Tensor> optional_output_tensor,
                                               std::optional<unary::FusedActivations> activations,
                                               std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    return invoke(DefaultQueueId,
                  input_tensor_a_arg,
                  input_tensor_b_arg,
                  output_dtype,
                  memory_config,
                  optional_output_tensor,
                  activations,
                  input_tensor_a_activation);
}

// TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
// Currently, this is exactly how tt::tt_metal::add_unary works
template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(const ttnn::Tensor &input_tensor_a,
                                               const float scalar,
                                               const std::optional<const DataType> &dtype,
                                               const std::optional<ttnn::MemoryConfig> &memory_config,
                                               const std::optional<Tensor> &optional_output_tensor,
                                               std::optional<unary::FusedActivations> activations,
                                               std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    return BinaryOperation::invoke(DefaultQueueId,
                                   input_tensor_a,
                                   scalar,
                                   dtype,
                                   memory_config,
                                   optional_output_tensor,
                                   activations,
                                   input_tensor_a_activation);
}

template <BinaryOpType binary_op_type>
Tensor BinaryOperation<binary_op_type>::invoke(uint8_t queue_id,
                                               const ttnn::Tensor &input_tensor_a,
                                               const float scalar,
                                               const std::optional<const DataType> &dtype,
                                               const std::optional<ttnn::MemoryConfig> &memory_config,
                                               const std::optional<Tensor> &optional_output_tensor,
                                               std::optional<unary::FusedActivations> activations,
                                               std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    using namespace tt::constants;
    // Cast Float Scalar to a device tensor
    auto host_buffer = owned_buffer::create<::bfloat16>(static_cast<std::size_t>(TILE_HEIGHT * TILE_WIDTH));
    host_buffer[0] = scalar;
    Tensor scalar_tensor_host =
        Tensor(OwnedStorage{host_buffer},
               ttnn::Shape(std::array<std::uint32_t, 2>{1, 1}, std::array<std::uint32_t, 2>{TILE_HEIGHT, TILE_WIDTH}),
               DataType::BFLOAT16,
               Layout::TILE);
    Tensor scalar_tensor_device = scalar_tensor_host.to(input_tensor_a.device());
    // TODO(arakhmati): #7637 pass in memory_config instead of operation::DEFAULT_OUTPUT_MEMORY_CONFIG
    return BinaryOperation::invoke(input_tensor_a,
                                   scalar_tensor_device,
                                   dtype,
                                   memory_config,
                                   optional_output_tensor,
                                   activations,
                                   input_tensor_a_activation);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(uint8_t queue_id,
                                                const Tensor &input_tensor_a_arg,
                                                const Tensor &input_tensor_b_arg,
                                                const std::optional<const DataType> &output_dtype,
                                                const std::optional<MemoryConfig> &memory_config,
                                                std::optional<Tensor> optional_output_tensor,
                                                std::optional<unary::FusedActivations> activations,
                                                std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    if (output_dtype.has_value() && optional_output_tensor.has_value()) {
        TT_FATAL(output_dtype.value() == optional_output_tensor.value().get_dtype(),
                 "If both output dtype and output tensor provided dtype should match");
    }

    auto [input_tensor_a, input_tensor_b] =
        detail::preprocess_inputs<binary_op_type>(input_tensor_a_arg, input_tensor_b_arg);

    auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());
    DataType dtype = output_dtype.value_or(input_tensor_a.get_dtype());
    if (optional_output_tensor.has_value()) {
        dtype = optional_output_tensor.value().get_dtype();
    }

    return ttnn::prim::binary(queue_id,
                              input_tensor_a,
                              input_tensor_b,
                              binary_op_type,
                              dtype,
                              output_memory_config,
                              optional_output_tensor,
                              activations,
                              input_tensor_a_activation);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(const Tensor &input_tensor_a_arg,
                                                const Tensor &input_tensor_b_arg,
                                                const std::optional<const DataType> &output_dtype,
                                                const std::optional<MemoryConfig> &memory_config,
                                                std::optional<Tensor> optional_output_tensor,
                                                std::optional<unary::FusedActivations> activations,
                                                std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    return invoke(DefaultQueueId,
                  input_tensor_a_arg,
                  input_tensor_b_arg,
                  output_dtype,
                  memory_config,
                  optional_output_tensor,
                  activations,
                  input_tensor_a_activation);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(const ttnn::Tensor &input_tensor_a,
                                                const float scalar,
                                                const std::optional<const DataType> &dtype,
                                                const std::optional<ttnn::MemoryConfig> &memory_config,
                                                const std::optional<Tensor> &optional_output_tensor,
                                                std::optional<unary::FusedActivations> activations,
                                                std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    return detail::binary_impl(
        DefaultQueueId, binary_op_type, input_tensor_a, scalar, memory_config, optional_output_tensor);
}

template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(uint8_t queue_id,
                                                const ttnn::Tensor &input_tensor_a,
                                                const float scalar,
                                                const std::optional<const DataType> &dtype,
                                                const std::optional<ttnn::MemoryConfig> &memory_config,
                                                const std::optional<Tensor> &optional_output_tensor,
                                                std::optional<unary::FusedActivations> activations,
                                                std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    return detail::binary_impl(
        DefaultQueueId, binary_op_type, input_tensor_a, scalar, memory_config, optional_output_tensor);
}
// scalar - tensor combination not available on Pytorch for this op
template <BinaryOpType binary_op_type>
Tensor RelationalBinary<binary_op_type>::invoke(uint8_t queue_id,
                                                const float scalar,
                                                const ttnn::Tensor &input_tensor_a,
                                                const std::optional<const DataType> &dtype,
                                                const std::optional<ttnn::MemoryConfig> &memory_config,
                                                const std::optional<Tensor> &optional_output_tensor) {
    return detail::binary_impl(
        DefaultQueueId, binary_op_type, scalar, input_tensor_a, memory_config, optional_output_tensor);
}

template <BinaryOpType binary_op_type>
Tensor InplaceRelationalBinary<binary_op_type>::invoke(const Tensor &input_tensor_a_arg,
                                                       const Tensor &input_tensor_b_arg) {
    return RelationalBinary<binary_op_type>::invoke(input_tensor_a_arg,
                                                    input_tensor_b_arg,
                                                    std::nullopt,
                                                    std::nullopt,
                                                    input_tensor_a_arg,
                                                    std::nullopt,
                                                    std::nullopt);
}

template <BinaryOpType binary_op_type>
Tensor InplaceRelationalBinary<binary_op_type>::invoke(const ttnn::Tensor &input_tensor_a, const float scalar) {
    return RelationalBinary<binary_op_type>::invoke(
        input_tensor_a, scalar, std::nullopt, std::nullopt, input_tensor_a, std::nullopt, std::nullopt);
}

template <BinaryOpType binary_op_type>
Tensor InplaceLogicalBinary<binary_op_type>::invoke(const Tensor &input_tensor_a_arg,
                                                    const Tensor &input_tensor_b_arg) {
    return BinaryOperation<binary_op_type>::invoke(input_tensor_a_arg,
                                                   input_tensor_b_arg,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   input_tensor_a_arg,
                                                   std::nullopt,
                                                   std::nullopt);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryOperation<binary_op_type>::invoke(const Tensor &input_tensor_a_arg,
                                                      const Tensor &input_tensor_b_arg,
                                                      std::optional<unary::FusedActivations> activations,
                                                      std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    return BinaryOperation<binary_op_type>::invoke(input_tensor_a_arg,
                                                   input_tensor_b_arg,
                                                   std::nullopt,
                                                   std::nullopt,
                                                   input_tensor_a_arg,
                                                   activations,
                                                   input_tensor_a_activation);
}

template <BinaryOpType binary_op_type>
Tensor InplaceBinaryOperation<binary_op_type>::invoke(const ttnn::Tensor &input_tensor_a,
                                                      const float scalar,
                                                      std::optional<unary::FusedActivations> activations,
                                                      std::optional<unary::UnaryWithParam> input_tensor_a_activation) {
    return BinaryOperation<binary_op_type>::invoke(
        input_tensor_a, scalar, std::nullopt, std::nullopt, input_tensor_a, activations, input_tensor_a_activation);
}

template struct BinaryOperation<BinaryOpType::ADD>;
template struct InplaceBinaryOperation<BinaryOpType::ADD>;
template struct BinaryOperation<BinaryOpType::SUB>;
template struct InplaceBinaryOperation<BinaryOpType::SUB>;
template struct BinaryOperation<BinaryOpType::MUL>;
template struct InplaceBinaryOperation<BinaryOpType::MUL>;
template struct BinaryOperation<BinaryOpType::LOGICAL_AND>;
template struct BinaryOperation<BinaryOpType::LOGICAL_OR>;
template struct BinaryOperation<BinaryOpType::LDEXP>;
template struct BinaryOperation<BinaryOpType::LOGADDEXP>;
template struct BinaryOperation<BinaryOpType::LOGADDEXP2>;
template struct BinaryOperation<BinaryOpType::SQUARED_DIFFERENCE>;
template struct BinaryOperation<BinaryOpType::DIV_FAST>;
template struct BinaryOperation<BinaryOpType::BIAS_GELU>;

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

}  // namespace ttnn::operations::binary
