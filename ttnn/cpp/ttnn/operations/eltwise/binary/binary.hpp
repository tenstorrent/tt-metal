
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/binary_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::binary {

namespace utils {

constexpr bool is_associative(BinaryOpType op) {
    return op == BinaryOpType::ADD ||
           op == BinaryOpType::MUL ||
           op == BinaryOpType::EQ ||
           op == BinaryOpType::NE ||
           op == BinaryOpType::LOGICAL_AND ||
           op == BinaryOpType::LOGICAL_OR ||
           op == BinaryOpType::LOGADDEXP ||
           op == BinaryOpType::LOGADDEXP2;
}
}

template <BinaryOpType binary_op_type, bool in_place>
struct Binary {

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<FusedActivations> activations = std::nullopt) {

        if(output_dtype.has_value() && optional_output_tensor.has_value()){
            TT_FATAL(output_dtype.value() == optional_output_tensor.value().get_dtype(), "If both output dtype and output tensor provided dtype should match");
        }

        auto &&[input_tensor_a, input_tensor_b] = [](const auto &input_tensor_a_arg, const auto &input_tensor_b_arg) {
            if(utils::is_associative(binary_op_type)) {
                const auto input_shape_a = input_tensor_a_arg.get_shape();
                const auto input_shape_b = input_tensor_b_arg.get_shape();
                // Swap tensors if input_tensor_a needs to be broadcasted to input_tensor_b
                if (tt::tt_metal::compute_volume(input_shape_a) < tt::tt_metal::compute_volume(input_shape_b)) {
                    return std::make_tuple(input_tensor_b_arg, input_tensor_a_arg);
                }
            }
            return std::make_tuple(input_tensor_a_arg, input_tensor_b_arg);
        }(input_tensor_a_arg, input_tensor_b_arg);

        auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());

        // TODO(arakhmati): #7731 - remove this!
        auto repeat_smaller = [&output_memory_config](const auto& first, auto& second){
            const auto first_shape = first.get_shape();
            const auto second_shape = second.get_shape();

            // repeats second if it is smaller
            if (first_shape.rank() == 4 and second_shape.rank() == 4 and
                first_shape[0] > second_shape[0] and
                first_shape[-1] == second_shape[-1] and
                first_shape[-2] == second_shape[-2] and
                first_shape[-3] == second_shape[-3]) {

                tt::log_warning(tt::LogOp, "Using repeat op to broadcast batch dim");
                Shape repeats({first_shape[0], 1, 1, 1});
                second = ttnn::repeat(second, repeats, output_memory_config);
            }
        };
        repeat_smaller(input_tensor_a, input_tensor_b);
        repeat_smaller(input_tensor_b, input_tensor_a);

        DataType dtype = output_dtype.value_or(input_tensor_a.get_dtype());
        if(optional_output_tensor.has_value()) {
            dtype = optional_output_tensor.value().get_dtype();
        }

        return ttnn::device_operation::run<BinaryDeviceOperation>(
            queue_id,
            BinaryDeviceOperation::operation_attributes_t{
                binary_op_type, in_place, activations, output_memory_config, dtype, std::nullopt},
            BinaryDeviceOperation::tensor_args_t{input_tensor_a, input_tensor_b, optional_output_tensor});
    }

    static Tensor execute_on_worker_thread(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<FusedActivations> activations = std::nullopt)
    {
        return execute_on_worker_thread(DefaultQueueId, input_tensor_a_arg, input_tensor_b_arg, output_dtype, memory_config, optional_output_tensor, activations);
    }

    // TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
    // Currently, this is exactly how tt::tt_metal::add_unary works
    static Tensor execute_on_worker_thread(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<FusedActivations> activations = std::nullopt) {
        return Binary::execute_on_worker_thread(
            DefaultQueueId,
            input_tensor_a,
            scalar,
            dtype,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            optional_output_tensor,
            activations);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<FusedActivations> activations = std::nullopt) {
        // Cast Float Scalar to a device tensor
        auto host_buffer = owned_buffer::create<::bfloat16>(static_cast<std::size_t>(TILE_HEIGHT * TILE_WIDTH));
        host_buffer[0] = scalar;
        Tensor scalar_tensor_host = Tensor(
            OwnedStorage{host_buffer},
            ttnn::Shape(std::array<std::uint32_t, 2>{1, 1}, std::array<std::uint32_t, 2>{TILE_HEIGHT, TILE_WIDTH}),
            DataType::BFLOAT16,
            Layout::TILE);
        Tensor scalar_tensor_device = scalar_tensor_host.to(input_tensor_a.device());
        // TODO(arakhmati): #7637 pass in memory_config instead of operation::DEFAULT_OUTPUT_MEMORY_CONFIG
        return Binary::execute_on_worker_thread(
            input_tensor_a,
            scalar_tensor_device,
            dtype,
            operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
            optional_output_tensor,
            activations);
    }
};

}  // operations::binary

constexpr auto add =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::ADD, false>>("ttnn::add");
constexpr auto add_ =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::ADD, true>>("ttnn::add_");
constexpr auto subtract =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::SUB, false>>(
        "ttnn::subtract");
constexpr auto subtract_ =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::SUB, true>>(
        "ttnn::subtract_");
constexpr auto multiply =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::MUL, false>>(
        "ttnn::multiply");
constexpr auto multiply_ =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::MUL, true>>(
        "ttnn::multiply_");

constexpr auto eq =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::EQ, false>>("ttnn::eq");
constexpr auto ne =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::NE, false>>("ttnn::ne");
constexpr auto ge =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::GTE, false>>("ttnn::ge");
constexpr auto gt =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::GT, false>>("ttnn::gt");
constexpr auto le =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::LTE, false>>("ttnn::le");
constexpr auto lt =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::LT, false>>("ttnn::lt");
constexpr auto logical_and =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::LOGICAL_AND, false>>(
        "ttnn::logical_and");
constexpr auto logical_or =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::LOGICAL_OR, false>>(
        "ttnn::logical_or");
constexpr auto ldexp =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::LDEXP, false>>("ttnn::ldexp");

constexpr auto logaddexp =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::LOGADDEXP, false>>(
        "ttnn::logaddexp");
constexpr auto logaddexp2 =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::LOGADDEXP2, false>>(
        "ttnn::logaddexp2");
constexpr auto squared_difference =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::SQUARED_DIFFERENCE, false>>(
        "ttnn::squared_difference");
constexpr auto divide =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::DIV_FAST, false>>(
        "ttnn::divide");
constexpr auto bias_gelu =
    ttnn::register_operation<operations::binary::Binary<operations::binary::BinaryOpType::BIAS_GELU, false>>(
        "ttnn::bias_gelu");

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
