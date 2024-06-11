
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/binary_op.hpp"

#include "ttnn/operations/data_movement.hpp"

namespace ttnn {

namespace operations::binary {

template <BinaryOpType binary_op_type, bool in_place>
struct ExecuteBinary {
    static inline const std::array<TensorSchema, 2> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::uint16},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                true,
                false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(uint8_t queue_id, const Tensor &input_tensor_a, const Tensor &input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(input_tensor_a, input_tensor_b);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<std::vector<std::string>> activations = std::nullopt) {

        if(output_dtype.has_value() && optional_output_tensor.has_value()){
            TT_FATAL(output_dtype.value() == optional_output_tensor.value().get_dtype(), "If both output dtype and output tensor provided dtype should match");
        }

        auto &&[input_tensor_a, input_tensor_b] = [](const auto &input_tensor_a_arg, const auto &input_tensor_b_arg) {
            const auto input_shape_a = input_tensor_a_arg.get_shape();
            const auto input_shape_b = input_tensor_b_arg.get_shape();
            // Swap tensors if input_tensor_a needs to be broadcasted to input_tensor_b
            if (tt::tt_metal::compute_volume(input_shape_a) < tt::tt_metal::compute_volume(input_shape_b)) {
                return std::make_tuple(input_tensor_b_arg, input_tensor_a_arg);
            }
            return std::make_tuple(input_tensor_a_arg, input_tensor_b_arg);
        }(input_tensor_a_arg, input_tensor_b_arg);

        auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());

        // TODO(arakhmati): #7731 - remove this!
        auto input_shape_a = input_tensor_a.get_shape();
        auto input_shape_b = input_tensor_b.get_shape();
        if (input_shape_a.rank() == 4 and input_shape_b.rank() == 4 and input_shape_a[0] > input_shape_b[0] and
            input_shape_a[-1] == input_shape_b[-1] and input_shape_a[-2] == input_shape_b[-2] and
            input_shape_a[-3] == input_shape_b[-3]) {
            tt::log_warning(tt::LogOp, "Using repeat op to broadcast batch dim");
            Shape repeats({input_shape_a[0], 1, 1, 1});
            input_tensor_b = ttnn::repeat(input_tensor_b, repeats, output_memory_config);
        }

        DataType dtype = output_dtype.value_or(input_tensor_a.get_dtype());
        if(optional_output_tensor.has_value()) {
            dtype = optional_output_tensor.value().get_dtype();
        }

        auto output_tensors = operation::run(Binary{BinaryProgramConfig{binary_op_type,
                                                                        in_place,
                                                                        activations,
                                                                        output_memory_config,
                                                                        dtype}},
                                                    {input_tensor_a, input_tensor_b},
                                                    {},
                                                    {optional_output_tensor},
                                                    queue_id);

        return output_tensors.at(0);
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(input_tensor_a, input_tensor_b);
    }

    static Tensor execute_on_worker_thread(
        const Tensor &input_tensor_a_arg,
        const Tensor &input_tensor_b_arg,
        const std::optional<const DataType> &output_dtype = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt,
        std::optional<std::vector<std::string>> activations = std::nullopt)
    {
        return execute_on_worker_thread(DefaultQueueId, input_tensor_a_arg, input_tensor_b_arg, output_dtype, memory_config, optional_output_tensor, activations);
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &input_tensor_a, const float input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(input_tensor_a, input_tensor_b);
    }

    // TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
    // Currently, this is exactly how tt::tt_metal::add_unary works
    static Tensor execute_on_worker_thread(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<std::vector<std::string>> activations = std::nullopt) {

        return ExecuteBinary::execute_on_worker_thread(DefaultQueueId, input_tensor_a, scalar, dtype, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, optional_output_tensor, activations);
    }

    template <typename... Args>
    static auto input_tensors_to_validate(uint8_t queue_id, const Tensor &input_tensor_a, const float input_tensor_b, Args &&...args) {
        return std::forward_as_tuple(input_tensor_a, input_tensor_b);
    }

    static Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<const DataType> &dtype = std::nullopt,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<Tensor> &optional_output_tensor = std::nullopt,
        std::optional<std::vector<std::string>> activations = std::nullopt) {
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
        return ExecuteBinary::execute_on_worker_thread(
            input_tensor_a, scalar_tensor_device, dtype, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, optional_output_tensor, activations);
    }
};

}  // operations::binary

template <ttnn::operations::binary::BinaryOpType OpType, bool InPlace>
constexpr auto register_op(const char* name) {
    return ttnn::register_operation<ttnn::operations::binary::ExecuteBinary<OpType, InPlace>>(name);
}

constexpr auto add = register_op<ttnn::operations::binary::BinaryOpType::ADD, false>("ttnn::add");
constexpr auto add_ = register_op<ttnn::operations::binary::BinaryOpType::ADD, true>("ttnn::add_");
constexpr auto subtract = register_op<ttnn::operations::binary::BinaryOpType::SUB, false>("ttnn::subtract");
constexpr auto subtract_ = register_op<ttnn::operations::binary::BinaryOpType::SUB, true>("ttnn::subtract_");
constexpr auto multiply = register_op<ttnn::operations::binary::BinaryOpType::MUL, false>("ttnn::multiply");
constexpr auto multiply_ = register_op<ttnn::operations::binary::BinaryOpType::MUL, true>("ttnn::multiply_");

constexpr auto eq = register_op<ttnn::operations::binary::BinaryOpType::EQ, false>("ttnn::eq");
constexpr auto ne = register_op<ttnn::operations::binary::BinaryOpType::NE, false>("ttnn::ne");
constexpr auto ge = register_op<ttnn::operations::binary::BinaryOpType::GTE, false>("ttnn::ge");
constexpr auto gt = register_op<ttnn::operations::binary::BinaryOpType::GT, false>("ttnn::gt");
constexpr auto le = register_op<ttnn::operations::binary::BinaryOpType::LTE, false>("ttnn::le");
constexpr auto lt = register_op<ttnn::operations::binary::BinaryOpType::LT, false>("ttnn::lt");
constexpr auto logical_and = register_op<ttnn::operations::binary::BinaryOpType::LOGICAL_AND, false>("ttnn::logical_and");
constexpr auto logical_or = register_op<ttnn::operations::binary::BinaryOpType::LOGICAL_OR, false>("ttnn::logical_or");
constexpr auto ldexp = register_op<ttnn::operations::binary::BinaryOpType::LDEXP, false>("ttnn::ldexp");

constexpr auto logaddexp = register_op<ttnn::operations::binary::BinaryOpType::LOGADDEXP, false>("ttnn::logaddexp");
constexpr auto logaddexp2 = register_op<ttnn::operations::binary::BinaryOpType::LOGADDEXP2, false>("ttnn::logaddexp2");
constexpr auto squared_difference = register_op<ttnn::operations::binary::BinaryOpType::SQUARED_DIFFERENCE, false>("ttnn::squared_difference");
constexpr auto div_fast = register_op<ttnn::operations::binary::BinaryOpType::DIV_FAST, false>("ttnn::div_fast");


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
