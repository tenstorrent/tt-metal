// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_eager/tensor/owned_buffer_functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_eager/tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace binary {

using BinaryOpType = tt::tt_metal::BinaryOpType;

struct BinaryProgramConfig {
    const std::optional<std::vector<std::string>> activations;
    const MemoryConfig memory_config;
    const DataType dtype;

    static constexpr auto attribute_names = std::make_tuple("activations", "memory_config", "dtype");
    const auto attribute_values() const {
        return std::make_tuple(
            std::cref(this->activations), std::cref(this->memory_config), std::cref(this->dtype));
    }
};

template <BinaryOpType binary_op_type, bool in_place>
struct Binary {
    const BinaryProgramConfig program_config;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;

    operation::OpPerformanceModel create_op_performance_model(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("program_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->program_config), std::cref(this->compute_kernel_config));
    }

    static inline const std::array<TensorSchema, 2> input_tensor_schemas() {
        return {
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                false,
                false},
            ttnn::TensorSchema{
                2,
                4,
                {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b},
                {ttnn::TILE_LAYOUT},
                true,
                false,
                true,
                false}};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &input_tensor_a, const Tensor &input_tensor_b, Args &&...args) {
        return std::make_tuple(input_tensor_a, input_tensor_b);
    }

    static Tensor execute(
        const Tensor &input_tensor_a,
        const Tensor &input_tensor_b,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::optional<const DataType> &dtype = std::nullopt,
        std::optional<std::vector<std::string>> activations = std::nullopt) {
        std::vector<Tensor> output_tensors = {
            Tensor(operation::get_workers_for_op_output({input_tensor_a, input_tensor_b}))};
        operation::launch_op(
            [activations, memory_config, dtype](
                const std::vector<Tensor> &input_tensors,
                const std::vector<std::optional<const Tensor>> &optional_input_tensors,
                const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                auto &&[input_tensor_a, input_tensor_b] = [](const auto &input_tensor_a_arg,
                                                             const auto &input_tensor_b_arg) {
                    // Swap tensors if input_tensor_a needs to be broadcasted to input_tensor_b
                    if (tt::tt_metal::compute_volume(input_tensor_a_arg.get_shape()) <
                        tt::tt_metal::compute_volume(input_tensor_b_arg.get_shape())) {
                        return std::make_tuple(input_tensor_b_arg, input_tensor_a_arg);
                    }
                    return std::make_tuple(input_tensor_a_arg, input_tensor_b_arg);
                }(input_tensors.at(0), input_tensors.at(1));

                auto output_memory_config = memory_config.value_or(input_tensor_a.memory_config());

                // TODO(arakhmati): #7731 - remove this!
                auto input_shape_a = input_tensor_a.get_shape();
                auto input_shape_b = input_tensor_b.get_shape();
                if (input_shape_a.rank() == 4 and input_shape_b.rank() == 4 and input_shape_a[0] > input_shape_b[0] and
                    input_shape_a[-1] == input_shape_b[-1] and input_shape_a[-2] == input_shape_b[-2] and
                    input_shape_a[-3] == input_shape_b[-3]) {
                    tt::log_warning(tt::LogOp, "Using repeat op to broadcast batch dim");
                    Shape repeats({input_shape_a[0], 1, 1, 1});
                    input_tensor_b = repeat(input_tensor_b, repeats.value(), output_memory_config);
                }

                return operation::run(
                    Binary{BinaryProgramConfig{
                        activations, output_memory_config, dtype.value_or(input_tensor_a.get_dtype())}},
                    {input_tensor_a, input_tensor_b});
            },
            {input_tensor_a, input_tensor_b},
            output_tensors);
        return output_tensors.at(0);
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const Tensor &input_tensor_a, const float input_tensor_b, Args &&...args) {
        return std::make_tuple(input_tensor_a, input_tensor_b);
    }

    // TODO: this case should use BinaryWithScalarProgramConfig and there should be a custom kernel to run this
    // Currently, this is exactly how tt::tt_metal::add_unary works
    static Tensor execute(
        const ttnn::Tensor &input_tensor_a,
        const float scalar,
        const std::optional<ttnn::MemoryConfig> &memory_config = std::nullopt,
        const std::optional<const DataType> &dtype = std::nullopt,
        std::optional<std::vector<std::string>> activations = std::nullopt) {
        // Cast Float Scalar to a device tensor
        auto host_buffer = owned_buffer::create<::bfloat16>(static_cast<std::size_t>(TILE_HEIGHT * TILE_WIDTH));
        host_buffer[0] = scalar;
        Tensor scalar_tensor_host = Tensor(
            OwnedStorage{host_buffer},
            ttnn::Shape(std::array<std::uint32_t, 2>{1, 1}, std::array<std::uint32_t, 2>{TILE_HEIGHT, TILE_WIDTH}),
            DataType::BFLOAT16,
            Layout::TILE);
        Tensor scalar_tensor_device = scalar_tensor_host.to(input_tensor_a.get_workers());
        // TODO(arakhmati): #7637 pass in memory_config instead of operation::DEFAULT_OUTPUT_MEMORY_CONFIG
        return Binary::execute(
            input_tensor_a, scalar_tensor_device, operation::DEFAULT_OUTPUT_MEMORY_CONFIG, dtype, activations);
    }
};

}  // namespace binary

}  // namespace operations

}  // namespace ttnn
