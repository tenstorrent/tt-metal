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
#include "tt_eager/tt_dnn/op_library/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "ttnn/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations {

namespace core {

struct ToLayoutProgramConfig {
    const Layout layout;
    const MemoryConfig memory_config;
    const DataType dtype;

    static constexpr auto attribute_names = std::make_tuple("layout", "memory_config", "dtype");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->layout), std::cref(this->memory_config), std::cref(this->dtype));
    }
};

struct ToLayout {
    static inline const std::vector<TensorSchema> input_schemas{ttnn::TensorSchema{
        1,
        4,
        {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::bfloat4_b, ttnn::float32, ttnn::uint16, ttnn::uint32, ttnn::int32},
        {ttnn::ROW_MAJOR_LAYOUT, ttnn::TILE_LAYOUT},
        true,
        false,
        false}};

    const ToLayoutProgramConfig program_config;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names = std::make_tuple("program_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::make_tuple(std::cref(this->program_config), std::cref(this->compute_kernel_config));
    }
};

Tensor to_layout(
    const Tensor &input_tensor,
    const Layout layout,
    const std::optional<const DataType> &dtype = std::nullopt,
    const std::optional<MemoryConfig> &memory_config = std::nullopt);

}  // namespace core

}  // namespace operations
}  // namespace ttnn
