// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ema_op.hpp"

#include <optional>

#include <tt_stl/assert.hpp>
#include "ttnn/operations/math.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::accumulation {

void Ema::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    TT_FATAL(input_tensors.size() == 1, "Must have exactly 1 input tensor, got {} tensors", input_tensors.size());

    const auto& input_tensor = input_tensors.at(0);

    // Dtype, Device and layout checks
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16, "Input tensor must be BFLOAT16, got: {}", input_tensor.dtype());
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Input tensor must be on device, got: {}",
        input_tensor.storage_type());
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must be allocated in a device buffer");
    TT_FATAL(
        input_tensor.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", input_tensor.layout());

    // Shape constraints: [1, B, C, T]
    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(input_shape.rank() == 4, "EMA input must be 4D [1, B, C, T], got rank {}", input_shape.rank());
    TT_FATAL(input_shape[0] == 1, "EMA expects leading dimension to be 1, got {}", input_shape[0]);

    // This OP produces as many elements in output as there are in input
    // Thus, the volume must be the same to avoid writing outside the output buffer
    if ((!optional_output_tensors.empty()) && (optional_output_tensors.at(0).has_value())) {
        const auto& output_tensor = optional_output_tensors.at(0).value();
        TT_FATAL(
            output_tensor.dtype() == DataType::BFLOAT16,
            "Output tensor must be BFLOAT16, got: {}",
            output_tensor.dtype());
        TT_FATAL(
            output_tensor.storage_type() == StorageType::DEVICE,
            "Output tensor must be on device, got: {}",
            output_tensor.storage_type());
        TT_FATAL(output_tensor.buffer() != nullptr, "Output tensor must be allocated in a device buffer");
        TT_FATAL(
            output_tensor.layout() == Layout::TILE,
            "Output tensor must have TILE layout, got: {}",
            output_tensor.layout());
        TT_FATAL(
            input_tensor.padded_shape().volume() == output_tensor.padded_shape().volume(),
            "Input and output must have the same volume");
    }

    // Alpha validation
    TT_FATAL(!std::isnan(this->alpha), "EMA alpha must be a valid number, got {}", this->alpha);
}

std::vector<TensorSpec> Ema::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.tensor_spec().with_memory_config(this->output_mem_config)};
}

std::vector<Tensor> Ema::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) const {
    return operation::default_create_output_tensors(*this, input_tensors, optional_output_tensors);
}

operation::ProgramWithCallbacks Ema::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    return ema_multi_core(a, output_tensor, this->alpha, this->grid_size, this->compute_kernel_config);
}

}  // namespace ttnn::operations::reduction::accumulation
