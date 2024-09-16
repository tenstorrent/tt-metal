// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_program_factory.hpp"

#include <numeric>

#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction::detail {

Tensor _fast_reduce_nc(
    uint8_t queue_id,
    const ttnn::Tensor& input,
    const int32_t& dim,
    const std::optional<const ttnn::Tensor>& output,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input}))};

    TT_FATAL(input.storage_type() == StorageType::DEVICE || input.storage_type() == StorageType::MULTI_DEVICE);
    auto kernel_config_val = init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);

    operation::launch_op(
        [dim, output_mem_config, kernel_config_val, queue_id](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                FastReduceNCDeviceOperation{.dim = dim, .output_mem_config = output_mem_config, .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors,
                queue_id);
        },
        {input},
        output_tensors,
        {},
        {output});

    return output_tensors.at(0);
}

void FastReduceNCDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    // validate tensor
    tt::operations::primary::check_tensor(input, "FastReduceNC", "input", {DataType::BFLOAT16, DataType::BFLOAT8_B});
    tt::operations::primary::check_tensor(output, "FastReduceNC", "output", {DataType::BFLOAT16, DataType::BFLOAT8_B});

    // validate input dim
    auto input_shape = input.get_legacy_shape();
    auto input_shape_wo_padding = input.get_legacy_shape().without_padding();
    const auto input_rank = input_shape.rank();
    TT_FATAL(
        (this->dim >= 0 && this->dim <= tt::tt_metal::MAX_NUM_DIMENSIONS - 2),
        "dim must be between 0 and {}.",
        tt::tt_metal::MAX_NUM_DIMENSIONS - 2);
    TT_FATAL((this->dim < input_rank), "dim must be smaller than input tensor rank {}.", input_rank);
}

std::vector<tt::tt_metal::LegacyShape> FastReduceNCDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto& input_shape = input.get_legacy_shape();
    const auto input_rank = input_shape.rank();

    // keepdim=true
    auto output_shape = input_shape;
    auto padding = output_shape.padding();

    // last 2-dim
    output_shape[this->dim] = 1;

    output_shape = tt::tt_metal::LegacyShape(output_shape, padding);
    return {output_shape};
}

std::vector<Tensor> FastReduceNCDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks FastReduceNCDeviceOperation::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    auto& input = inputs.at(0);
    auto& output = outputs.at(0);

    return reduce_nc_factory(input, output, dim, this->compute_kernel_config);
}

Tensor fast_reduce_nc(
    uint8_t queue_id,
    const ttnn::Tensor& input,
    const std::vector<int32_t>& dims,
    const std::optional<const ttnn::Tensor> output,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {

    std::vector<int32_t> sorted_dims = dims;
    std::sort(sorted_dims.begin(), sorted_dims.end());

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        auto temp_output = _fast_reduce_nc(queue_id, temp_input, sorted_dims[i], std::nullopt, output_mem_config, compute_kernel_config);
        temp_input = temp_output;
    }
    return _fast_reduce_nc(queue_id, temp_input, sorted_dims.front(), output, output_mem_config, compute_kernel_config);
}

}  // namespace ttnn::operations::experimental::reduction::detail
