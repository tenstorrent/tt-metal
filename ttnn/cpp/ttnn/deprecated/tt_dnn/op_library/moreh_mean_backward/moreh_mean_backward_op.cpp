// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_mean_backward/moreh_mean_backward_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace tt {

using namespace constants;

namespace operations {
namespace primary {

////////////////////////////////////////////////////////////////////////////
//                         MorehMeanBackward
////////////////////////////////////////////////////////////////////////////
void MorehMeanBackward::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& output_grad = input_tensors.at(0);
    const auto& input_grad = output_tensors.at(0);

    TT_FATAL(
        input_grad.has_value() || this->input_grad_shape.has_value() || this->keepdim,
        "Either input_grad tensor or input_grad_shape or keepdim must be present");

    check_tensor(output_grad, "moreh_mean_backward", "output_grad", {DataType::BFLOAT16});
    check_tensor(input_grad, "moreh_mean_backward", "input_grad", {DataType::BFLOAT16});
}

operation::ProgramWithCallbacks MorehMeanBackward::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    auto& output_grad = input_tensors.at(0);
    auto& input_grad = output_tensors.at(0);

    return moreh_mean_backward_impl(output_grad, input_grad, this->dims, this->keepdim, this->compute_kernel_config);
}

std::vector<Tensor> MorehMeanBackward::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    }

    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors[0].get_dtype(), Layout::TILE, this->memory_config);
}

std::vector<tt::tt_metal::LegacyShape> MorehMeanBackward::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto input_grad_shape = this->input_grad_shape.value();
    auto rank = input_grad_shape.rank();

    std::vector<uint32_t> shape;
    std::vector<Padding::PadDimension> dimensions_pads;

    for (uint32_t dim = 0; dim < rank; dim++) {
        if (is_hw_dim(dim, rank)) {
            uint32_t up32_shape = round_up(input_grad_shape[dim], 32);
            uint32_t padding_back = up32_shape - input_grad_shape[dim];
            shape.push_back(up32_shape);
            dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = padding_back});

        } else {
            shape.push_back(input_grad_shape[dim]);
            dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 0});
        }
    }

    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape = tt::tt_metal::LegacyShape(shape, padding);

    return {output_shape};
}

Tensor moreh_mean_backward_(
    const Tensor& output_grad,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    std::optional<tt::tt_metal::LegacyShape> input_grad_shape,
    const std::optional<const Tensor> input_grad,
    const std::optional<MemoryConfig> memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto output_grad_rank = output_grad.get_legacy_shape().rank();
    auto input_grad_rank = output_grad_rank;
    if (keepdim == false) {
        if (!dim.has_value()) {
            // do nothing
        } else if (std::holds_alternative<int64_t>(dim.value())) {
            input_grad_rank += 1;
        } else {
            auto dims = std::get<std::vector<int64_t>>(dim.value());
            input_grad_rank += dims.size();
        }
    }

    std::vector<int64_t> dims = get_dim(dim, input_grad_rank);

    std::vector<Tensor> dummy_output_tensors = {Tensor(operation::get_workers_for_op_output({output_grad}))};

    auto device = output_grad.device();
    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    operation::launch_op(
        [keepdim, dims, input_grad_shape, memory_config, all_cores, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehMeanBackward{
                    .dims = dims,
                    .keepdim = keepdim,
                    .input_grad_shape = input_grad_shape,
                    .memory_config = memory_config.value_or(input_tensors.at(0).memory_config()),
                    .core_range = all_cores,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {output_grad},
        dummy_output_tensors,
        {},
        {input_grad});

    return dummy_output_tensors.at(0);
}

Tensor moreh_mean_backward(
    const Tensor& output_grad,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    std::optional<tt::tt_metal::LegacyShape> input_grad_shape,
    const std::optional<const Tensor> input_grad,
    const std::optional<MemoryConfig> memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return moreh_mean_backward_(
        output_grad, dim, keepdim, input_grad_shape, input_grad, memory_config, compute_kernel_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
