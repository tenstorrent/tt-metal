// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_op.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sum/moreh_sum_op.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehNllLossStep1::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 1, "Must have 1 input tensors");
    TT_ASSERT(optional_input_tensors.size() == 1, "Must have 1 optional input tensors");

    auto& target_tensor = input_tensors.at(0);
    auto& weight_tensor = optional_input_tensors.at(0);

    TT_ASSERT(target_tensor.storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
    TT_ASSERT(target_tensor.buffer() != nullptr, "Operands to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");

    if (weight_tensor.has_value()) {
        TT_ASSERT(
            weight_tensor.value().storage_type() == StorageType::DEVICE, "Operands to nll_loss need to be on device!");
        TT_ASSERT(
            weight_tensor.value().buffer() != nullptr,
            "Operands to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT(weight_tensor.value().get_dtype() == DataType::BFLOAT16);
    }
}

std::vector<Shape> MorehNllLossStep1::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& target_tensor = input_tensors.at(0);
    auto target_shape = target_tensor.get_legacy_shape();

    return {target_shape};
}

std::vector<Tensor> MorehNllLossStep1::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& target_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->output_dtype, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehNllLossStep1::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& target = input_tensors.at(0);
    auto& weight = optional_input_tensors.at(0);
    auto& output = output_tensors.at(0);

    return {moreh_nll_loss_step1_impl(
        target,
        weight,
        output,
        this->ignore_index,
        this->reduction_mean,
        this->channel_size,
        this->core_range,
        this->compute_kernel_config)};
}

void MorehNllLossStep2::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_ASSERT(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_ASSERT(optional_input_tensors.size() == 2, "Must have 2 optional input tensors");

    auto& input_tensor = input_tensors.at(0);
    auto& target_tensor = input_tensors.at(1);
    auto& weight_tensor = optional_input_tensors.at(0);
    auto& divisor_tensor = optional_input_tensors.at(1);

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "intput_tensor to nll_loss need to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "intput_tensor to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((input_tensor.get_layout() == Layout::TILE), "intput_tensor to nll_loss must be tilized");
    TT_ASSERT(input_tensor.get_dtype() == DataType::BFLOAT16);

    TT_ASSERT(target_tensor.storage_type() == StorageType::DEVICE, "target_tensor to nll_loss need to be on device!");
    TT_ASSERT(
        target_tensor.buffer() != nullptr, "target_tensor to nll_loss need to be allocated in buffers on device!");
    TT_ASSERT((target_tensor.get_layout() == Layout::TILE), "target_tensor to nll_loss must be tilized");
    TT_ASSERT(target_tensor.get_dtype() == DataType::INT32);

    if (weight_tensor.has_value()) {
        TT_ASSERT(
            weight_tensor.value().storage_type() == StorageType::DEVICE,
            "weight_tensor to nll_loss need to be on device!");
        TT_ASSERT(
            weight_tensor.value().buffer() != nullptr,
            "weight_tensor to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT(
            (weight_tensor.value().get_layout() == Layout::TILE),
            "weight_tensor to nll_loss must be in row major layout");
        TT_ASSERT(weight_tensor.value().get_dtype() == DataType::BFLOAT16);
    }

    if (divisor_tensor.has_value()) {
        TT_ASSERT(
            divisor_tensor.value().storage_type() == StorageType::DEVICE,
            "divisor_tensor to nll_loss need to be on device!");
        TT_ASSERT(
            divisor_tensor.value().buffer() != nullptr,
            "divisor_tensor to nll_loss need to be allocated in buffers on device!");
        TT_ASSERT((divisor_tensor.value().get_layout() == Layout::TILE), "divisor_tensor to nll_loss must be tilized");
        TT_ASSERT(divisor_tensor.value().get_dtype() == DataType::BFLOAT16);
    }
}

std::vector<Shape> MorehNllLossStep2::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto input_shape = input_tensor.get_legacy_shape();
    auto input_shape_without_padding = input_shape.without_padding();
    auto input_rank = input_shape.rank();

    auto C = input_shape[1];

    auto dimensions_pads = std::vector<Padding::PadDimension>();
    std::vector<uint32_t> output_shape_vec;

    // Need extend 1d output to 2d, because TT not support 1d tensor
    if (input_rank == 2) {
        output_shape_vec.push_back(1);
        dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 0});
    }

    for (uint32_t dim = 0; dim < input_rank; dim++) {
        // skip C dim
        if (dim == 1) {
            continue;
        }

        output_shape_vec.push_back(input_shape_without_padding[dim]);
        dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 0});
    }

    // padding output
    {
        uint32_t output_rank = output_shape_vec.size();
        for (uint32_t dim = output_rank - 2; dim < output_rank; dim++) {
            uint32_t up32_shape = round_up(output_shape_vec[dim], 32);
            uint32_t padding_back = up32_shape - output_shape_vec[dim];

            output_shape_vec[dim] = up32_shape;
            dimensions_pads[dim].back = padding_back;
        }
    }

    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    auto output_shape = Shape(output_shape_vec, padding);

    return {output_shape};
}

std::vector<Tensor> MorehNllLossStep2::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehNllLossStep2::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& input = input_tensors.at(0);
    auto& target = input_tensors.at(1);
    auto& weight = optional_input_tensors.at(0);
    auto& divisor = optional_input_tensors.at(1);
    auto& output = output_tensors.at(0);

    return {moreh_nll_loss_step2_impl(
        input,
        target,
        weight,
        divisor,
        output,
        this->ignore_index,
        this->reduction_mean,
        this->core_range,
        this->compute_kernel_config)};
}

Tensor moreh_nll_loss_step1(
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const DataType output_dtype,
    const uint32_t channel_size,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = target_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(
        operation::get_workers_for_op_output({target_tensor}, {weight_tensor}))};

    operation::launch_op(
        [ignore_index, reduction_mean, output_dtype, channel_size, output_mem_config, all_cores, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNllLossStep1{
                   .ignore_index = ignore_index,
                   .reduction_mean = reduction_mean,
                   .output_dtype = output_dtype,
                   .channel_size = channel_size,
                   .output_mem_config = output_mem_config,
                   .core_range = all_cores,
                   .compute_kernel_config = kernel_config_val,
                },
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {target_tensor},
        output_tensors,
        {weight_tensor},
        {});
    return output_tensors.at(0);
}

Tensor moreh_nll_loss_step2(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(
        operation::get_workers_for_op_output({input_tensor, target_tensor}, {weight_tensor, divisor_tensor}))};

    operation::launch_op(
        [ignore_index, reduction_mean, output_mem_config, all_cores, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNllLossStep2{
                   .ignore_index = ignore_index,
                   .reduction_mean = reduction_mean,
                   .output_mem_config = output_mem_config,
                   .core_range = all_cores,
                   .compute_kernel_config = kernel_config_val,
                },
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {input_tensor, target_tensor},
        output_tensors,
        {weight_tensor, divisor_tensor},
        {});
    return output_tensors.at(0);
}

Tensor moreh_nll_loss(
    const Tensor& input_tensor,
    const Tensor& target_tensor,
    const std::optional<const Tensor> weight_tensor,
    const std::optional<const Tensor> divisor_tensor,
    const std::optional<const Tensor> output_tensor,
    const int32_t ignore_index,
    const bool reduction_mean,
    const MemoryConfig& output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    if (reduction_mean) {
        TT_ASSERT(divisor_tensor.has_value());

        auto input_shape = input_tensor.get_legacy_shape();
        const uint32_t channel_size = input_shape[1];
        auto output_dtype = output_tensor.has_value() ? output_tensor.value().get_dtype() : input_tensor.get_dtype();

        const Tensor& step1_result = moreh_nll_loss_step1(
            target_tensor,
            weight_tensor,
            ignore_index,
            reduction_mean,
            output_dtype,
            channel_size,
            output_mem_config,
            compute_kernel_config);

        moreh_sum(step1_result, std::nullopt, false, divisor_tensor.value(), output_mem_config, compute_kernel_config);

        const Tensor& step2_result = moreh_nll_loss_step2(
            input_tensor,
            target_tensor,
            weight_tensor,
            divisor_tensor,
            ignore_index,
            reduction_mean,
            output_mem_config,
            compute_kernel_config);
        return moreh_sum(step2_result, std::nullopt, false, output_tensor, output_mem_config, compute_kernel_config);
    } else {
        const Tensor& step2_result = moreh_nll_loss_step2(
            input_tensor,
            target_tensor,
            weight_tensor,
            std::nullopt,
            ignore_index,
            reduction_mean,
            output_mem_config,
            compute_kernel_config);

        return moreh_sum(step2_result, std::nullopt, false, output_tensor, output_mem_config, compute_kernel_config);
    }
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
