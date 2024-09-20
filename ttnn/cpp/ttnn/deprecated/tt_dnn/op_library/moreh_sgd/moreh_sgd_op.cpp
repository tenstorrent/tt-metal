// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_sgd/moreh_sgd_op.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void MorehSGD::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    TT_ASSERT(input_tensors.size() == 2, "Must have 2 input tensors");
    TT_ASSERT(
        optional_input_tensors.size() == 0 || optional_input_tensors.size() == 1, "Must have 0 or 1 optional tensors");

    auto& param_in = input_tensors.at(0);
    auto& grad = input_tensors.at(1);
    auto& param_out = output_tensors.at(0);

    TT_ASSERT(param_in.storage_type() == StorageType::DEVICE, "param_in to SGD need to be on device!");
    TT_ASSERT(param_in.buffer() != nullptr, "param_in to SGD need to be allocated in buffers on device!");
    TT_ASSERT((param_in.get_layout() == Layout::TILE), "param_in to SGD must be tilized");
    TT_ASSERT(param_in.get_dtype() == DataType::BFLOAT16 || param_in.get_dtype() == DataType::BFLOAT8_B);

    TT_ASSERT(grad.storage_type() == StorageType::DEVICE, "grad to SGD need to be on device!");
    TT_ASSERT(grad.buffer() != nullptr, "grad to SGD need to be allocated in buffers on device!");
    TT_ASSERT((grad.get_layout() == Layout::TILE), "grad to SGD must be tilized");
    TT_ASSERT(grad.get_dtype() == DataType::BFLOAT16 || grad.get_dtype() == DataType::BFLOAT8_B);

    if (param_out.has_value()) {
        TT_ASSERT(param_out.value().storage_type() == StorageType::DEVICE, "param_out to SGD need to be on device!");
        TT_ASSERT(param_out.value().buffer() != nullptr, "param_out to SGD need to be allocated in buffers on device!");
        TT_ASSERT((param_out.value().get_layout() == Layout::TILE), "param_out to SGD must be tilized");
        TT_ASSERT(
            param_out.value().get_dtype() == DataType::BFLOAT16 ||
            param_out.value().get_dtype() == DataType::BFLOAT8_B);
    }
}

std::vector<tt::tt_metal::LegacyShape> MorehSGD::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = input_tensors.at(0).get_legacy_shape();
    return {output_shape, output_shape};
}

std::vector<Tensor> MorehSGD::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& output_shapes = this->compute_output_shapes(input_tensors);
    auto dtype = input_tensors.at(0).get_dtype();
    Layout layout{Layout::TILE};
    auto device = input_tensors.at(0).device();

    std::vector<Tensor> result;
    result.reserve(2);

    if (output_tensors.at(0).has_value()) {
        result.push_back(output_tensors.at(0).value());
    } else {
        result.push_back(create_device_tensor(output_shapes.at(0), dtype, layout, device, this->param_out_mem_config));
    }

    if (output_tensors.at(1).has_value()) {
        result.push_back(output_tensors.at(1).value());
    } else if (this->momentum != 0.0f) {
        result.push_back(
            create_device_tensor(output_shapes.at(1), dtype, layout, device, this->momentum_buffer_out_mem_config));
    }

    return std::move(result);
}

operation::ProgramWithCallbacks MorehSGD::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    auto& param_in = input_tensors.at(0);
    auto& grad = input_tensors.at(1);
    auto& momentum_buffer_in = optional_input_tensors.at(0);
    auto& param_out = output_tensors.at(0);
    std::optional<Tensor> momentum_buffer_out =
        (output_tensors.size() == 2) ? (std::make_optional<Tensor>(output_tensors.at(1))) : (std::nullopt);

    return {moreh_sgd_(
        param_in,
        grad,
        momentum_buffer_in,
        param_out,
        momentum_buffer_out,
        this->lr,
        this->momentum,
        this->dampening,
        this->weight_decay,
        this->nesterov,
        this->momentum_initialized,
        this->core_range,
        this->compute_kernel_config)};
}

std::vector<std::optional<Tensor>> moreh_sgd(
    const Tensor& param_in,
    const Tensor& grad,
    std::optional<const Tensor> momentum_buffer_in,
    std::optional<const Tensor> param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const MemoryConfig& param_out_mem_config,
    const MemoryConfig& momentum_buffer_out_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    auto device = param_in.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    auto kernel_config_val =
        init_device_compute_kernel_config(device->arch(), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {
        Tensor(operation::get_workers_for_op_output({param_in, grad}, {momentum_buffer_in}))};

    if (momentum_buffer_out.has_value() || momentum != 0.0f) {
        output_tensors.push_back(Tensor(operation::get_workers_for_op_output({param_in, grad}, {momentum_buffer_in})));
    }

    operation::launch_op(
        [lr,
         momentum,
         dampening,
         weight_decay,
         nesterov,
         momentum_initialized,
         all_cores,
         param_out_mem_config,
         momentum_buffer_out_mem_config,
         kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehSGD{
                    .lr = lr,
                    .momentum = momentum,
                    .dampening = dampening,
                    .weight_decay = weight_decay,
                    .nesterov = nesterov,
                    .momentum_initialized = momentum_initialized,
                    .core_range = all_cores,
                    .param_out_mem_config = param_out_mem_config,
                    .momentum_buffer_out_mem_config = momentum_buffer_out_mem_config,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {param_in, grad},
        output_tensors,
        {momentum_buffer_in},
        {param_out, momentum_buffer_out});

    std::vector<std::optional<Tensor>> result;
    result.reserve(2);
    result.push_back(output_tensors.at(0));
    if (output_tensors.size() == 2) {
        result.push_back(output_tensors.at(1));
    } else {
        result.push_back(std::nullopt);
    }

    return result;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
