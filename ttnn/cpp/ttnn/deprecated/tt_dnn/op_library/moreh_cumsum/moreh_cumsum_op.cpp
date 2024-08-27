// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_op.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"

namespace tt {
using namespace constants;
namespace operations {
namespace primary {

void MorehCumSum::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input = input_tensors.at(0);
    auto& output = output_tensors.at(0);

    check_tensor(input, "moreh_cumsum", "input", {DataType::BFLOAT16});
    check_tensor(output, "moreh_cumsum", "output", {DataType::BFLOAT16});

    if (output.has_value()) {
        const auto& input_shape = input.get_legacy_shape();
        const auto& input_shape_wo_padding = input_shape.without_padding();
        const auto& output_shape = output.value().get_legacy_shape();
        const auto& output_shape_wo_padding = output_shape.without_padding();

        for (int i = 0; i < input_shape.rank(); ++i) {
            TT_ASSERT(input_shape[i] == output_shape[i]);
            TT_ASSERT(input_shape_wo_padding[i] == output_shape_wo_padding[i]);
        }

        TT_ASSERT(input.get_dtype() == output.value().get_dtype());
    }
}

std::vector<Shape> MorehCumSum::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input = input_tensors.at(0);
    return {input.get_legacy_shape()};
}

std::vector<Tensor> MorehCumSum::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        log_debug(LogOp, "{}:{} use output tensor", __func__, __LINE__);
        return {output_tensors.at(0).value()};
    }

    log_debug(LogOp, "{}:{} create output tensor", __func__, __LINE__);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks MorehCumSum::create_program(
    const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) const {
    auto& input = inputs.at(0);
    auto& output = outputs.at(0);

    if (dim + 2 >= input.get_legacy_shape().rank()) {
        TT_ASSERT(false, "currenty last 2 dims are not supported");
    }

    return moreh_cumsum_nc_impl(input, output, dim, flip, compute_kernel_config);
}

Tensor moreh_cumsum_(
    const Tensor& input,
    const int64_t& dim,
    std::optional<const Tensor> output,
    const bool flip,
    const MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input}))};

    TT_FATAL(input.storage_type() == StorageType::DEVICE || input.storage_type() == StorageType::MULTI_DEVICE);
    auto kernel_config_val =
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);

    operation::launch_op(
        [dim, flip, output_mem_config, kernel_config_val](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehCumSum{
                    .dim = dim,
                    .flip = flip,
                    .output_mem_config = output_mem_config,
                    .compute_kernel_config = kernel_config_val},
                input_tensors,
                optional_input_tensors,
                optional_output_tensors);
        },
        {input},
        output_tensors,
        {},
        {output});

    return output_tensors.at(0);
}

Tensor moreh_cumsum(
    const Tensor& input,
    const int64_t& dim,
    std::optional<const Tensor> output,
    const MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return moreh_cumsum_(input, dim, output, false, output_mem_config, compute_kernel_config);
}

Tensor moreh_cumsum_backward(
    const Tensor& output_grad,
    const int64_t& dim,
    std::optional<const Tensor> input_grad,
    const MemoryConfig& output_mem_config,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    return moreh_cumsum_(output_grad, dim, input_grad, true, output_mem_config, compute_kernel_config);
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
