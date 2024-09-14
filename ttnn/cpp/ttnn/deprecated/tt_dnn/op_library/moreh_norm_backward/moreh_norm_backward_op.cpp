// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/op_library/moreh_norm_backward/moreh_norm_backward_op.hpp"

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

void MorehNormBackward::validate_with_output_tensors(
    const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);
    const auto &output_grad = input_tensors.at(2);

    const auto &input_grad = output_tensors.at(0);

    check_tensor(input, "moreh_norm_backward", "input");
    check_tensor(output, "moreh_norm_backward", "output");
    check_tensor(output_grad, "moreh_norm_backward", "output_grad");

    check_tensor(input_grad, "moreh_norm_backward", "input_grad");
}

std::vector<Shape> MorehNormBackward::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> MorehNormBackward::create_output_tensors(const std::vector<Tensor> &input_tensors,
    const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    auto input = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input.get_dtype(), input.get_layout(), this->memory_config);
}

operation::ProgramWithCallbacks MorehNormBackward::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input = input_tensors.at(0);
    const auto &output = input_tensors.at(1);
    const auto &output_grad = input_tensors.at(2);
    const auto &input_grad = output_tensors.at(0);

    return moreh_norm_backward_(input, output, output_grad, this->p, this->dims, this->keepdim, input_grad, this->compute_kernel_config);
}

Tensor moreh_norm_backward(
    const Tensor &input,
    const Tensor &output,
    const Tensor &output_grad,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<const Tensor> input_grad,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config
    ) {
    return moreh_norm_backward_impl(input, output, output_grad, p, dim, keepdim, input_grad, memory_config, compute_kernel_config);
}

Tensor moreh_norm_backward_impl(
    const Tensor &input,
    const Tensor &output,
    const Tensor &output_grad,
    float p,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
    const bool keepdim,
    const std::optional<const Tensor> input_grad,
    const std::optional<MemoryConfig> &memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config
    ) {

    uint32_t rank = input.get_legacy_shape().rank();
    std::vector<int64_t> dims = get_dim(dim, rank);
    std::sort(dims.begin(), dims.end());

    auto device = input.device();

    auto kernel_config_val =
        init_device_compute_kernel_config(DeviceArch(device), compute_kernel_config, MathFidelity::HiFi4);

    std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({input, output, output_grad}))};
    operation::launch_op(
        [p, dims, keepdim, memory_config, kernel_config_val](
            const std::vector<Tensor> &input_tensors,
            const std::vector<std::optional<const Tensor>> &optional_input_tensors,
            const std::vector<std::optional<Tensor>> &optional_output_tensors) mutable -> std::vector<Tensor> {
            return operation::run(
                MorehNormBackward{.p = p,
                .dims=dims,
                .keepdim=keepdim,
                .memory_config=memory_config.value_or(input_tensors.at(0).memory_config()),
                .compute_kernel_config = kernel_config_val,
                }, input_tensors, optional_input_tensors, optional_output_tensors);
        },
        {input, output, output_grad},
        output_tensors,
        {},
        {input_grad});

    return output_tensors.at(0);
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
