// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehCumSum {
    int64_t dim;
    bool flip;
    MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;
};

operation::ProgramWithCallbacks moreh_cumsum_h_impl(
    const Tensor &input,
    const Tensor &output,
    const bool &flip,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config);
operation::ProgramWithCallbacks moreh_cumsum_w_impl(
    const Tensor &input,
    const Tensor &output,
    const bool &flip,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config);
operation::ProgramWithCallbacks moreh_cumsum_nc_impl(
    const Tensor &input,
    const Tensor &output,
    const int64_t &dim,
    const bool &flip,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config);

Tensor moreh_cumsum(
    const Tensor &input,
    const int64_t &dim,
    std::optional<const Tensor> output,
    const MemoryConfig &output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);
Tensor moreh_cumsum_backward(
    const Tensor &output_grad,
    const int64_t &dim,
    std::optional<const Tensor> input_grad,
    const MemoryConfig &output_mem_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config);

}  // namespace primary

}  // namespace operations

}  // namespace tt
