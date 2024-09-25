// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <utility>
#include <variant>
#include <vector>
#include <tuple>

#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "ttnn/common/constants.hpp"

namespace tt {

namespace operations {

namespace primary {

using namespace tt_metal;

struct MorehSum {
    int64_t dim;
    bool keep_batch_dim;
    MemoryConfig output_mem_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<ttnn::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) const;
};

operation::ProgramWithCallbacks moreh_sum_nc_impl(const Tensor &input, const Tensor &output, int64_t dim, const ttnn::DeviceComputeKernelConfig &compute_kernel_config);
// revised from reduce_op
operation::ProgramWithCallbacks moreh_sum_w_impl(const Tensor &a, const Tensor &output, const ttnn::DeviceComputeKernelConfig &compute_kernel_config);
operation::ProgramWithCallbacks moreh_sum_h_impl(const Tensor &a, const Tensor &output, const ttnn::DeviceComputeKernelConfig &compute_kernel_config);

operation::ProgramWithCallbacks moreh_sum_int_nc_impl(const Tensor &input, const Tensor &output, int64_t dim, const ttnn::DeviceComputeKernelConfig &compute_kernel_config);
operation::ProgramWithCallbacks moreh_sum_int_w_impl(const Tensor &input, const Tensor &output, const ttnn::DeviceComputeKernelConfig &compute_kernel_config);
operation::ProgramWithCallbacks moreh_sum_int_h_impl(const Tensor &input, const Tensor &output, const ttnn::DeviceComputeKernelConfig &compute_kernel_config);

Tensor moreh_sum(
    const Tensor &input,
    std::optional<std::variant<int64_t, std::vector<int64_t>>> dim = std::nullopt,
    const bool keep_batch_dim = false,
    const std::optional<const Tensor> output = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    uint8_t queue_id = ttnn::DefaultQueueId);

}  // namespace primary

}  // namespace operations

}  // namespace tt
