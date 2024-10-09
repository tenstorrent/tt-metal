/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"

#include <optional>

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

void get_tensor_dim(std::vector<uint32_t> &dim, const tt::tt_metal::LegacyShape& shape);
std::vector<int64_t> find_reduce_dim(const tt::tt_metal::LegacyShape& a_shape, const tt::tt_metal::LegacyShape& b_shape);
bool is_same_batch_dim(const Tensor &tensor_a, const Tensor &tensor_b);

operation::ProgramWithCallbacks moreh_matmul_multi_core(
    const Tensor &input,
    const Tensor &other,
    const Tensor &output,
    const std::optional<const Tensor> &bias,
    bool transpose_input,
    bool transpose_other,
    const ttnn::DeviceComputeKernelConfig &compute_kernel_config);

struct MorehMatmul {
    const MemoryConfig output_mem_config;
    bool transpose_input;
    bool transpose_other;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    void validate_with_output_tensors(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        const std::vector<std::optional<Tensor>> &output_tensors) const;
    std::vector<ttnn::SimpleShape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>> &optional_input_tensors,
        std::vector<Tensor> &output_tensors) const;
};

Tensor moreh_matmul(
    const Tensor &input,
    const Tensor &other,
    bool transpose_input = false,
    bool transpose_other = false,
    const std::optional<const Tensor> output = std::nullopt,
    const std::optional<const Tensor> bias = std::nullopt,
    const MemoryConfig &output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt);

}  // namespace primary

}  // namespace operations

}  // namespace tt
