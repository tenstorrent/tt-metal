// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::matmul {

tt::tt_metal::operation::ProgramWithCallbacks multi_core_attn_matmul(
    const Tensor& a,
    const Tensor& b,
    Tensor& output,
    std::optional<const uint32_t> num_tokens,
    std::optional<const bool> transpose_hw,
    CoreCoord compute_with_storage_grid_size,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);

struct AttnMatmulDeviceOperation {
    std::optional<const uint32_t> num_tokens;
    std::optional<const bool> transpose_hw;
    CoreCoord compute_with_storage_grid_size;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

}  // namespace ttnn::operations::experimental::matmul
