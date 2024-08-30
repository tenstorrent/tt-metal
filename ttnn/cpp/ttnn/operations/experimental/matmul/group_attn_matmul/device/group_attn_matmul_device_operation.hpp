// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::matmul {

// TODO: Group attention matmul will support sharding, mcasting, and should be faster; we should make attn_matmul (ie. KV heads = 1) a special case of group_attn_matmul and run the same op
operation::ProgramWithCallbacks multi_core_group_attn_matmul(const Tensor &a, const Tensor &b, Tensor& output, std::optional<const uint32_t> num_tokens, std::optional<const bool> transpose_hw, const uint32_t out_subblock_w, CoreCoord compute_with_storage_grid_size, const bool row_major, ttnn::DeviceComputeKernelConfig compute_kernel_config);

struct GroupAttnMatmulDeviceOperation {
    std::optional<const uint32_t> num_tokens;
    std::optional<const bool> transpose_hw;
    const uint32_t out_subblock_w;
    CoreCoord compute_with_storage_grid_size;
    MemoryConfig output_mem_config;
    DataType output_dtype;
    const bool row_major;  // Specifies how work is distributed across cores
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
    const operation::Hash compute_program_hash(const std::vector<Tensor> &input_tensors) const;
};

}  // namespace ttnn::operations::experimental::matmul
