// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "groupnorm_types.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

/**
Ref: https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
>>> input = torch.randn(20, 6, 10, 10)
>>> # Separate 6 channels into 3 groups
>>> m = nn.GroupNorm(3, 6)
>>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
>>> m = nn.GroupNorm(6, 6)
>>> # Put all 6 channels into a single group (equivalent with LayerNorm)
>>> m = nn.GroupNorm(1, 6)
>>> # Activating the module
>>> output = m(input)
*/

operation::ProgramWithCallbacks groupnorm_multi_core(
    const Tensor& a,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& input_mask,
    Tensor& output,
    float eps,
    uint32_t num_groups,
    uint32_t num_batches,
    DataType im_data_format,
    CoreCoord grid_size,
    bool inplace,
    uint32_t num_out_blocks,
    const DeviceComputeKernelConfig& compute_kernel_config);

operation::ProgramWithCallbacks groupnorm_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& input_mask,
    Tensor& output,
    float eps,
    uint32_t num_groups,
    uint32_t num_batches,
    tt::tt_metal::DataType im_data_format,
    CoreCoord grid_size,
    bool inplace,
    const DeviceComputeKernelConfig& compute_kernel_config);

struct GroupNorm {
    float eps;
    uint32_t num_groups;
    MemoryConfig output_mem_config;
    GroupNormProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::normalization
