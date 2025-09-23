// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "layernorm_types.hpp"

namespace ttnn::operations::normalization {

tt::tt_metal::operation::ProgramWithCallbacks layernorm_multi_core(
    const Tensor& a,
    const std::optional<const Tensor>& b,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    DeviceComputeKernelConfig compute_kernel_config);

tt::tt_metal::operation::ProgramWithCallbacks layernorm_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& b,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& stats,
    Tensor& output,
    LayerNormType norm_type,
    DistributedLayerNormStage distributed_norm_stage,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config);

struct LayerNorm {
    LayerNormType norm_type;
    DistributedLayerNormStage distributed_norm_stage;
    float eps;
    MemoryConfig output_mem_config;
    LayerNormProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;

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
