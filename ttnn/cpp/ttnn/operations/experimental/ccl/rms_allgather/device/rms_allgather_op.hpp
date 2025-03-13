// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::operations::fused::normalization {

tt::tt_metal::operation::ProgramWithCallbacks frmsnorm_pre_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& b,      // residual
    const std::optional<const Tensor>& gamma,  // weight
    const std::optional<const Tensor>& beta,   // bias
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config);

tt::tt_metal::operation::ProgramWithCallbacks frmsnorm_post_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& b,      // residual
    const std::optional<const Tensor>& gamma,  // weight
    const std::optional<const Tensor>& beta,   // bias
    Tensor& output,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config);

struct RMSAllGather {
    float eps;
    MemoryConfig output_mem_config;
    ttnn::operations::normalization::LayerNormProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;
    std::optional<DataType> dtype;
    const ttnn::ccl::Topology topology;
    const bool is_pre;
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

}  // namespace ttnn::operations::fused::normalization
