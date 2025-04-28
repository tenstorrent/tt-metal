// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::kv_cache {

enum class UpdateCacheOpParallelizationStrategy { MULTI_CORE };

enum class UpdateCacheOpType { FILL, UPDATE };

tt::tt_metal::operation::ProgramWithCallbacks update_cache_multi_core(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const uint32_t update_idx,
    const uint32_t batch_offset,
    ttnn::DeviceComputeKernelConfig compute_kernel_config);
tt::tt_metal::operation::ProgramWithCallbacks fill_cache_multi_core(
    const Tensor& cache_tensor, const Tensor& input_tensor, const uint32_t batch_idx, const uint32_t update_idx);

struct UpdateCache {
    const uint32_t batch_idx;
    const uint32_t update_idx;
    const uint32_t batch_offset;
    const UpdateCacheOpType op_type;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;

    UpdateCacheOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;

    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

inline Tensor fill_cache_impl(const Tensor& cache_tensor, const Tensor& input_tensor, const uint32_t batch_idx) {
    tt::tt_metal::operation::run(UpdateCache{batch_idx, 0, 0, UpdateCacheOpType::FILL}, {cache_tensor, input_tensor});
    return cache_tensor;
}

inline Tensor update_cache_impl(
    const Tensor& cache_tensor,
    const Tensor& input_tensor,
    const uint32_t update_idx,
    const uint32_t batch_offset,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    auto kernel_config_val = init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);
    tt::tt_metal::operation::run(
        UpdateCache{0, update_idx, batch_offset, UpdateCacheOpType::UPDATE, kernel_config_val},
        {cache_tensor, input_tensor});

    return cache_tensor;
}

}  // namespace ttnn::operations::kv_cache
