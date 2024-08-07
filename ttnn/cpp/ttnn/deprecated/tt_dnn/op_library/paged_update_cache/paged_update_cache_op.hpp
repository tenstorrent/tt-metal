// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/compute_kernel_config.hpp"

namespace tt {
namespace operations {
namespace primary {
using namespace tt_metal;

enum class PagedUpdateCacheOpParallelizationStrategy {
    MULTI_CORE
};

enum class PagedUpdateCacheOpType {
    UPDATE
};

operation::ProgramWithCallbacks paged_update_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, const std::vector<uint32_t> update_idxs, const uint32_t batch_offset, DeviceComputeKernelConfig compute_kernel_config);

struct PagedUpdateCache {
    const uint32_t batch_idx;
    const std::vector<uint32_t> update_idxs;
    const uint32_t batch_offset;
    const PagedUpdateCacheOpType op_type;
    const DeviceComputeKernelConfig compute_kernel_config;

    PagedUpdateCacheOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor> &input_tensors) const;
    std::vector<Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;


    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor> &input_tensors,
        std::vector<Tensor> &output_tensors) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("batch_idx", "update_idxs", "batch_offset", "op_type", "compute_kernel_config");

    const auto attribute_values() const {
        return std::forward_as_tuple(batch_idx, update_idxs, batch_offset, op_type, compute_kernel_config);
    }

    const operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors) const;
};

namespace transformers {
inline Tensor paged_update_cache(const Tensor& cache_tensor, const Tensor& input_tensor, const std::vector<uint32_t> update_idxs, const uint32_t batch_offset, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {
    operation::launch_op(
        [update_idxs, batch_offset, compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& cache_tensor = input_tensors.at(0);
            auto& input_tensor = input_tensors.at(1);
            auto kernel_config_val = init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);
            return operation::run(PagedUpdateCache{0, update_idxs, batch_offset, PagedUpdateCacheOpType::UPDATE, kernel_config_val}, {cache_tensor, input_tensor});
        }, {cache_tensor, input_tensor});
    return cache_tensor;
}
} // namespace transformers


}   // namespace primary
}   // namespace operations
}   // namespace tt
