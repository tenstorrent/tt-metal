// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>


#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::paged_cache {

enum class PagedUpdateCacheOpParallelizationStrategy {
    MULTI_CORE
};

enum class PagedUpdateCacheOpType {
    UPDATE, FILL
};

struct PagedUpdateCacheDeviceOperation {
    const uint32_t batch_idx;
    const std::vector<uint32_t> update_idxs;
    const uint32_t batch_offset;
    const PagedUpdateCacheOpType op_type;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    const bool share_cache;

    PagedUpdateCacheOpParallelizationStrategy get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const;

    void validate(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    const std::vector<ttnn::Shape> compute_output_shapes(
        const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(
        const std::vector<Tensor> &input_tensors) const;


    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors
    ) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("batch_idx", "update_idxs", "batch_offset", "op_type", "compute_kernel_config", "share_cache");

    const auto attribute_values() const {
        return std::forward_as_tuple(batch_idx, update_idxs, batch_offset, op_type, compute_kernel_config, share_cache);
    }

    const operation::Hash compute_program_hash(
        const std::vector<Tensor> &input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

}  // namespace ttnn::operations::experimental::paged_cache
