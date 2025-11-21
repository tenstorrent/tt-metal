// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::paged_cache {

enum class PagedUpdateCacheOpParallelizationStrategy { MULTI_CORE };

enum class PagedUpdateCacheOpType { UPDATE, FUSED_UPDATE, FILL };

struct PagedUpdateCacheDeviceOperation {
    uint32_t batch_idx_fallback;
    std::optional<Tensor>
        batch_idx_tensor_opt;  // This will be handled by create_program, not directly in attributes for simple hashing

    const std::vector<uint32_t> update_idxs;
    const uint32_t batch_offset;
    const PagedUpdateCacheOpType op_type;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    const bool share_cache;
    const std::optional<std::set<ttnn::MeshCoordinate>>
        mesh_coords;  // Optional mesh coordinates to use for the operation

    PagedUpdateCacheOpParallelizationStrategy get_parallelization_strategy(
        const std::vector<Tensor>& input_tensors) const;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& _,  // Unused
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "batch_idx_fallback",
        "update_idxs",
        "batch_offset",
        "op_type",
        "compute_kernel_config",
        "share_cache",
        "mesh_coords");

    auto attribute_values() const {
        return std::forward_as_tuple(
            batch_idx_fallback, update_idxs, batch_offset, op_type, compute_kernel_config, share_cache, mesh_coords);
    }

    tt::tt_metal::operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

}  // namespace ttnn::operations::experimental::paged_cache
