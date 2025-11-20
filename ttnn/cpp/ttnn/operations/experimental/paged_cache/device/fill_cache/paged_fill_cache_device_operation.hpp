// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "../update_cache/paged_update_cache_device_operation.hpp"  // For PagedUpdateCacheOpParallelizationStrategy

namespace ttnn::operations::experimental::paged_cache {

struct PagedFillCacheDeviceOperation {
    uint32_t batch_idx_fallback;
    std::optional<Tensor>
        batch_idx_tensor_opt;  // This will be handled by create_program, not directly in attributes for simple hashing
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

    static constexpr auto attribute_names = std::forward_as_tuple("batch_idx_fallback", "mesh_coords");

    auto attribute_values() const { return std::forward_as_tuple(batch_idx_fallback, mesh_coords); }

    tt::tt_metal::operation::Hash compute_program_hash(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
};

}  // namespace ttnn::operations::experimental::paged_cache
