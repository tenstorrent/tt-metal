// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "neighbor_pad_async_device_operation_types.hpp"
#include "neighbor_pad_async_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::ccl::neighbor_pad {

struct NeighborPadAsyncDeviceOperation {
    using operation_attributes_t = neighbor_pad::operation_attributes_t;
    using tensor_args_t = neighbor_pad::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<NeighborPadAsyncMeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::ccl::neighbor_pad

namespace ttnn::prim {

ttnn::operations::experimental::ccl::neighbor_pad::NeighborPadAsyncDeviceOperation::tensor_return_value_t
neighbor_pad_async(
    const Tensor& input_tensor,
    int32_t dim,
    uint32_t padding_left,
    uint32_t padding_right,
    const std::string& padding_mode,
    uint32_t cluster_axis,
    const GlobalSemaphore& final_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    std::optional<size_t> num_preferred_links,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<ttnn::ccl::Topology> topology,
    std::optional<uint32_t> secondary_cluster_axis,
    const std::optional<std::vector<uint32_t>>& secondary_mesh_shape);

}  // namespace ttnn::prim
