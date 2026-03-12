// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

struct AllToAllDispatchBackwardDeviceOperation {
    struct operation_attributes_t {
        const MemoryConfig output_mem_config;
        const std::optional<uint32_t> axis;
        const uint32_t num_links;
        const tt::tt_fabric::Topology topology;
        const CoreRangeSet worker_core_range_set;
        const uint32_t output_shard_dim;
        static constexpr auto attribute_names = std::forward_as_tuple(
            "output_mem_config",
            "axis",
            "num_links",
            "topology",
            "worker_core_range_set",
            "output_shard_dim");
        auto attribute_values() const {
            return std::forward_as_tuple(
                output_mem_config, axis, num_links, topology, worker_core_range_set, output_shard_dim);
        };
    };
    struct tensor_args_t {
        const ttnn::Tensor grad_output;
        const ttnn::Tensor mapping_tensor;
        const ttnn::Tensor metadata_tensor;
        const std::optional<ttnn::Tensor> optional_output_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = ttnn::Tensor;

    struct AllToAllDispatchBackwardToDense {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle ternary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            std::vector<CoreCoord> cores;
            const GlobalSemaphore init_semaphore;
            const GlobalSemaphore cross_device_semaphore;
        };
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const GlobalSemaphore& init_semaphore,
            const GlobalSemaphore& cross_device_semaphore);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<AllToAllDispatchBackwardToDense>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor all_to_all_dispatch_backward(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& expert_metadata_tensor,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<uint32_t>& axis,
    const std::optional<ttnn::Tensor>& optional_output_tensor,
    const CoreRangeSet& worker_core_range_set,
    uint32_t output_shard_dim);
}  // namespace ttnn::prim
