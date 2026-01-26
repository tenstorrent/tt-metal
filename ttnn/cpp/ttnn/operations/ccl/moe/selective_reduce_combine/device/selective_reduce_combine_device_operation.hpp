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

namespace ttnn::operations::ccl::moe {

struct SelectiveReduceCombineDeviceOperation {
    struct operation_attributes_t {
        const uint32_t hidden_size;
        const uint32_t batch_size;
        const uint32_t seq_size;
        const uint32_t select_experts_k;
        const uint32_t experts;
        const uint32_t num_links;

        const std::optional<uint32_t> axis;
        tt::tt_fabric::Topology topology;

        const uint32_t num_token_parallel_cores;
        const uint32_t num_data_parallel_cores;
        const CoreRangeSet worker_core_range_set;
        const CoreRangeSet mux_core_range_set;
        const ttnn::MemoryConfig output_memory_config;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "hidden_size",
            "batch_size",
            "seq_size",
            "select_experts_k",
            "experts",
            "num_links",
            "axis",
            "topology",
            "num_token_parallel_cores",
            "num_data_parallel_cores",
            "worker_core_range_set",
            "mux_core_range_set",
            "output_memory_config");

        auto attribute_values() const {
            return std::forward_as_tuple(
                hidden_size,
                batch_size,
                seq_size,
                select_experts_k,
                experts,
                num_links,
                axis,
                topology,
                num_token_parallel_cores,
                num_data_parallel_cores,
                worker_core_range_set,
                mux_core_range_set,
                output_memory_config);
        };
    };
    struct tensor_args_t {
        const ttnn::Tensor dense_input_tensor;
        const ttnn::Tensor dense_metadata_tensor;
        const ttnn::Tensor dense_token_counts_tensor;
        const std::optional<ttnn::Tensor> optional_output_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = ttnn::Tensor;

    struct UnifiedSelectReduce {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
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

    using program_factory_t = std::variant<UnifiedSelectReduce>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::ccl::moe

namespace ttnn::prim {
ttnn::Tensor selective_reduce_combine(
    const ttnn::Tensor& dense_input_tensor,
    const ttnn::Tensor& dense_metadata_tensor,
    const ttnn::Tensor& dense_token_counts_tensor,
    const uint32_t hidden_size,
    const uint32_t batch_size,
    const uint32_t seq_size,
    const uint32_t select_experts_k,
    const uint32_t experts,
    const std::optional<uint32_t>& axis,
    tt::tt_fabric::Topology topology,
    const uint32_t num_links,
    const uint32_t num_token_parallel_cores,
    const uint32_t num_data_parallel_cores,
    const CoreRangeSet worker_core_range_set,
    const CoreRangeSet mux_core_range_set,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor);
}  // namespace ttnn::prim
