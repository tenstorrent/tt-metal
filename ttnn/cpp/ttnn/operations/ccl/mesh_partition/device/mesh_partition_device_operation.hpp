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
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

namespace ttnn::operations::ccl {

struct MeshPartitionDeviceOperation {
    struct operation_attributes_t {
        uint32_t dim;
        std::optional<uint32_t> cluster_axis;
        const MemoryConfig output_mem_config;
    };
    struct tensor_args_t {
        const ttnn::Tensor input_tensor;
        const std::optional<ttnn::Tensor> optional_output_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = ttnn::Tensor;

    struct MeshPartition {
        using OverrideRuntimeArgsCallback = std::function<void(
            const void*,
            tt::tt_metal::Program&,  // ‼  no const, exact type
            const std::vector<tt::tt_metal::Tensor>&,
            const std::vector<std::optional<const tt::tt_metal::Tensor>>&,
            const std::vector<tt::tt_metal::Tensor>&)>;

        // -- shared variables --------------------------------------------
        using SliceSharedVariables = std::variant<
            prim::SliceRmProgramFactory::shared_variables_t,
            prim::SliceRmShardedProgramFactory::shared_variables_t,
            prim::SliceRmStrideProgramFactory::shared_variables_t,
            prim::SliceTileProgramFactory::shared_variables_t,
            prim::SliceTileTensorArgsProgramFactory::shared_variables_t>;

        struct shared_variables_t {
            prim::SliceDeviceOperation::program_factory_t slice_program_factory;
            SliceSharedVariables slice_shared_variables;
        };
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<MeshPartition>;

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

namespace detail {
uint32_t get_cluster_axis_size(const ttnn::Tensor& input_tensor, const std::optional<uint32_t>& cluster_axis);
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor mesh_partition(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt);
}  // namespace ttnn::prim
