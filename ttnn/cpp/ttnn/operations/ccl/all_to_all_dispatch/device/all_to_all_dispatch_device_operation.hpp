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
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn::operations::ccl {

struct AllToAllDispatchDeviceOperation {
    struct operation_attributes_t {
        const tt::tt_metal::SubDeviceId subdevice_id;
        const MemoryConfig output_mem_config;
        const std::optional<uint32_t> axis;
        const uint32_t num_links;
        const tt::tt_fabric::Topology topology;
        const std::optional<GlobalSemaphore> cross_device_semaphore;
    };
    struct tensor_args_t {
        const Tensor input_tensor;
        const Tensor expert_indices_tensor;
        const Tensor expert_mapping_tensor;
        const std::optional<std::array<Tensor, 2>> optional_output_tensors;
    };

    using spec_return_value_t = std::array<ttnn::TensorSpec, 2>;

    using tensor_return_value_t = std::array<Tensor, 2>;

    struct AllToAllDispatchSparse {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle ternary_reader_kernel_id;
            tt::tt_metal::KernelHandle binary_writer_kernel_id;
            CoreCoord core;
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
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<AllToAllDispatchSparse>;

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

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& expert_indices_tensor,
        const ttnn::Tensor& expert_mapping_tensor,
        std::optional<uint32_t> axis,
        const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors,
        uint32_t num_links,
        tt::tt_fabric::Topology topology,
        const ttnn::MemoryConfig& memory_config,
        tt::tt_metal::SubDeviceId subdevice_id,
        const std::optional<GlobalSemaphore>& global_semaphore);
};
}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::example
constexpr auto all_to_all_dispatch = ttnn::
    register_operation<"ttnn::prim::all_to_all_dispatch", ttnn::operations::ccl::AllToAllDispatchDeviceOperation>();
}  // namespace ttnn::prim
