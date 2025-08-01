// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::operations::experimental::ccl {

struct LlamaReduceScatterDeviceOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const std::optional<GlobalSemaphore> cross_device_semaphore;
        const std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
        const uint32_t cluster_axis;
        const std::optional<MemoryConfig> output_mem_config;
        const uint32_t ring_devices;
        const uint32_t num_links;
        tt::tt_fabric::Topology topology;
        bool use_noc1_only;
    };
    struct tensor_args_t {
        const Tensor input_tensor;
        Tensor intermediate_packet_buffer;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = Tensor;

    struct LlamaReduceScatterAdd {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::KernelHandle quaternary_reduce_reader_kernel_id;
            tt::tt_metal::KernelHandle quaternary_reduce_writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            std::vector<tt::tt_metal::CBHandle> cb_handles;
            CoreRangeSet core_range;
        };
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at_helper(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            tt::tt_metal::Program& program);

        static shared_variables_t create_at_program_processing(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            tt::tt_metal::Program& program,
            const std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& signaler);
        static void override_runtime_arguments_per_program(
            const shared_variables_t& shared_variables,
            tt::tt_metal::Program& program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            LlamaReduceScatterDeviceOperation::tensor_return_value_t& tensor_return_value);
        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<LlamaReduceScatterAdd>;

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
    static std::tuple<CoreRangeSet, CoreRangeSet> get_rs_core_grids(
        const LlamaReduceScatterDeviceOperation::operation_attributes_t& operation_attributes,
        const LlamaReduceScatterDeviceOperation::tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        ttnn::Tensor& intermediate_packet_buffer,
        int32_t dim,
        const GlobalSemaphore& semaphore,
        tt::tt_metal::SubDeviceId subdevice_id,
        uint32_t cluster_axis,
        uint32_t ring_devices,
        uint32_t num_links,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        tt::tt_fabric::Topology topology = tt::tt_fabric::Topology::Linear,
        bool use_noc1_only = false);
};
}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {
// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::example
constexpr auto llama_reduce_scatter = ttnn::register_operation<
    "ttnn::prim::llama_reduce_scatter",
    ttnn::operations::experimental::ccl::LlamaReduceScatterDeviceOperation>();
}  // namespace ttnn::prim
