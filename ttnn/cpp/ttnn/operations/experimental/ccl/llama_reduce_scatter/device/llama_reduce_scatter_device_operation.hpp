// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
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
        // Legacy Program& builder.  Still used transitionally by op factories
        // that have not yet migrated; the descriptor variant below is the
        // preferred path.  Retained as a free declaration so the unmigrated
        // callers compile while we cut over.
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::KernelHandle quaternary_reduce_reader_kernel_id;
            tt::tt_metal::KernelHandle quaternary_reduce_writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            std::vector<tt::tt_metal::CBHandle> cb_handles;
            CoreRangeSet core_range;
        };
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

        // ProgramDescriptor variant of create_at_program_processing.  Mirrors
        // the legacy Program& builder above but constructs the per-coord
        // ProgramDescriptor in-place (CBs/kernels/semaphores/runtime args
        // recorded as descriptor pushes) so it can be consumed by Contract-2
        // factories.  Used by descriptor-based llama_reduce_scatter and
        // rs_matmul migrations.
        static tt::tt_metal::ProgramDescriptor create_at_program_processing_descriptor(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& signaler);

        // Append-style overload of create_at_program_processing_descriptor.
        // Instead of returning a fresh ProgramDescriptor, this variant appends
        // CBs/kernels/semaphores/runtime args onto an existing `desc` that the
        // caller is composing.  Used by rs_matmul_op (Contract-2), which
        // builds a single ProgramDescriptor holding both the reduce-scatter
        // half (this builder) and the matmul half (gather_in0 descriptor
        // helper) so they can share kernels on overlapping cores.
        //
        // Semaphore IDs are allocated via desc.find_available_semaphore_id()
        // (rather than desc.semaphores.size()) so a local_semaphore allocated
        // here cannot collide with semaphores the caller already registered on
        // the same cores (e.g. the signaler's rs_semaphore from
        // MatmulFusedOpSignaler::init_llama_rs_cores_rs).
        static void create_at_program_processing_descriptor(
            tt::tt_metal::ProgramDescriptor& desc,
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler>& signaler);

        // Contract-2 (WorkloadDescriptor) factory.  Builds one ProgramDescriptor
        // per mesh coord via create_at_program_processing_descriptor.  No
        // workload-scoped semaphores or intermediate Tensors are required
        // beyond what the caller already provided in operation_attributes.
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<LlamaReduceScatterAdd>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
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
};

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

ttnn::operations::experimental::ccl::LlamaReduceScatterDeviceOperation::tensor_return_value_t llama_reduce_scatter(
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

}  // namespace ttnn::prim
