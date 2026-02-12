// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

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
#include <vector>
#include "ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/all_to_all_dispatch_metadata.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {

std::pair<std::array<uint32_t, 7>, std::array<uint32_t, 7>> get_cb_sizes(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& scores_tensor,
    const ttnn::Tensor& mapping_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis);

}  // namespace detail

struct AllToAllDispatchMetadataDeviceOperation {
    struct operation_attributes_t {
        const CoreRangeSet worker_core_range_set;
        const std::optional<uint32_t> axis;
        const uint32_t num_links;
        const tt::tt_fabric::Topology topology;
        // Core where indices/scores are sharded for selective_tilize
        // Optional: when persistent output tensors are provided, extracted from their shard spec
        const std::optional<CoreCoord> drain_sync_tilizer_core;
        const WorkerMode worker_mode;           // Worker distribution mode (DIRECT, MUX_TOKEN_SPLIT, MUX_PAYLOAD_SPLIT)
        const CoreRangeSet mux_core_range_set;  // Cores to run mux kernels on (only used if worker_mode uses mux)
        const DispatchAlgorithm dispatch_algorithm;  // Algorithm for routing tokens to destination devices
        // Note: cross_device_semaphore is NOT included in attribute_names/attribute_values because
        // GlobalSemaphore contains CoreRangeSet in its attribute_values() which isn't supported by the visitor.
        // It's still part of the struct for use in the program factory.
        const std::optional<GlobalSemaphore> cross_device_semaphore;  // Optional external semaphore for persistent mode
        static constexpr auto attribute_names = std::forward_as_tuple(
            "worker_core_range_set",
            "axis",
            "num_links",
            "topology",
            "drain_sync_tilizer_core",
            "worker_mode",
            "mux_core_range_set",
            "dispatch_algorithm",
            "cross_device_semaphore");
        auto attribute_values() const {
            return std::forward_as_tuple(
                worker_core_range_set,
                axis,
                num_links,
                topology,
                drain_sync_tilizer_core,
                worker_mode,
                mux_core_range_set,
                dispatch_algorithm,
                cross_device_semaphore);
        };
    };
    struct tensor_args_t {
        const Tensor input_tensor;
        const Tensor expert_indices_tensor;
        const Tensor expert_scores_tensor;
        const Tensor expert_mapping_tensor;
        const std::optional<std::array<Tensor, 3>> optional_output_tensors;
        // Note: GlobalSemaphore moved to operation_attributes_t because tensor_args_t is visited
        // by visit_object_of_type<Tensor> and GlobalSemaphore's attribute_values() contains CoreRangeSet
        // which isn't supported by the visitor.

        static constexpr auto attribute_names = std::forward_as_tuple(
            "input_tensor",
            "expert_indices_tensor",
            "expert_scores_tensor",
            "expert_mapping_tensor",
            "optional_output_tensors");
        auto attribute_values() const {
            return std::forward_as_tuple(
                input_tensor,
                expert_indices_tensor,
                expert_scores_tensor,
                expert_mapping_tensor,
                optional_output_tensors);
        }
    };

    using spec_return_value_t = std::array<ttnn::TensorSpec, 3>;

    using tensor_return_value_t = std::array<Tensor, 3>;

    struct AllToAllDispatchMetadataSparse {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle ternary_reader_kernel_id;
            tt::tt_metal::KernelHandle binary_writer_kernel_id;
            std::vector<CoreCoord> cores;
            const std::optional<GlobalSemaphore> init_semaphore;  // Optional - not used in persistent mode
            const GlobalSemaphore cross_device_semaphore;
            const bool
                skip_init_semaphore;  // True when using persistent mode (all outputs persistent + external semaphore)
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
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const std::optional<GlobalSemaphore>& init_semaphore,
            const GlobalSemaphore& cross_device_semaphore,
            bool skip_init_semaphore);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<AllToAllDispatchMetadataSparse>;

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
}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {
ttnn::operations::experimental::ccl::AllToAllDispatchMetadataDeviceOperation::tensor_return_value_t
all_to_all_dispatch_metadata(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 3>>& optional_output_tensors,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const CoreRangeSet& worker_core_range_set,
    const std::optional<CoreCoord>& drain_sync_tilizer_core,
    ttnn::operations::experimental::ccl::WorkerMode worker_mode,
    const CoreRangeSet& mux_core_range_set,
    ttnn::operations::experimental::ccl::DispatchAlgorithm dispatch_algorithm,
    const std::optional<ttnn::GlobalSemaphore>& cross_device_semaphore = std::nullopt);
}  // namespace ttnn::prim
