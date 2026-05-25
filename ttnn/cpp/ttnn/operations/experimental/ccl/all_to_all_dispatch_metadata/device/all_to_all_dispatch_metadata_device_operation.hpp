// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
#include <tt-metalium/workload_descriptor.hpp>
#include <vector>
#include "ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/all_to_all_dispatch_metadata.hpp"

namespace ttnn::operations::experimental::ccl {

struct AllToAllDispatchMetadataDeviceOperation {
    struct operation_attributes_t {
        const std::optional<std::vector<uint32_t>> shared_expert_ids;
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
            "shared_expert_ids",
            "worker_core_range_set",
            "axis",
            "num_links",
            "topology",
            "drain_sync_tilizer_core",
            "worker_mode",
            "mux_core_range_set",
            "dispatch_algorithm");
        auto attribute_values() const {
            return std::forward_as_tuple(
                shared_expert_ids,
                worker_core_range_set,
                axis,
                num_links,
                topology,
                drain_sync_tilizer_core,
                worker_mode,
                mux_core_range_set,
                dispatch_algorithm);
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
    };

    using spec_return_value_t = std::array<ttnn::TensorSpec, 3>;

    using tensor_return_value_t = std::array<Tensor, 3>;

    struct AllToAllDispatchMetadataSparse {
        // Contract-2: declarative WorkloadDescriptor.  Per coord, builds a
        // ProgramDescriptor containing the mux (when worker_mode != DIRECT),
        // reader, and writer kernels plus their CBs.  Workload-scoped init /
        // cross-device GlobalSemaphores (cache miss only -- in persistent
        // mode the caller-provided semaphore on operation_attributes is used
        // instead) are parked on workload_descriptor.semaphores.  Tensor base
        // addresses are bound via emplace_runtime_args(Buffer*) so the
        // framework patches them on every dispatch.
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<AllToAllDispatchMetadataSparse>;

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
};
}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {
ttnn::operations::experimental::ccl::AllToAllDispatchMetadataDeviceOperation::tensor_return_value_t
all_to_all_dispatch_metadata(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const std::optional<std::vector<uint32_t>>& shared_expert_ids,
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
