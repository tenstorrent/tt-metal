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
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>
#include <vector>

namespace ttnn::operations::ccl {

namespace detail {

std::pair<std::array<uint32_t, 6>, std::array<uint32_t, 6>> get_cb_sizes(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& mapping_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis);

}  // namespace detail

struct AllToAllDispatchDeviceOperation {
    enum AllToAllTransferType {
        FullPacket,  // All pages are sent to the intermediate buffer and then written to the output buffer later
        PageByPage,  // Each page is sent directly to the output buffer to conserve L1 space via intermediates
    };
    struct operation_attributes_t {
        const CoreRangeSet worker_core_range_set;
        const MemoryConfig output_mem_config;
        const std::optional<uint32_t> axis;
        const uint32_t num_links;
        const tt::tt_fabric::Topology topology;
        const AllToAllTransferType impl;
        const uint32_t output_concat_dim;
        static constexpr auto attribute_names = std::forward_as_tuple(
            "worker_core_range_set", "output_mem_config", "axis", "num_links", "topology", "impl", "output_concat_dim");
        auto attribute_values() const {
            return std::forward_as_tuple(
                worker_core_range_set, output_mem_config, axis, num_links, topology, impl, output_concat_dim);
        };
    };
    struct tensor_args_t {
        const Tensor input_tensor;
        const Tensor expert_indices_tensor;
        const Tensor expert_mapping_tensor;
        const std::optional<std::array<Tensor, 2>> optional_output_tensors;
    };

    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 2>;

    using tensor_return_value_t = std::array<Tensor, 2>;

    struct AllToAllDispatchSparse {
        // Builds the entire workload in one call (cache miss):
        //   1. Allocates the two GlobalSemaphores used by the kernels (init / cross-device)
        //      and parks them on `WorkloadDescriptor::semaphores` so the framework keeps
        //      them alive for the cached workload's lifetime.
        //   2. Runs the cross-device Synchronize barrier once per workload.
        //   3. Loops `tensor_coords.coords()` and pushes a per-coord ProgramDescriptor
        //      into `programs`.  Each program depends on its mesh coordinate (fabric
        //      routing / DEST_CHIP_ID define / linearized mesh index), so descriptors
        //      cannot be shared across coords.
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<AllToAllDispatchSparse>;

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
}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::operations::ccl::AllToAllDispatchDeviceOperation::tensor_return_value_t all_to_all_dispatch(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis,
    const std::optional<std::array<ttnn::Tensor, 2>>& optional_output_tensors,
    uint32_t num_links,
    tt::tt_fabric::Topology topology,
    const ttnn::MemoryConfig& memory_config,
    const CoreRangeSet& worker_core_range_set,
    ttnn::operations::ccl::AllToAllDispatchDeviceOperation::AllToAllTransferType impl,
    uint32_t output_concat_dim);
}  // namespace ttnn::prim
