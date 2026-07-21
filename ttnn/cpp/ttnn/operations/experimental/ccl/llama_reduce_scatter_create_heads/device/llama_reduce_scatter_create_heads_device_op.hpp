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
#include "ttnn/operations/ccl/ccl_common.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/workload_descriptor.hpp>

namespace ttnn::operations::experimental::ccl {

struct LlamaReduceScatterCreateHeadsDeviceOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const std::optional<GlobalSemaphore> cross_device_semaphore;
        const std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
        const uint32_t cluster_axis;
        const std::optional<MemoryConfig> output_mem_config;
        const uint32_t ring_devices;
        const ttnn::ccl::Topology topology;
        const uint32_t num_links;
        const uint32_t num_heads;
        const uint32_t num_kv_heads;
        const uint32_t head_dim;
        const uint32_t slice_size;
        const std::optional<MemoryConfig> qkv_memory_config;
        bool use_noc1_only;
        bool use_optimal_ccl_for_llama;
    };
    struct tensor_args_t {
        const Tensor input_tensor;
        Tensor intermediate_packet_buffer;
    };

    using spec_return_value_t = std::vector<ttnn::TensorSpec>;

    using tensor_return_value_t = std::vector<ttnn::Tensor>;

    struct LlamaReduceScatterCreateHeads {
        // Contract (2): declarative WorkloadDescriptor.  Builds one
        // ProgramDescriptor per coord.  The cross-device GlobalSemaphore lives
        // on operation_attributes (caller-allocated) so the factory needs no
        // workload-scoped resources.  Dynamic CBs that point at the input
        // tensor and intermediate packet buffer are wired up via
        // CBDescriptor::buffer so the framework patches their addresses on
        // every dispatch.  Q/K/V output buffer base addresses are wired up
        // via Buffer* runtime args.
        static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);
    };

    using program_factory_t = std::variant<LlamaReduceScatterCreateHeads>;

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

    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {

ttnn::operations::experimental::ccl::LlamaReduceScatterCreateHeadsDeviceOperation::tensor_return_value_t
llama_reduce_scatter_create_heads(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& intermediate_packet_buffer,
    int32_t dim,
    const GlobalSemaphore& semaphore,
    tt::tt_metal::SubDeviceId subdevice_id,
    uint32_t cluster_axis,
    uint32_t ring_devices,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t slice_size,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::MemoryConfig>& qkv_memory_config = std::nullopt,
    bool use_noc1_only = false,
    bool use_optimal_ccl_for_llama = false);

}  // namespace ttnn::prim
