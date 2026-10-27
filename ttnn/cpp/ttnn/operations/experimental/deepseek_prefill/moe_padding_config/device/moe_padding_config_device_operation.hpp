// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config {

struct MoePaddingConfigDeviceOperation {
    struct operation_attributes_t {
        // Tokens this chip carries per chunk (the MoE's sp_dim). Structural: fixed for the workload,
        // and the rotation math is expressed in it, so it is hashed.
        uint32_t tokens_per_chip;
        // 0 = right, 1 = left. Structural (rotated chunked prefill is right-padded by construction).
        uint32_t pad_side;
        // Mesh axis the config is sharded along (the SP axis). Structural.
        uint32_t cluster_axis;
    };

    struct tensor_args_t {
        // In-place output: the per-device [.., 2] UINT32 ROW_MAJOR config row the consumers read.
        // Caller-owned and persistent, so its address is stable across trace replays.
        const Tensor& config;
        // Two 1-element uint32 DRAM tensors ([1,1,1,1], ROW_MAJOR, replicated across the mesh):
        // absolute KV position of this chunk's first real token, and one past its last. The kernel
        // reads element [0] of each on-device, keeping the per-chunk values off the host dispatch
        // path (trace-safe).
        const Tensor& actual_start;
        const Tensor& actual_end;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& args,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    // Minimal operation-shaped helper so the descriptor factory can be adapted into a mesh workload.
    struct DescriptorAdapterOperation {
        using operation_attributes_t = MoePaddingConfigDeviceOperation::operation_attributes_t;
        using tensor_args_t = MoePaddingConfigDeviceOperation::tensor_args_t;
        using spec_return_value_t = MoePaddingConfigDeviceOperation::spec_return_value_t;
        using tensor_return_value_t = MoePaddingConfigDeviceOperation::tensor_return_value_t;
    };

    // Wraps the ProgramDescriptor factory so the default adapter patches buffer bindings on cache
    // hits, and override_runtime_arguments additionally refreshes the two metadata tensors' raw DRAM
    // addresses (common runtime args, which the buffer-binding fast path would leave stale).
    struct MeshWorkloadFactory {
        using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
            DescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<ProgramFactory>;
        using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& args,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& args,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output);
    };

    using program_factory_t = std::variant<MeshWorkloadFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_padding_config

namespace ttnn::prim {

ttnn::Tensor moe_padding_config(
    const ttnn::Tensor& config,
    const ttnn::Tensor& actual_start,
    const ttnn::Tensor& actual_end,
    uint32_t tokens_per_chip,
    uint32_t pad_side,
    uint32_t cluster_axis);

}  // namespace ttnn::prim
