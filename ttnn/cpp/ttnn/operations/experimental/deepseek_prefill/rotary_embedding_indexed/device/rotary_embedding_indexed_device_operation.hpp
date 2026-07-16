// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <variant>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed {

struct RotaryEmbeddingIndexedDeviceOperation {
    struct operation_attributes_t {
        uint32_t cluster_axis;  // mesh axis the cos/sin caches are SP-sharded along.
        // Prior valid global KV length in tokens. A per-call scalar that is intentionally NOT hashed:
        // it lives in a common runtime arg and is patched on cache hits by
        // MeshWorkloadFactory::override_runtime_arguments, so one cached program is reused across
        // chunks while the value stays current.
        uint32_t kv_actual_global;  // TODO: move to metadata
        MemoryConfig output_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;

        static constexpr auto attribute_names = std::forward_as_tuple("cluster_axis", "compute_kernel_config");
        auto attribute_values() const { return std::forward_as_tuple(cluster_axis, compute_kernel_config); }
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& cos;
        const Tensor& sin;
        const Tensor& trans_mat;
    };

    using spec_return_value_t = TensorSpec;
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
        using operation_attributes_t = RotaryEmbeddingIndexedDeviceOperation::operation_attributes_t;
        using tensor_args_t = RotaryEmbeddingIndexedDeviceOperation::tensor_args_t;
        using spec_return_value_t = RotaryEmbeddingIndexedDeviceOperation::spec_return_value_t;
        using tensor_return_value_t = RotaryEmbeddingIndexedDeviceOperation::tensor_return_value_t;
    };

    // Wraps the ProgramDescriptor factory so the default adapter patches buffer bindings on cache
    // hits, and override_runtime_arguments additionally patches the per-call kv_actual_global scalar
    // (a common runtime arg) -- the value the buffer-binding fast path would otherwise leave stale.
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
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed

namespace ttnn::prim {

ttnn::Tensor rotary_embedding_indexed(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t kv_actual_global,
    uint32_t cluster_axis,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim
