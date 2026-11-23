// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

// Reads back the page-rows update_padded_kv_cache stored for one chunk (see read_padded_kv_cache.hpp).
struct ReadPaddedKvCacheDeviceOperation {
    struct operation_attributes_t {
        // Cache entries read: slot * num_layers + first_layer + l for l in [0, num_read_layers). All structural.
        uint32_t first_layer;
        uint32_t num_read_layers;
        uint32_t num_layers;
        // Tokens one chunk spans across the mesh; each chip holds chunk_size_global / (sp * tp) rows of it.
        uint32_t chunk_size_global;
        // The cache's linearization, as given to the update_padded_kv_cache call that wrote it.
        std::optional<uint32_t> cluster_axis;
        std::optional<uint32_t> tp_axis;
    };

    struct tensor_args_t {
        const Tensor& cache;
        // 1-element uint32 DRAM tensors ([1,1,1,1] ROW_MAJOR, replicated), read on device: the user slot and the
        // chunk's start in tokens (the kv_actual_global its write was given).
        const Tensor& slot_idx;
        const Tensor& kv_actual_global;
        // Preallocated destination (a persistent address for trace replay), or nullopt to allocate one.
        std::optional<Tensor> output;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& args,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate);
    };

    struct DescriptorAdapterOperation {
        using operation_attributes_t = ReadPaddedKvCacheDeviceOperation::operation_attributes_t;
        using tensor_args_t = ReadPaddedKvCacheDeviceOperation::tensor_args_t;
        using spec_return_value_t = ReadPaddedKvCacheDeviceOperation::spec_return_value_t;
        using tensor_return_value_t = ReadPaddedKvCacheDeviceOperation::tensor_return_value_t;
    };

    // As UpdatePaddedKvCacheDeviceOperation::MeshWorkloadFactory: one program for the whole mesh when a chip table
    // is available, and the metadata tensors' raw addresses patched on every cache hit.
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
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor read_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& slot_idx,
    const ttnn::Tensor& kv_actual_global,
    uint32_t first_layer,
    uint32_t num_read_layers,
    uint32_t num_layers,
    uint32_t chunk_size_global,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> tp_axis,
    const std::optional<ttnn::Tensor>& output);

}  // namespace ttnn::prim
