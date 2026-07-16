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
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache {

struct UpdatePaddedKvCacheDeviceOperation {
    struct operation_attributes_t {
        // Cache slot is linearized as users-outer, layers-inner:
        //   batch_idx = slot_idx * num_layers + layer_idx
        // layer_idx is hashed (structural): it takes only num_layers distinct values, so one cached
        // program per layer is reused across users and chunks. slot_idx and kv_actual_global are
        // per-call scalars held in common runtime args and patched on cache hits by
        // MeshWorkloadFactory::override_runtime_arguments, so they stay out of the program hash.
        uint32_t slot_idx;          // TODO: move to metadata
        uint32_t kv_actual_global;  // TODO: move to metadata
        uint32_t layer_idx;
        uint32_t num_layers;
        uint32_t cluster_axis;
        DataType input_dtype;
        Layout input_layout;
        MemoryConfig input_memory_config;
        Shape input_padded_shape;
        MemoryConfig cache_memory_config;
        Shape cache_padded_shape;

        static constexpr auto attribute_names = std::forward_as_tuple(
            "layer_idx",
            "num_layers",
            "cluster_axis",
            "input_dtype",
            "input_layout",
            "input_memory_config",
            "input_padded_shape",
            "cache_memory_config",
            "cache_padded_shape");
        auto attribute_values() const {
            return std::forward_as_tuple(
                layer_idx,
                num_layers,
                cluster_axis,
                input_dtype,
                input_layout,
                std::cref(input_memory_config),
                input_padded_shape,
                std::cref(cache_memory_config),
                cache_padded_shape);
        }
    };

    struct tensor_args_t {
        const Tensor& cache;
        const Tensor& input;
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
        using operation_attributes_t = UpdatePaddedKvCacheDeviceOperation::operation_attributes_t;
        using tensor_args_t = UpdatePaddedKvCacheDeviceOperation::tensor_args_t;
        using spec_return_value_t = UpdatePaddedKvCacheDeviceOperation::spec_return_value_t;
        using tensor_return_value_t = UpdatePaddedKvCacheDeviceOperation::tensor_return_value_t;
    };

    // Wraps the ProgramDescriptor factory so the default adapter patches buffer bindings on cache
    // hits, and override_runtime_arguments additionally patches the per-call slot_idx/kv_actual_global
    // scalars (common runtime args) -- the values the buffer-binding fast path would leave stale.
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor update_padded_kv_cache(
    const ttnn::Tensor& cache,
    const ttnn::Tensor& input,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t kv_actual_global,
    uint32_t cluster_axis);

}  // namespace ttnn::prim
