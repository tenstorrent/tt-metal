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

namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache {

struct ZeroPaddedKvCacheDeviceOperation {
    struct operation_attributes_t {
        // Zeroes the migration pad window [valid_global, ceil_pad_align(valid_global)) of the KV cache
        // (the garbage tail the full-tile fill/update left after the last real token), so the decode
        // side reads clean zeros. The window is up to pad_align-1 tokens and may straddle a chip
        // boundary in the block-cyclic layout; each chip zeroes its own share, derived on-device from
        // valid_global + chunk_size_global + the device's coordinate along cluster_axis.
        //
        // Cache slot is linearized as users-outer, layers-inner: batch_idx = slot_idx*num_layers + layer_idx.
        // layer_idx is hashed (structural): one cached program per layer, reused across users/chunks.
        // slot_idx and valid_global are per-call scalars held in common runtime args and patched on
        // cache hits by override_runtime_arguments, so they stay out of the program hash.
        uint32_t slot_idx;           // per-call (patched, not hashed)
        uint32_t valid_global;       // per-call (patched, not hashed): # real tokens; window starts here
        uint32_t chunk_size_global;  // structural: block-cyclic chunk size (= sp_factor * chunk_local)
        uint32_t pad_align;          // structural: migration read alignment (128); window end = ceil
        uint32_t layer_idx;          // hashed (structural)
        uint32_t num_layers;         // hashed (structural)
        uint32_t cluster_axis;       // hashed (structural): which mesh dim is sequence-parallel
    };

    struct tensor_args_t {
        // In-place: the op reads the boundary tile from the cache, masks it, and writes it back; the
        // full pad tiles are written from the L1 zeros buffer. No separate input tensor.
        const Tensor& cache;
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
        using operation_attributes_t = ZeroPaddedKvCacheDeviceOperation::operation_attributes_t;
        using tensor_args_t = ZeroPaddedKvCacheDeviceOperation::tensor_args_t;
        using spec_return_value_t = ZeroPaddedKvCacheDeviceOperation::spec_return_value_t;
        using tensor_return_value_t = ZeroPaddedKvCacheDeviceOperation::tensor_return_value_t;
    };

    // Wraps the ProgramDescriptor factory so the default adapter patches buffer bindings on cache
    // hits, and override_runtime_arguments additionally patches the per-call slot_idx/valid_global
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
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache

namespace ttnn::prim {

ttnn::Tensor zero_padded_kv_cache(
    const ttnn::Tensor& cache,
    uint32_t slot_idx,
    uint32_t layer_idx,
    uint32_t num_layers,
    uint32_t valid_global,
    uint32_t chunk_size_global,
    uint32_t cluster_axis,
    uint32_t pad_align);

}  // namespace ttnn::prim
