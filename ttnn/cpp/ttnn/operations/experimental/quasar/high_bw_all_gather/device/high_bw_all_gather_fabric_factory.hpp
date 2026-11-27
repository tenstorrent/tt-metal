// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "high_bw_all_gather_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"

#include <tt-metalium/memory/cross_program/global_semaphore.hpp>
#include <tt-metalium/workload/legacy/program_descriptors.hpp>
#include <tt-metalium/workload/legacy/workload_descriptor.hpp>

namespace ttnn::operations::experimental::quasar::high_bw_all_gather {

// Chip shard layout of one cached program (terms: kernels/fabric/fabric_all_gather_chunk_walk.hpp). The cache slot's
// first page and the valid pages per outer slice are runtime values derived from it on every dispatch (host scalars) or
// on device (metadata tensors).
struct ChipShardPageGeometry {
    uint32_t num_outer_slices = 0;
    uint32_t pages_per_outer_slice = 0;
    uint32_t pages_per_cache_slot = 0;          // input pages of one cache slot (selected-batch gathers)
    uint32_t local_gather_dim_length = 0;       // this chip's (input) length of the gather dim
    uint32_t gather_dim_elements_per_page = 1;  // tile height / width when gathering a tile dim, else 1
    uint32_t num_ranks = 0;
    uint32_t pages_per_kv_slab = 0;  // valid pages per outer slice per KV slab (prefix metadata path)
};

// Builds the whole mesh workload once per program-cache miss: resolves the gather plan, allocates (or adopts the
// caller's) shards-arrived / downstream-started counters, parks them on the WorkloadDescriptor so they outlive the
// cached workload, and emits one ProgramDescriptor per chip. Tensor addresses are declared as Buffer* bindings.
struct HighBwAllGatherFabricFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const HighBwAllGatherParams& args,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

// Minimal operation-shaped helper so the descriptor factory can be adapted into a mesh workload.
struct HighBwAllGatherDescriptorAdapterOperation {
    using operation_attributes_t = HighBwAllGatherParams;
    using tensor_args_t = HighBwAllGatherInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
};

// Wraps the WorkloadDescriptor factory so a cache hit first patches the Buffer* bindings exactly as the default
// adapter does (input, output and the optional metadata tensors), then re-applies the common runtime args the bindings
// cannot cover: the hash-excluded slot / prefix scalars (input_batch_index, gathered_dim_size, batch_slot_num_layers,
// batch_slot_layer_idx) and, for caller-owned semaphores, their current addresses. It is workload-scoped rather than a
// per-Program override_runtime_arguments hook because a semaphore swap must also replace the GlobalSemaphores parked in
// each program's shared variables (keeping them alive with the cached workload), which a per-Program hook cannot
// reach. It also writes the tensor addresses itself when the cache-filling call aliased two tensor slots and the
// adapter therefore recorded no bindings.
struct HighBwAllGatherMeshWorkloadFactory {
    using descriptor_adapter_t = ttnn::device_operation::MeshDeviceOperationAdapter<
        HighBwAllGatherDescriptorAdapterOperation>::DescriptorMeshWorkloadAdapter<HighBwAllGatherFabricFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const HighBwAllGatherParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const HighBwAllGatherParams& args,
        const HighBwAllGatherInputs& tensor_args,
        Tensor& output_tensor);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<HighBwAllGatherMeshWorkloadFactory>);

}  // namespace ttnn::operations::experimental::quasar::high_bw_all_gather
