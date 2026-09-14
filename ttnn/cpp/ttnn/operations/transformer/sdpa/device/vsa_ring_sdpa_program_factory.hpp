// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_kv_gather.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// One Program per mesh coordinate (the device's ring position, neighbours and per-chain shard counts depend on
// it): the VSA streaming descriptor is materialized into a Program, then the op's own K/V gather
// (build_vsa_kv_gather) adds the sender cores that forward this device's K and V shards around the ring,
// token-major, and signal the VSA leaders per landed shard.
struct VsaRingSdpaMeshWorkloadFactory {
    using shared_variables_t = VsaKvGatherArtifacts;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const VsaRingSdpaParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);

    // Cache hit: the gather's override re-applies its buffer and GlobalSemaphore addresses (excluded from the
    // program hash); the VSA patch re-applies the VSA kernels' addresses and the gathered/semaphore addresses.
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const VsaRingSdpaParams& args,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<VsaRingSdpaMeshWorkloadFactory>);

}  // namespace ttnn::prim
