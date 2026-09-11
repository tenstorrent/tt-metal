// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// One Program per mesh coordinate (the device's ring position, neighbors and per-chain shard counts depend on
// it): the VSA streaming descriptor is materialized into a Program, then the multi-worker all-gather's
// fusable builder adds the sender cores that forward the flat K|V shard around the ring and signal
// the VSA leaders per landed shard.
struct VsaRingSdpaMeshWorkloadFactory {
    using shared_variables_t = ttnn::AllGatherProgramArtifacts;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const VsaRingSdpaParams& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);

    // Cache hit: the all-gather's own override re-applies its buffer and GlobalSemaphore addresses (excluded
    // from the program hash); the VSA patch re-applies the VSA kernels' addresses and the gathered address.
    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const VsaRingSdpaParams& args,
        const VsaRingSdpaInputs& tensor_args,
        Tensor& output);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<VsaRingSdpaMeshWorkloadFactory>);

}  // namespace ttnn::prim
