// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "indexer_score_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/mesh_device_operation_adapter.hpp"
#include "ttnn/distributed/types.hpp"  // MeshCoordinate(RangeSet)

namespace ttnn::operations::experimental::indexer_score::program {

// Ring-fused indexer_score factory (descriptor model). Selected only when operation_attributes.fused_ring is
// set; the classic IndexerScoreProgramFactory stays byte-identical for all unfused usage. Mirrors
// RingJointSDPAMeshWorkloadFactory: one ProgramDescriptor per coordinate co-scheduling the indexer compute +
// the ring_attention all-gather (the only Linear+fuse-capable AG) into ONE program, with a producer->consumer
// semaphore handshake so the reader coarse-barriers on the gather before scoring. DSA-only (relu, num_groups
// == 1, no block-pool): the fused path serves the lightning indexer's all-gather-then-score flow.
namespace detail {
struct IndexerScoreFusedDescriptorAdapterOperation {
    using operation_attributes_t = indexer_score::operation_attributes_t;
    using tensor_args_t = indexer_score::tensor_args_t;
    using spec_return_value_t = indexer_score::spec_return_value_t;
    using tensor_return_value_t = indexer_score::tensor_return_value_t;
};
}  // namespace detail

struct IndexerScoreFusedProgramFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);
};

struct IndexerScoreFusedMeshWorkloadFactory {
    using descriptor_adapter_t =
        ttnn::device_operation::MeshDeviceOperationAdapter<detail::IndexerScoreFusedDescriptorAdapterOperation>::
            DescriptorMeshWorkloadAdapter<IndexerScoreFusedProgramFactory>;
    using cached_mesh_workload_t = typename descriptor_adapter_t::cached_mesh_workload_t;

    static cached_mesh_workload_t create_mesh_workload(
        const operation_attributes_t& args,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const tensor_args_t& tensors,
        tensor_return_value_t& out);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached,
        const operation_attributes_t& args,
        const tensor_args_t& tensors,
        tensor_return_value_t& out);
};

static_assert(ttnn::device_operation::MeshWorkloadFactoryConcept<IndexerScoreFusedMeshWorkloadFactory>);

}  // namespace ttnn::operations::experimental::indexer_score::program
