// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "indexer_score_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/distributed/types.hpp"  // MeshCoordinate(RangeSet)

namespace ttnn::operations::experimental::indexer_score::program {

struct IndexerScoreSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel;
    tt::tt_metal::KernelHandle compute_kernel;
    tt::tt_metal::KernelHandle writer_kernel;
    std::vector<CoreCoord> worker_cores;
    // This device's linearized SP-ring index (from its mesh coordinate); chunk_start = base + idx*stride.
    // Stored so override_runtime_arguments can recompute chunk_start for new base/stride on a cache hit.
    uint32_t device_index = 0;
    // This device's TP-rank along the TP axis (seq_shard_axes[1]; 0 when not 2D-sub-sharded); the 2D block-cyclic
    // geometry adds tp_index*Sq to the slab position. Stored for the same override recompute.
    uint32_t tp_index = 0;
};

// Native mesh-workload factory: one program per mesh coordinate so each device derives its own
// chunk_start from its coordinate (mirrors ring_joint_sdpa). A single device is a 1x1 mesh, so the
// single-chip path flows through the same code with one coordinate (index 0).
struct IndexerScoreProgramFactory {
    using shared_variables_t = IndexerScoreSharedVariables;
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

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

private:
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    // Build one device's program; chunk_start is derived from `coord`.
    static cached_program_t create_at(
        const operation_attributes_t& args,
        const ttnn::MeshCoordinate& coord,
        const tensor_args_t& tensors,
        tensor_return_value_t& out);
};

}  // namespace ttnn::operations::experimental::indexer_score::program
