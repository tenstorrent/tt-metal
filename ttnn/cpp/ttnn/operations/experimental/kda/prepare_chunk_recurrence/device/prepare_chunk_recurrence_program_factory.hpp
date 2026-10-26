// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "prepare_chunk_recurrence_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

// The batched inverse holds one FP32 tile per work item in akk, t_inv, tile_workspace_0 and tile_workspace_2, and
// two in tile_workspace_1 (its Horner total is double depth).
inline constexpr uint32_t prepare_chunk_recurrence_inverse_tiles_per_item = 6;

// Work items per stage batch: compute runs every per-item stage over a batch before the next stage, so the batch's
// destination-register acquires pipeline unpack, math and pack and each stage's fixed costs are paid once per batch.
// The per-item buffers hold one batch; two items fit L1 beside the inverse's max_items buffers at the production shape
// (prepare_chunk_recurrence_l1_bytes).
inline constexpr uint32_t prepare_chunk_recurrence_batch_items = 2;

// The L1 the program's dataflow buffers take on the core with the most work items; validation checks it against the
// L1 free per core. Kept in step with the factory's buffer list.
uint64_t prepare_chunk_recurrence_l1_bytes(const PrepareChunkRecurrenceParams& attrs, uint32_t max_items);

struct PrepareChunkRecurrenceProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const PrepareChunkRecurrenceParams&,
        const PrepareChunkRecurrenceInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
