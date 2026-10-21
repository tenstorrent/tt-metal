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
