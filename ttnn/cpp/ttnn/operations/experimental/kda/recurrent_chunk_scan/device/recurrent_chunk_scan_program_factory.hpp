// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recurrent_chunk_scan_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct RecurrentChunkScanProgramFactory {
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const RecurrentChunkScanParams&,
        const RecurrentChunkScanInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
