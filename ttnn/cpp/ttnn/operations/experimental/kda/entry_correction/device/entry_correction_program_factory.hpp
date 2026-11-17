// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "entry_correction_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct EntryCorrectionProgramFactory {
    // One program per SP rank: the chronology depends on it.
    static ttnn::device_operation::MeshWorkloadArtifacts create_mesh_workload_artifacts(
        const EntryCorrectionParams&,
        const EntryCorrectionInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
