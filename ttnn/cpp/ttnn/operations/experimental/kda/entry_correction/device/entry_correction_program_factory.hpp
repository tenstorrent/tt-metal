// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/workload_descriptor.hpp>

#include "ttnn/distributed/types.hpp"

#include "entry_correction_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct EntryCorrectionProgramFactory {
    // One program per coordinate: the chronology depends on the sequence-parallel rank.
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const EntryCorrectionParams&,
        const EntryCorrectionInputs&,
        std::vector<Tensor>&,
        const ttnn::MeshCoordinateRangeSet&);
};

}  // namespace ttnn::experimental::prim
