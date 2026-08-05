// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kda_final_scan_device_operation_types.hpp"

namespace ttnn::prim {

struct KdaFinalChunkScanProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaFinalChunkScanParams&, const KdaFinalChunkScanInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::prim
