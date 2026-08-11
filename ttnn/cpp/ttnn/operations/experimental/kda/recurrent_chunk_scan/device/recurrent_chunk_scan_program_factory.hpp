// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "recurrent_chunk_scan_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct RecurrentChunkScanProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RecurrentChunkScanParams&, const RecurrentChunkScanInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
