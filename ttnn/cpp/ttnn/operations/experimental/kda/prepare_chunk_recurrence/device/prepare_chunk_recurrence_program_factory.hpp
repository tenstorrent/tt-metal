// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "prepare_chunk_recurrence_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

struct PrepareChunkRecurrenceProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PrepareChunkRecurrenceParams&, const PrepareChunkRecurrenceInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
