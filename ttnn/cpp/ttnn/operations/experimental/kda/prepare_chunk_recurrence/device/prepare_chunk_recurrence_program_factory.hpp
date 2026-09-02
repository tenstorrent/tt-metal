// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "prepare_chunk_recurrence_device_operation_types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::experimental::prim {

uint32_t prepare_chunk_recurrence_cb_size_bytes(
    uint32_t chunk_size,
    uint32_t key_dim,
    uint32_t value_dim,
    tt::tt_metal::DataType gate_dtype,
    uint32_t output_bf16_mask);

struct PrepareChunkRecurrenceProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const PrepareChunkRecurrenceParams&, const PrepareChunkRecurrenceInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::experimental::prim
