// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "kda_chunk_preparation_device_operation_types.hpp"

namespace ttnn::prim {

uint32_t kda_chunk_preparation_cb_size_bytes(
    uint32_t chunk_size, uint32_t key_dim, uint32_t value_dim, DataType gate_dtype, uint32_t output_bf16_mask);

struct KdaChunkPreparationProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const KdaChunkPreparationParams&, const KdaChunkPreparationInputs&, std::vector<Tensor>&);
};

}  // namespace ttnn::prim
