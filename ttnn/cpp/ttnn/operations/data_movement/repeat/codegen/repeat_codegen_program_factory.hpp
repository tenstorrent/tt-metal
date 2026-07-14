// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

struct RepeatCodegenParams {
    uint32_t rep_dim{};
    uint32_t num_repeats{};
    uint32_t lower_pages{};
    uint32_t rep_dim_pages{};
    uint32_t total_out_pages{};
    // RM only; unused on the TILE branch (tile size is fixed by dtype).
    uint32_t stick_size{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct RepeatCodegenInputs {
    Tensor input;
};

struct RepeatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RepeatCodegenParams& operation_attributes,
        const RepeatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
