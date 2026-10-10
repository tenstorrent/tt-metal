// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Staging-buffer depth (DFB entries) shared by all three program-factory branches
// (TILE-interleaved, RM last-dim, RM higher-dim). Also consulted by
// repeat_codegen_supported.cpp's L1-capacity gate, so this is the single source of
// truth rather than a value duplicated across files.
inline constexpr uint32_t kRepeatCbDepth = 8;

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
    std::optional<Tensor> optional_output_tensor;
};

struct RepeatCodegenProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const RepeatCodegenParams& operation_attributes,
        const RepeatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
