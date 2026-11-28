// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "block_max_bf16_device_operation_types.hpp"

namespace ttnn::operations::block_max_bf16 {

// Both factories: one scratch CB and kernels READER / WRITER (kernels/block_max_bf16_args.hpp) on the grid
// rectangle (0,0)..(gx-1, gy-1), gx, gy = the device grid clamped to max_grid (each at least 1).
// override_runtime_arguments rewrites the common args IN_ADDR, OUT_ADDR, LEN_ADDR and N of both kernels; the key
// pins ROWS and every compile-time arg.

// strip_major == false: fp32 [.., R, NI] -> bf16 [.., R, NI / factor], kernel block_max_bf16.cpp.
// Scratch: 64 + 4 * round_up(CAST_CHUNK * 4 + CAST_CHUNK / factor * 2 + 64, 64) bytes.
struct BlockMaxReduceProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BlockMaxBf16Params& attrs, const BlockMaxBf16Inputs& t, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const BlockMaxBf16Params& attrs,
        const BlockMaxBf16Inputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

// strip_major == true: bf16 [.., S NB / 8, 256] -> bf16 [.., 32 S, NB], kernel block_max_strip.cpp.
// Scratch: 64 + 2 * 4 * STRIP_CHUNK * 64 bytes.
struct BlockMaxStripProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BlockMaxBf16Params& attrs, const BlockMaxBf16Inputs& t, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const BlockMaxBf16Params& attrs,
        const BlockMaxBf16Inputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::block_max_bf16
