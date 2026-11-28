// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "block_keys_device_operation_types.hpp"

namespace ttnn::operations::block_keys {

// Python _program: ncores = max(1, min(gx * gy, rows)) cores in row-major order (core i at (i % gx, i / gx)),
// merged into rectangles; one uint8 CB of `scratch` bytes on them; one kernel (ReaderConfigDescriptor) with the
// common args pos / ids / out addresses (as Buffer*), per = rows / ncores, extra = rows % ncores.
// scratch = 64 + round_up(4 KP, 64) + round_up(4 K, 64).
struct BlockKeysProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BlockKeysParams& attrs, const BlockKeysInputs& t, Tensor& output);

    // Writes the three address slots (common_arg::POS_ADDR / IDS_ADDR / OUT_ADDR); PER / EXTRA and every
    // compile-time arg are pinned by the key (shapes, block, nbw, grid).
    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const BlockKeysParams& attrs,
        const BlockKeysInputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::block_keys
