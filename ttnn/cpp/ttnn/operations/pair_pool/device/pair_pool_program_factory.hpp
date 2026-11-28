// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/distributed/types.hpp"
#include "pair_pool_device_operation_types.hpp"

namespace ttnn::operations::pair_pool {

// One program for the whole mesh range (no mesh coordinate in create_descriptor): every chip runs the same one.
// create_descriptor plans (units = rows / 64 * HT, ncores = max(1, min(units, gx * gy)), cores row-wise
// i -> (i % gx, i / gx), merged rectangles; per, extra = divmod(units, ncores)). override_runtime_arguments writes
// only the two common address slots (reader SRC_ADDR, writer OUT_ADDR); PER / EXTRA and everything else are
// pinned by the key.
struct PairPoolProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PairPoolParams& attrs, const PairPoolInputs& t, Tensor& output);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const PairPoolParams& attrs,
        const PairPoolInputs& t,
        Tensor& output,
        const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
};

}  // namespace ttnn::operations::pair_pool
