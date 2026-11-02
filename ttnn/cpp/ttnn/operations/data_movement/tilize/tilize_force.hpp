// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/tile.hpp>

#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which tilize implementation runs.
//
// `ttnn.tilize` chooses between the native and codegen programs on its own and takes no argument to
// override that, so the choice is not observable through it. These make it checkable: they are bound
// only under `ttnn._ttnn.operations.data_movement`, not registered into the `ttnn.*` namespace, and
// this header is kept out of the installed `api` file set.
//
// Prefer `ttnn::tilize` everywhere else, including for native-only cases: it already declines to route
// sharded, off-default-tile or execution-control-constrained cases to codegen.

// The native implementation, unconditionally.
ttnn::Tensor tilize_force_native(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt,
    bool use_multicore = true,
    bool use_low_perf = false,
    tt::tt_metal::Tile tile = tt::tt_metal::Tile(),
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// The codegen implementation, unconditionally. Throws for a case outside the codegen support scope
// instead of falling back to native, so a comparison against native cannot end up measuring native
// twice and reporting it as agreement.
ttnn::Tensor tilize_force_codegen(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt);

}  // namespace ttnn::operations::data_movement::detail
