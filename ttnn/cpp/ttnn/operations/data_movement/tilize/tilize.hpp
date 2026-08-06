// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>
#include "ttnn/types.hpp"
#include <tt-metalium/tile.hpp>

namespace ttnn {

// `implementation` selects the low-level primitive: "auto" (default) picks codegen for in-scope,
// non-demoted inputs and native otherwise; "native"/"codegen" force the respective prim (a forced
// "codegen" call TT_FATALs if the inputs are out of the codegen prim's scope).
ttnn::Tensor tilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt,
    bool use_multicore = true,
    bool use_low_perf = false,
    tt::tt_metal::Tile tile = tt::tt_metal::Tile(),
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::string& implementation = "auto");

}  // namespace ttnn
