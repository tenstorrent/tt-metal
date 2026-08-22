// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/types.hpp"

#include <ranges>
#include <string>
#include "ttnn/types.hpp"

namespace ttnn {

// `implementation`: "auto" (default) picks ConcatCodegen when the codegen prim
// supports the call (row-major, in-scope dtype, interleaved, no sub_core_grids
// override, groups == 1) and it isn't perf-demoted, else native; "native" and
// "codegen" force the respective prim ("codegen" TT_FATALs if unsupported).
ttnn::Tensor concat(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
    unsigned int groups = 1,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::string& implementation = "auto");

}  // namespace ttnn
