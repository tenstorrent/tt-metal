// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>

#include "ttnn/types.hpp"

namespace ttnn {

// # This operation does not support the following cases:
// #   - `repeats` as a per-element Tensor (e.g. Shape([2, 2]) -> repeats = Tensor[1,2], dim = 1).
// #     Only a single scalar `repeats` applied uniformly across the dim is supported.
// # (Small-shape cases such as Shape([2, 2]) -> repeats = 2, dim = 0 are supported.)

// `implementation` selects the dispatch path: "auto" (default) picks the codegen prim iff it is
// supported and not perf-demoted for these inputs, else the native (host-composed) path;
// "native" and "codegen" force one or the other ("codegen" TT_FATALs if unsupported).
ttnn::Tensor repeat_interleave(
    const ttnn::Tensor& input_a,
    uint32_t repeats,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::string& implementation = "auto");

}  // namespace ttnn
