// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>

#include "ttnn/types.hpp"

namespace ttnn {

// # This operation does not support the following cases:
// #   - Shape([2[32], 2[32]]) -> repeats = 2, dim = 0
// #   - Shape([2[32], 2[32]]) -> repeats = Tensor[1,2], dim = 1

// `implementation` selects "auto" (default; codegen when supported and not perf-demoted, else
// native), "native" (always the composite host implementation below), or "codegen" (always
// prim::repeat_interleave_codegen; TT_FATALs if the input is not codegen-supported).
ttnn::Tensor repeat_interleave(
    const ttnn::Tensor& input_a,
    uint32_t repeats,
    int32_t dim,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::string& implementation = "auto");

}  // namespace ttnn
