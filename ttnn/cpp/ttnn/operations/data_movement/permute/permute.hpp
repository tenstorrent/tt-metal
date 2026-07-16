// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include "ttnn/types.hpp"

namespace ttnn {

// `implementation` selects the dispatch path: "auto" (default) routes ROW_MAJOR, DRAM-interleaved
// inputs eligible for the codegen port (see permute/codegen/permute_codegen_supported.hpp) to it
// unless perf-demoted, "native" always uses the existing device kernels, "codegen" forces the
// codegen prim and TT_FATALs if the input is out of its scope.
ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value = 0.0f,
    const std::string& implementation = "auto");

ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int64_t>& dims,
    float pad_value = 0.0f,
    const std::string& implementation = "auto");

}  // namespace ttnn
