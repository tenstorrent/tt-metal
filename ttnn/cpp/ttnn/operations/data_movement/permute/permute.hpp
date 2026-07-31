// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>
#include "ttnn/types.hpp"

namespace ttnn {

// `implementation` selects the dispatch path: "auto" (default) routes to the codegen prim iff
// PermuteCodegen's supported_by_codegen()/is_demoted() gates admit the call, else falls back to
// this native prim; "native" always uses this native prim; "codegen" always uses the codegen prim
// (TT_FATALs if unsupported). See codegen/permute_codegen_supported.hpp.
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
