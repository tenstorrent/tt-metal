// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <string>

#include "ttnn/types.hpp"

namespace ttnn {

// `implementation`: "auto" (default) picks codegen when the codegen prim
// supports the call and it isn't perf-demoted, else native; "native" and
// "codegen" force the respective prim ("codegen" TT_FATALs if unsupported).
ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::string& implementation = "auto");

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor, const ttnn::Shape& repeat_dims, const std::string& implementation = "auto");

}  // namespace ttnn
