// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <string>

#include "ttnn/types.hpp"

namespace ttnn {

// `implementation` selects the prim used for each repeated dim: "auto" (default) defers to
// supported_by_codegen(); "native" / "codegen" force a side (forcing "codegen" on an ineligible
// input TT_FATALs rather than silently falling back — see repeat/device/repeat_codegen_supported.hpp).
ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<std::string>& implementation = std::nullopt);

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& repeat_dims,
    const std::optional<std::string>& implementation = std::nullopt);

}  // namespace ttnn
