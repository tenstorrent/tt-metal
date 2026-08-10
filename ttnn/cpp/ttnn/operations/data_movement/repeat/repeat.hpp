// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <optional>
#include <string>

#include "ttnn/types.hpp"

namespace ttnn {

// `implementation`: "auto"/"native"/"codegen"; auto picks codegen when supported and not demoted.
ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::string& implementation = "auto",
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& repeat_dims,
    const std::string& implementation = "auto",
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
