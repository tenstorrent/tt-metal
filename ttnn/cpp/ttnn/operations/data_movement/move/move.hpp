// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {

Tensor move(const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config = std::nullopt);

}  // namespace ttnn
