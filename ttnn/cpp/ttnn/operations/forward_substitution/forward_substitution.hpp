// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor forward_substitution(const Tensor& input, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
