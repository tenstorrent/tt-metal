// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
