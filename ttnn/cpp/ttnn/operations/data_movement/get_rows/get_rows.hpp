// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn {

// Gathers rows of a [H, W] row-major tensor on device: output[i] = input[indices[i]], matching
// torch.index_select(input, 0, indices) for the supported (non-negative, in-range) index domain.
Tensor get_rows(
    const Tensor& input_tensor,
    const Tensor& indices_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
