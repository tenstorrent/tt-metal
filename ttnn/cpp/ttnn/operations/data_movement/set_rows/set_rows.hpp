// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn {

// Writes rows of a [N, W] updates tensor into a [H, W] input on device: output preserves the input
// everywhere except output[indices[i]] = updates[i], matching torch.index_put_(input, (indices,),
// updates) for the supported (non-negative, in-range, unique-index) domain. Duplicate indices take
// the later update (index_copy_ order). Implemented on top of indexed_fill.
Tensor set_rows(
    const Tensor& input_tensor,
    const Tensor& indices_tensor,
    const Tensor& updates_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
