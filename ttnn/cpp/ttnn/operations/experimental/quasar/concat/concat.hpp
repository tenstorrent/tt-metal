// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::quasar {

// Quasar (Metal 2.0) port of ttnn::concat. Same signature; `groups` must be 1 (grouped concat
// exists only in the original's L1 height-sharded zero-copy factory, which is not ported).
ttnn::Tensor concat(
    const std::vector<ttnn::Tensor>& input_tensors,
    int dim,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
    unsigned int groups = 1,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar
