// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn {

// PoC: TILE -> ROW_MAJOR untilize on a 4D bfloat16 interleaved tensor, interleaved
// tile-row path only. Output defaults to the input's memory config.
ttnn::Tensor untilize_codegen(
    const ttnn::Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
