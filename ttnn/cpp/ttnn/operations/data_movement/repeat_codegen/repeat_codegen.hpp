// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn {

// PoC: single-dim higher-dim repeat on a 4D TILE bfloat16 interleaved tensor.
ttnn::Tensor repeat_codegen(
    const ttnn::Tensor& input_tensor,
    uint32_t dim,
    uint32_t repetitions,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
