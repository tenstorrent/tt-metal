// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor untilize_with_unpadding(
    const Tensor& input_tensor,
    const Shape& output_tensor_end,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool use_multicore = true,
    bool use_pack_untilize = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn
