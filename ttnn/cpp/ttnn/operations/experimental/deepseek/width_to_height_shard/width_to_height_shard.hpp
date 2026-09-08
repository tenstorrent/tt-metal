// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Untilize and gather a WIDTH_SHARDED input, then broadcast a full copy to every output core.
ttnn::Tensor width_to_height_shard(
    const ttnn::Tensor& input_tensor,
    const CoreRangeSet& output_core_range_set,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn::experimental::deepseek
