// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn::experimental::deepseek {

// Gather a WIDTH_SHARDED input, or multicast a single-core HEIGHT_SHARDED input,
// replicating a full row-major M x K copy onto every matmul core. TILE inputs are
// untilized; ROW_MAJOR inputs skip untilize.
ttnn::Tensor all_gather_for_matmul(
    const ttnn::Tensor& input_tensor,
    const CoreRangeSet& output_core_range_set,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn::experimental::deepseek
