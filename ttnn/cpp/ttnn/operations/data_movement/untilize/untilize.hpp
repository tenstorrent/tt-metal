// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/types.hpp"

namespace ttnn {

// Dispatches to the codegen prim when it supports the case and is expected to win, and to the
// native prim otherwise; see codegen/untilize_codegen_supported.hpp for the routing predicates.
// The choice is internal and not caller-selectable.
//
// gathered_prefix_tensor (optional, trace-safe): a 1-element uint32 DRAM tensor holding a chunk start. Only the
// gathered prefix of a [1, 1, H, W] input is untilized: g = min(round_up(start + gathered_slab_global,
// gathered_slab_global), H) rows split over num_segments equal row segments (rank-major, as high_bw_all_gather writes
// them), the first g / num_segments rows of each; the rest of the output is left as it is. Always the codegen prim.
ttnn::Tensor untilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool use_multicore = true,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    const std::optional<ttnn::Tensor>& gathered_prefix_tensor = std::nullopt,
    uint32_t gathered_slab_global = 0,
    uint32_t num_segments = 1);

}  // namespace ttnn
