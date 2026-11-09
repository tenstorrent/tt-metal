// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::reshape_new {

// The new ttnn::reshape (TT_METAL_EXECUTION_POLICY_RESHAPE=new). Same contract as the legacy implementation:
//   - one dim of `logical_shape` may be -1 (inferred); `padded_shape` equal to `logical_shape` means "default";
//   - `memory_config` defaults to the input's, with the shard spec re-derived for the new shape when the input's
//     own spec cannot describe it;
//   - `pad_value` (default 0) is written into the output's tile padding unless `skip_padding_fill` is set
//     (block-float outputs are always filled: their shared exponents cover the padding);
//   - work runs on `sub_core_grid` when given.
// The output is a zero-copy view of the input whenever its page sequence and distribution are unchanged; otherwise
// a fresh tensor written by one device program (two typecasts around it for block-float data). Mesh placements are
// relabeled to the output dims (#64322).
Tensor reshape(
    const Tensor& input,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<PadValue>& pad_value,
    const std::optional<CoreRangeSet>& sub_core_grid,
    bool skip_padding_fill);

}  // namespace ttnn::operations::data_movement::reshape_new
