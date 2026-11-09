// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which reshape implementation runs, so bit-exactness and device time
// of each can be checked. Bound only under `ttnn._ttnn.operations.data_movement`; use `ttnn::reshape` elsewhere.

// The native implementation, unconditionally. Input normalization is identical to `ttnn::reshape`.
ttnn::Tensor reshape_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int32_t>& shape,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    bool skip_padding_fill = false);

// The codegen implementation, unconditionally. Throws outside its scope instead of falling back, so a comparison
// against native cannot measure native twice. The no-op, view and same-shape shortcuts are not applied.
// `reshape_map_mode` and `skip_padding_fill` are ignored: the map mode only governs native's mapping tensor, and
// codegen always zeroes tile padding.
ttnn::Tensor reshape_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int32_t>& shape,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    bool skip_padding_fill = false);

// The two implementations behind the TT_METAL_EXECUTION_POLICY_RESHAPE switch, unconditionally and with the
// full `ttnn::reshape` argument list. Each pins the policy for the calls nested inside it, so a reshape that an
// implementation issues internally runs on that same implementation. `reshape_map_mode` only affects legacy.
ttnn::Tensor reshape_legacy(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    bool skip_padding_fill = false);

ttnn::Tensor reshape_new(
    const ttnn::Tensor& input_tensor,
    const ttnn::Shape& logical_shape,
    const ttnn::Shape& padded_shape,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt,
    bool skip_padding_fill = false);

}  // namespace ttnn::operations::data_movement::detail
