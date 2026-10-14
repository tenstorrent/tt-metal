// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which reshape implementation runs.
//
// `ttnn.reshape` chooses on its own and takes no argument to override that. These exist because the
// choice still has to be checkable: bit-exactness against the existing implementation over the
// supported scope and a device-time win are not observable through an entry point that silently
// picks one. They are bound only under `ttnn._ttnn.operations.data_movement`, are not registered
// into `ttnn.*`, and this header is kept out of the installed `api` file set.
//
// Prefer `ttnn::reshape` everywhere else: it already declines the cases the second entry rejects.

// The existing implementation, unconditionally. Input normalization is identical to `ttnn::reshape`.
ttnn::Tensor reshape_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int32_t>& shape,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    bool skip_padding_fill = false);

// The generated implementation, unconditionally. Throws for a case outside its support scope instead
// of falling back, so a comparison against the native path cannot silently measure native twice.
// Only shape inference and `shape_corrector` normalization are shared with `ttnn::reshape`; the no-op,
// view and same-shape shortcuts are not applied.
ttnn::Tensor reshape_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<int32_t>& shape,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<PadValue>& pad_value = std::nullopt,
    TileReshapeMapMode reshape_map_mode = TileReshapeMapMode::CACHE,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    bool skip_padding_fill = false);

}  // namespace ttnn::operations::data_movement::detail
