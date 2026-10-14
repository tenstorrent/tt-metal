// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only entry points that pin which pad implementation runs.
//
// `ttnn.pad` chooses on its own and takes no argument to override that. These exist so the choice stays checkable:
// tests and benchmarks compare the two paths for bit-exactness and device time, which an entry point that silently
// picks one cannot show. They are bound only under `ttnn._ttnn.operations.data_movement`, not registered into
// `ttnn.*`, and sit in `detail` so a caller reaching for one has to say so. Do not use them in production code.

// The native implementation, unconditionally.
ttnn::Tensor pad_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids);

// The codegen implementation. Raises for an input outside its support scope instead of falling back to native, so a
// comparison cannot end up measuring native twice, and ignores the perf demotions. A pad that adds nothing still
// returns the input, as ttnn.pad does.
ttnn::Tensor pad_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::detail
