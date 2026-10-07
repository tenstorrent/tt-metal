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

enum class PadRoute { Auto, Native, Codegen };

// Shared body of every entry; recursive paths re-enter here with the same route, never through ttnn::pad.
ttnn::Tensor pad_with_route(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids,
    PadRoute route);

// The native implementation, with no codegen routing.
ttnn::Tensor pad_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids);

// Always native.
ttnn::Tensor pad_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids);

// Always codegen; fails instead of falling back and ignores the perf demotions.
ttnn::Tensor pad_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::operations::data_movement::detail
