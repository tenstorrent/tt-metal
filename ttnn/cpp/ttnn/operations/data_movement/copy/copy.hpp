// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {

Tensor copy(
    const Tensor& src_tensor,
    const Tensor& dst_tensor,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

Tensor assign(
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    std::optional<const DataType> output_dtype = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor assign(const Tensor& input_a, const Tensor& input_b);

}  // namespace ttnn
