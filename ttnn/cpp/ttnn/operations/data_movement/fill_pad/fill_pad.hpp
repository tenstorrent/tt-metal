// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {
// fill_value_is_packed_bits: when true, fill_value carries the raw int32 bit pattern (for int32 fill
// values that are not float-representable, e.g. reduce min/max pad sentinels). Default false decodes
// int32 fill_value numerically.
Tensor fill_implicit_tile_padding(
    const Tensor& input_tensor,
    float fill_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool fill_value_is_packed_bits = false);
}  // namespace ttnn
