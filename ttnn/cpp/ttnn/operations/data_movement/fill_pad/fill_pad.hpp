// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {
// fill_value: PadValue whose uint32_t arm carries a raw int32 bit pattern (for int32 fill values that
// are not float-representable, e.g. reduce min/max pad sentinels) and whose float arm carries a numeric
// value (the default).
Tensor fill_implicit_tile_padding(
    const Tensor& input_tensor,
    tt::tt_metal::PadValue fill_value,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);
}  // namespace ttnn
