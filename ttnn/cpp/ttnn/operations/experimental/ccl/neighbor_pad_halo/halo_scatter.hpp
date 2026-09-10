// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Local (no-fabric) repack for the persistent-padded activation pipeline
ttnn::Tensor halo_scatter(
    const ttnn::Tensor& compact_buffer,
    const ttnn::Tensor& interior_src,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool border_only = false);

}  // namespace ttnn::experimental
