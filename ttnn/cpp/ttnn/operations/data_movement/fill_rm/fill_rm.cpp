// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor FillOnesRMOperation::invoke(
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t hFill,
    uint32_t wFill,
    const ttnn::Tensor& any,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    return ttnn::prim::fill_rm(N, C, H, W, hFill, wFill, any, 1.0f, 0.0f, memory_config);
}

}  // namespace ttnn::operations::data_movement
