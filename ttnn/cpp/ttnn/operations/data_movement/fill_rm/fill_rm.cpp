// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm.hpp"
#include "device/fill_rm_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::data_movement {

ttnn::Tensor FillRMOperation::invoke(
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t hFill,
    uint32_t wFill,
    const ttnn::Tensor& any,
    float val_hi,
    float val_lo,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const auto output_memory_config = memory_config.value_or(any.memory_config());
    return ttnn::prim::fill_rm(N, C, H, W, hFill, wFill, any, val_hi, val_lo, output_memory_config);
}

ttnn::Tensor FillOnesRMOperation::invoke(
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t hFill,
    uint32_t wFill,
    const ttnn::Tensor& any,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    const auto output_memory_config = memory_config.value_or(any.memory_config());
    return ttnn::prim::fill_rm(N, C, H, W, hFill, wFill, any, 1.0f, 0.0f, output_memory_config);
}

}  // namespace ttnn::operations::data_movement
