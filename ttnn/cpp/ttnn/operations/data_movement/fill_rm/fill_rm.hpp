// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/fill_rm_device_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {

// Expose prim::fill_rm directly
using prim::fill_rm;

namespace operations::data_movement {

// Convenience wrapper that hardcodes val_hi=1.0f, val_lo=0.0f
struct FillOnesRMOperation {
    static ttnn::Tensor invoke(
        uint32_t N,
        uint32_t C,
        uint32_t H,
        uint32_t W,
        uint32_t hFill,
        uint32_t wFill,
        const ttnn::Tensor& any,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto fill_ones_rm =
    ttnn::register_operation<"ttnn::fill_ones_rm", ttnn::operations::data_movement::FillOnesRMOperation>();

}  // namespace ttnn
