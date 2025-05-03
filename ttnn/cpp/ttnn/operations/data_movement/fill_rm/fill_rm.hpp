// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct FillRMOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        uint32_t N,
        uint32_t C,
        uint32_t H,
        uint32_t W,
        uint32_t hFill,
        uint32_t wFill,
        const ttnn::Tensor& any,
        float val_hi,
        float val_lo,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

struct FillOnesRMOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        uint32_t N,
        uint32_t C,
        uint32_t H,
        uint32_t W,
        uint32_t hFill,
        uint32_t wFill,
        const ttnn::Tensor& any,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace data_movement
}  // namespace operations

constexpr auto fill_rm = ttnn::register_operation<"ttnn::fill_rm", ttnn::operations::data_movement::FillRMOperation>();
constexpr auto fill_ones_rm =
    ttnn::register_operation<"ttnn::fill_ones_rm", ttnn::operations::data_movement::FillOnesRMOperation>();

}  // namespace ttnn
