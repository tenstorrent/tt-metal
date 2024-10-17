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
        uint8_t queue_id,
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

    static ttnn::Tensor invoke(
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
        uint8_t queue_id,
        uint32_t N,
        uint32_t C,
        uint32_t H,
        uint32_t W,
        uint32_t hFill,
        uint32_t wFill,
        const ttnn::Tensor& any,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

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

struct FullOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        uint32_t N,
        uint32_t C,
        uint32_t H,
        uint32_t W,
        float fill_value,
        const ttnn::Tensor& any,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

    static ttnn::Tensor invoke(
        uint32_t N,
        uint32_t C,
        uint32_t H,
        uint32_t W,
        float fill_value,
        const ttnn::Tensor& any,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace fill_rm
}  // namespace operations

constexpr auto fill_rm = ttnn::register_operation<"ttnn::fill_rm", ttnn::operations::data_movement::FillRMOperation>();
constexpr auto fill_ones_rm = ttnn::register_operation<"ttnn::fill_ones_rm", ttnn::operations::data_movement::FillOnesRMOperation>();
constexpr auto full = ttnn::register_operation<"ttnn::full", ttnn::operations::data_movement::FullOperation>();

}  // namespace ttnn
