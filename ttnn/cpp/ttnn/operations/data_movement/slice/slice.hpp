// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct SliceOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        std::span<const uint32_t> begins,
        std::span<const uint32_t> ends,
        std::span<const uint32_t> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::span<const uint32_t> begins,
        std::span<const uint32_t> ends,
        std::span<const uint32_t> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        std::span<const int> begins,
        std::span<const int> ends,
        std::span<const int> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::span<const int> begins,
        std::span<const int> ends,
        std::span<const int> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    template<typename T, size_t N>
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const std::array<T, N>& begins,
        const std::array<T, N>& ends,
        const std::array<T, N>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(queue_id, input_tensor, std::span<const T>(begins), std::span<const T>(ends), std::span<const T>(step), memory_config_arg, optional_output_tensor);
    }

    template<typename T, size_t N>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::array<T, N>& begins,
        const std::array<T, N>& ends,
        const std::array<T, N>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(ttnn::DefaultQueueId, input_tensor, std::span<const T>(begins), std::span<const T>(ends), std::span<const T>(step), memory_config_arg, optional_output_tensor);
    }
};

}  // namespace data_movement
}  // namespace operations

constexpr auto slice =
    ttnn::register_operation_with_auto_launch_op<"ttnn::slice", ttnn::operations::data_movement::SliceOperation>();

}  // namespace ttnn
