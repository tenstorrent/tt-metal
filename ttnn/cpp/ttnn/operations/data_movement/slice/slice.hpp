// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct SliceOperation {
    template<typename T>
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        std::span<const T> begins,
        std::span<const T> ends,
        std::span<const T> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    template<typename T>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        std::span<const T> output_tensor_start,
        std::span<const T> output_tensor_end,
        std::span<const T> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    template<typename T>
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<T>& begins,
        const ttnn::SmallVector<T>& ends,
        const ttnn::SmallVector<T>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(queue_id, input_tensor, std::span<const T>(begins.begin(), begins.end()), std::span<const T>(ends.begin(), ends.end()), std::span<const T>(step.begin(), step.end()), memory_config_arg, optional_output_tensor);
    }

    template<typename T>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<T>& begins,
        const ttnn::SmallVector<T>& ends,
        const ttnn::SmallVector<T>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt) {
        return invoke(input_tensor, std::span<const T>(begins.begin(), begins.end()), std::span<const T>(ends.begin(), ends.end()), std::span<const T>(step.begin(), step.end()), memory_config_arg, optional_output_tensor);
    }

    template<typename T, std::size_t N>
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const std::array<T, N> &output_tensor_start,
        const std::array<T, N> &output_tensor_end,
        const std::array<T, N> &step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);

    template<typename T, std::size_t N>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::array<T, N> &output_tensor_start,
        const std::array<T, N> &output_tensor_end,
        const std::array<T, N> &step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);


};

}  // namespace data_movement
}  // namespace operations

constexpr auto slice =
    ttnn::register_operation_with_auto_launch_op<"ttnn::slice", ttnn::operations::data_movement::SliceOperation>();

}  // namespace ttnn
