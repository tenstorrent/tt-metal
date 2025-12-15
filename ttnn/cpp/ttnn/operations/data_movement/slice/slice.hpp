// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::data_movement {

struct SliceOperation {
    template <typename T>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        tt::stl::Span<const T> begins,
        tt::stl::Span<const T> ends,
        tt::stl::Span<const T> step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    template <typename T>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::SmallVector<T>& begins,
        const ttnn::SmallVector<T>& ends,
        const ttnn::SmallVector<T>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
        return invoke(
            input_tensor,
            tt::stl::Span<const T>(begins),
            tt::stl::Span<const T>(ends),
            tt::stl::Span<const T>(step),
            memory_config_arg,
            optional_output_tensor,
            pad_value,
            sub_core_grids);
    }

    template <typename T, std::size_t N>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::array<T, N>& output_tensor_start,
        const std::array<T, N>& output_tensor_end,
        const std::array<T, N>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

    template <typename T>
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const ttnn::Tensor& output_tensor_start,
        const ttnn::Tensor& output_tensor_end,
        const std::optional<ttnn::SmallVector<T>>& step,
        const std::optional<MemoryConfig>& memory_config_arg = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt,
        const std::optional<float>& pad_value = std::nullopt,
        const std::optional<uint32_t>& slice_dim = std::nullopt,
        const std::optional<uint32_t>& num_devices = std::nullopt,
        const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);
};

}  // namespace operations::data_movement

constexpr auto slice = ttnn::register_operation<"ttnn::slice", ttnn::operations::data_movement::SliceOperation>();

}  // namespace ttnn
