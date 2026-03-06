// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

template <typename T>
Tensor padded_slice(
    const Tensor& input_tensor,
    const ttnn::SmallVector<T>& padded_slice_start,
    const ttnn::SmallVector<T>& padded_slice_end,
    const ttnn::SmallVector<T>& padded_slice_step,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<float>& pad_value = std::nullopt);

template <typename T>
Tensor padded_slice(
    const Tensor& input_tensor,
    const ttnn::SmallVector<T>& padded_slice_start,
    const ttnn::SmallVector<T>& padded_slice_end,
    const std::optional<ttnn::SmallVector<T>>& padded_slice_step,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<float>& pad_value = std::nullopt) {
    const auto step_value = padded_slice_step.value_or(ttnn::SmallVector<T>(padded_slice_end.size(), 1));
    return padded_slice(
        input_tensor,
        padded_slice_start,
        padded_slice_end,
        step_value,
        memory_config,
        optional_output_tensor,
        pad_value);
}

}  // namespace ttnn::experimental
