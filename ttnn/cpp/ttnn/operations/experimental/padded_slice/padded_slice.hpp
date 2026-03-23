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
    const ttnn::Tensor& input_tensor,
    ttsl::Span<const T> begins,
    ttsl::Span<const T> ends,
    ttsl::Span<const T> step,
    const MemoryConfig& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<float>& pad_value = std::nullopt);

template <typename T>
Tensor padded_slice(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<T>& begins,
    const ttnn::SmallVector<T>& ends,
    const ttnn::SmallVector<T>& step,
    const MemoryConfig& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<float>& pad_value = std::nullopt) {
    return padded_slice(
        input_tensor,
        ttsl::Span<const T>(begins),
        ttsl::Span<const T>(ends),
        ttsl::Span<const T>(step),
        memory_config_arg,
        optional_output_tensor,
        pad_value);
}

}  // namespace ttnn::experimental
