// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// With runtime_shapes, a row-major interleaved input's slice of the same extent and channel range uses one program at
// every start and input size, which take their effect through runtime args, and the slice's H and W ends may lie past
// the input, where it reads zeros. It costs host time on every call, to recompute the per-core args.
template <typename T>
Tensor padded_slice(
    const ttnn::Tensor& input_tensor,
    ttsl::Span<const T> begins,
    ttsl::Span<const T> ends,
    ttsl::Span<const T> step,
    const MemoryConfig& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<float>& pad_value = std::nullopt,
    bool runtime_shapes = false);

template <typename T>
Tensor padded_slice(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<T>& begins,
    const ttsl::SmallVector<T>& ends,
    const ttsl::SmallVector<T>& step,
    const MemoryConfig& memory_config_arg,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<float>& pad_value = std::nullopt,
    bool runtime_shapes = false) {
    return padded_slice(
        input_tensor,
        ttsl::Span<const T>(begins),
        ttsl::Span<const T>(ends),
        ttsl::Span<const T>(step),
        memory_config_arg,
        optional_output_tensor,
        pad_value,
        runtime_shapes);
}

}  // namespace ttnn::experimental
