// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <optional>

#include "bcast_to.hpp"
#include <tt_stl/small_vector.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/experimental/bcast_to/device/bcast_to_device_operation.hpp"

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {
auto check_shape(const ttnn::Tensor& input, const ttnn::Shape& output_shape) {
    auto input_shape = input.logical_shape();
    TT_FATAL(
        input_shape.size() <= output_shape.size(),
        "Input tensor shape {}({}) must be at least as large as the broadcast shape {}({}), which it is not",
        input_shape,
        input_shape.size(),
        output_shape,
        output_shape.size());

    TT_FATAL(
        input_shape.size() <= 4 and output_shape.size() <= 4,
        "Tensor shape and broadcast size {}({}) {}({}) must be at most 4D",
        input_shape,
        input_shape.size(),
        output_shape,
        output_shape.size());

    // Validate broadcasting rules (checking from right to left)
    size_t input_ndim = input_shape.size();

    for (int i = -1; i >= -static_cast<int>(input_ndim); --i) {
        // Check dimensions from the right side
        uint32_t input_dim = input_shape[i];
        uint32_t output_dim = output_shape[i];

        // For broadcasting, either dimensions must match or input must be 1
        TT_FATAL(
            (input_dim == output_dim) || (input_dim == 1),
            "Input dimension {} (size {}) cannot be broadcast to output dimension {} (size {})",
            i,
            input_dim,
            i,
            output_dim);
    }
}
}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

namespace ttnn::operations::experimental {
Tensor BcastTo::invoke(
    const Tensor& input,
    const Shape& output_shape,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output) {
    CMAKE_UNIQUE_NAMESPACE::check_shape(input, output_shape);

    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 or input.dtype() == DataType::FLOAT32,
        "For input dtype {}, only bfloat16 and float32 are supported",
        input.dtype());

    if (output.has_value()) {
        TT_FATAL(
            output.value().dtype() == DataType::BFLOAT16 or output.value().dtype() == DataType::FLOAT32,
            "For output dtype {}, only bfloat16 and float32 are supported",
            output.value().dtype());
    }

    return ttnn::prim::bcast_to(input, output_shape, memory_config, output);
}
}  // namespace ttnn::operations::experimental
