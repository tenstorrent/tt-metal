// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "set_rows.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/indexed_fill/indexed_fill.hpp"

namespace ttnn {

Tensor set_rows(
    const Tensor& input_tensor,
    const Tensor& indices_tensor,
    const Tensor& updates_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    // Fail fast on the set_rows shapes before any normalization; indexed_fill re-validates the
    // normalized 4D call.
    TT_FATAL(
        input_tensor.logical_shape().rank() == 2,
        "set_rows: input tensor must be rank 2 [H, W], got rank {} with shape {}",
        input_tensor.logical_shape().rank(),
        input_tensor.logical_shape());
    TT_FATAL(
        indices_tensor.logical_shape().rank() == 1,
        "set_rows: indices tensor must be rank 1 [N], got rank {} with shape {}",
        indices_tensor.logical_shape().rank(),
        indices_tensor.logical_shape());
    TT_FATAL(
        updates_tensor.logical_shape().rank() == 2,
        "set_rows: updates tensor must be rank 2 [N, W], got rank {} with shape {}",
        updates_tensor.logical_shape().rank(),
        updates_tensor.logical_shape());
    TT_FATAL(indices_tensor.logical_shape()[0] > 0, "set_rows: empty index vectors are not supported");
    TT_FATAL(
        updates_tensor.logical_shape()[0] == indices_tensor.logical_shape()[0],
        "set_rows: updates must supply one row per index; got {} updates for {} indices",
        updates_tensor.logical_shape()[0],
        indices_tensor.logical_shape()[0]);
    TT_FATAL(
        updates_tensor.logical_shape()[-1] == input_tensor.logical_shape()[-1],
        "set_rows: updates width {} must match the input width {}",
        updates_tensor.logical_shape()[-1],
        input_tensor.logical_shape()[-1]);
    TT_FATAL(
        updates_tensor.dtype() == input_tensor.dtype(),
        "set_rows: updates dtype {} must match the input dtype {}",
        updates_tensor.dtype(),
        input_tensor.dtype());

    // torch.index_put_ / index_copy_ raise for out-of-range indices; the kernel writes a whole row
    // per index and cannot check before the NoC write, so the bounds check happens here, on the
    // host, before dispatch. The index vector is tiny next to the rows being written.
    const auto num_input_rows = input_tensor.logical_shape()[0];
    const auto host_indices = indices_tensor.cpu();
    if (indices_tensor.dtype() == DataType::INT32) {
        for (const int32_t index : host_indices.to_vector<int32_t>()) {
            TT_FATAL(
                index >= 0 && static_cast<uint32_t>(index) < num_input_rows,
                "set_rows: index {} out of range for an input with {} rows (torch.index_put_ semantics)",
                index,
                num_input_rows);
        }
    } else {
        for (const uint32_t index : host_indices.to_vector<uint32_t>()) {
            TT_FATAL(
                index < num_input_rows,
                "set_rows: index {} out of range for an input with {} rows (torch.index_put_ semantics)",
                index,
                num_input_rows);
        }
    }

    // Normalize to the [B, H, W, D] form indexed_fill expects; dim 2 of [1, 1, H, W] is the row
    // axis, which is the native ROW_MAJOR kernel path (dim < rank - 1, no permute fallback).
    const auto input_4d = ttnn::reshape(input_tensor, ttnn::Shape({1, 1, num_input_rows, input_tensor.logical_shape()[-1]}));
    const auto updates_4d = ttnn::reshape(
        updates_tensor, ttnn::Shape({1, 1, indices_tensor.logical_shape()[0], updates_tensor.logical_shape()[-1]}));
    const auto indices_4d = ttnn::reshape(indices_tensor, ttnn::Shape({1, 1, 1, indices_tensor.logical_shape()[0]}));

    auto output_4d = ttnn::indexed_fill(indices_4d, input_4d, updates_4d, memory_config, /*dim=*/2);
    auto output = ttnn::reshape(output_4d, ttnn::Shape({num_input_rows, input_tensor.logical_shape()[-1]}));

    // `out` takes precedence over memory_config: write straight into the provided tensor.
    if (optional_output_tensor.has_value()) {
        return ttnn::to_memory_config(
            output, optional_output_tensor->memory_config(), std::nullopt, optional_output_tensor);
    }
    return output;
}

}  // namespace ttnn
