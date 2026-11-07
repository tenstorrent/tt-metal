// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/embedding/embedding.hpp"

#include <utility>
#include "ttnn/operations/core/core.hpp"
#include "device/embedding_device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include <ttnn/operations/copy/typecast/typecast.hpp>

namespace ttnn {

ttnn::Tensor embedding(
    const Tensor& input_tensor,
    const Tensor& weight_arg,
    const std::optional<int>& pad_token,
    const std::optional<ttnn::Layout>& layout,
    ttnn::prim::EmbeddingsType embeddings_type,
    const std::optional<const DataType> dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    if (pad_token.has_value()) {
        embeddings_type = ttnn::prim::EmbeddingsType::PADDED;
    }
    Tensor mutable_weight = weight_arg;

    if (mutable_weight.layout() == ttnn::TILE_LAYOUT) {
        mutable_weight = ttnn::to_layout(mutable_weight, ttnn::ROW_MAJOR_LAYOUT);
    }
    auto hidden_embedding_dim = mutable_weight.logical_shape()[-1];
    auto original_input_rank = input_tensor.logical_shape().rank();
    auto weight = ttnn::unsqueeze_to_4D(mutable_weight);

    // If indices tensor is 1 dimensional, batch size is 1
    auto batch_size = (input_tensor.logical_shape().rank() == 1) ? 1 : input_tensor.logical_shape()[0];
    auto sentence_size = input_tensor.logical_shape()[-1];
    auto embedding_input_tensor = input_tensor;
    if (input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        embedding_input_tensor = ttnn::reshape(input_tensor, ttnn::Shape({batch_size, 1, 1, sentence_size}));
    }

    // If layout is row major, OR if the input tensor is not a multiple of TILE_HEIGHT, then we cannot use tilized
    bool fused_tilized = false;
    if (embedding_input_tensor.padded_shape()[-1] % tt::constants::TILE_HEIGHT == 0 &&
        weight.padded_shape()[-1] % tt::constants::TILE_WIDTH == 0) {
        if (layout.has_value()) {
            if (layout.value() == ttnn::TILE_LAYOUT) {
                fused_tilized = true;
            }
        } else if (weight_arg.layout() == ttnn::TILE_LAYOUT) {
            fused_tilized = true;
        } else {
            bool typecast_needed = dtype.has_value() && (dtype.value() != weight.dtype());
            TT_FATAL(!typecast_needed, "Can only typecast output embeddings when producing TILE_LAYOUT output");
        }
    }

    // Decide ahead of time whether this invocation needs the reshape / to_layout /
    // typecast adjustments after the primitive. If the caller's out= already has the
    // final shape, layout, and dtype, we can forward it to the primitive so the
    // primitive writes straight into it and no copy is needed. Otherwise, run the
    // primitive without out=, perform the adjustments, and copy into out= once at the
    // end.
    const ttnn::Shape final_shape = (original_input_rank == 1)
        ? ttnn::Shape({sentence_size, hidden_embedding_dim})
        : ttnn::Shape({batch_size, sentence_size, hidden_embedding_dim});
    const ttnn::Layout final_layout = layout.value_or(weight_arg.layout());
    const ttnn::DataType final_dtype = dtype.value_or(weight.dtype());
    // Layout adjustments only apply when the final layout is TILE; in ROW_MAJOR the
    // primitive output passes through to_layout as a no-op and typecast is skipped.
    const bool typecast_needed_after =
        final_layout == ttnn::TILE_LAYOUT && final_dtype != weight.dtype();

    const bool out_matches_final = optional_output_tensor.has_value() &&
        optional_output_tensor->storage_type() == StorageType::DEVICE &&
        optional_output_tensor->logical_shape() == final_shape &&
        optional_output_tensor->layout() == final_layout &&
        optional_output_tensor->dtype() == final_dtype;

    // Forward out= to the primitive only when no post-primitive reshape / layout /
    // typecast would change the buffer. Otherwise a stale copy of the primitive's
    // output would live in out= after the subsequent adjustments produced a new
    // tensor.
    const bool reshape_changes_tensor = embedding_input_tensor.logical_shape() != final_shape;
    const bool forward_out_to_primitive =
        out_matches_final && !reshape_changes_tensor && !typecast_needed_after;

    auto embeddings = ttnn::prim::embedding(
        embedding_input_tensor,
        weight,
        fused_tilized,
        embeddings_type,
        memory_config,
        pad_token,
        forward_out_to_primitive ? optional_output_tensor : std::nullopt);

    if (forward_out_to_primitive) {
        // Primitive wrote into out= directly; no reshape / layout / dtype adjustment
        // applies on this path by construction.
        return embeddings;
    }

    // Don't include batch_size if there was none
    if (original_input_rank == 1) {
        embeddings = ttnn::reshape(embeddings, Shape({sentence_size, hidden_embedding_dim}));
    } else {
        embeddings = ttnn::reshape(embeddings, Shape({batch_size, sentence_size, hidden_embedding_dim}));
    }
    embeddings = ttnn::to_layout(embeddings, final_layout);
    if (embeddings.layout() == ttnn::TILE_LAYOUT && embeddings.dtype() != final_dtype) {
        embeddings = ttnn::typecast(embeddings, final_dtype);
    }
    if (optional_output_tensor.has_value() &&
        optional_output_tensor->storage_type() == StorageType::DEVICE &&
        embeddings.storage_type() == StorageType::DEVICE &&
        embeddings.buffer() != optional_output_tensor->buffer()) {
        return ttnn::copy(embeddings, *optional_output_tensor);
    }
    return embeddings;
}

}  // namespace ttnn
