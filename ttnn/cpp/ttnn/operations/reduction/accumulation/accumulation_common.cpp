// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "accumulation_common.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"

namespace ttnn::operations::reduction::accumulation::common {

Tensor preprocess_input_tensor(
    const Tensor& input_tensor,
    const int32_t& cum_axis,
    permutation_t& permutation,
    int32_t& accumulation_axis,
    std::optional<DataType>& dtype) {
    Tensor processed_tensor = input_tensor;
    const auto& input_dtype = input_tensor.dtype();
    if (dtype.has_value() && (input_dtype != *dtype)) {
        processed_tensor = ttnn::typecast(input_tensor, input_dtype, *dtype);
    }
    const auto& input_shape = processed_tensor.logical_shape();
    const auto& input_rank = input_shape.rank();
    if (input_rank - cum_axis < FOUR_DIMENSIONS) {
        int32_t final_rank = input_rank;
        int32_t final_cum_axis = cum_axis;
        if (input_rank < FOUR_DIMENSIONS) {
            ttsl::SmallVector<uint32_t> new_dims = {};
            for (int32_t i = input_rank; i < FOUR_DIMENSIONS; ++i) {
                new_dims.push_back(1);
            }
            new_dims.insert(new_dims.end(), input_shape.cbegin(), input_shape.cend());
            ttnn::Shape new_shape(new_dims);

            processed_tensor = ttnn::reshape(processed_tensor, new_shape);

            // Update params
            final_rank = FOUR_DIMENSIONS;
            final_cum_axis += (FOUR_DIMENSIONS - input_rank);
        }

        // Create permutation that just swaps cumulation axis with the first dim
        permutation = std::decay_t<decltype(permutation)>(final_rank);
        std::iota(permutation.begin(), permutation.end(), FIRST_DIMENSION);
        accumulation_axis = FIRST_DIMENSION;
        permutation[accumulation_axis] = final_cum_axis;
        permutation[final_cum_axis] = accumulation_axis;

        return ttnn::permute(processed_tensor, permutation, processed_tensor.memory_config());
    }
    accumulation_axis = cum_axis;

    return processed_tensor;
}

Tensor postprocess_output_tensor(
    const Tensor& output_tensor,
    const int32_t& dim,
    const permutation_t& permutation,
    const ttnn::Shape& original_shape,
    const int32_t& original_rank) {
    Tensor processed_tensor = output_tensor;

    if (original_rank - dim < FOUR_DIMENSIONS) {
        processed_tensor = ttnn::permute(processed_tensor, permutation, processed_tensor.memory_config());
        if (original_rank < FOUR_DIMENSIONS) {
            processed_tensor = ttnn::reshape(processed_tensor, original_shape);
        }
    }

    return processed_tensor;
}

void validate_output_tensor(const Tensor& input_tensor, const Tensor& output_tensor) {
    TT_FATAL(is_device_tensor(output_tensor), "Preallocated output tensor must be on device");
    TT_FATAL(
        input_tensor.logical_shape() == output_tensor.logical_shape(),
        "Shape mismatch: input tensor shape {} does not match output tensor shape {}.",
        input_tensor.logical_shape(),
        output_tensor.logical_shape());
    // The accumulation and permute device ops that now write directly into the preallocated output only have
    // interleaved, tiled kernels; neither validates the layout/memory_layout of the provided output tensor. Assert
    // the assumption here so a non-tile or sharded preallocated output fails early and clearly, rather than
    // producing undefined behaviour when the op writes through its buffer.
    TT_FATAL(
        output_tensor.layout() == Layout::TILE, "Preallocated output tensor must have TILE layout.");
    TT_FATAL(
        output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Preallocated output tensor must have INTERLEAVED memory layout.");
}

Tensor accumulation_invoke(
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_out,
    const bool& reverse_order,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::prim::AccumulationOp op) {
    const auto& input_shape = input_tensor.logical_shape();
    const int32_t& input_rank = input_shape.rank();

    if (optional_out.has_value()) {
        validate_output_tensor(input_tensor, *optional_out);
    }

    if (input_rank == 0 || input_tensor.logical_volume() == 0) {
        if (!optional_out.has_value()) {
            return ttnn::clone(
                input_tensor, /*dtype=*/std::nullopt, memory_config, /*compute_kernel_config=*/std::nullopt);
        }

        Tensor& preallocated_tensor = optional_out.value();
        // It only makes sense to copy non-zero volume tensor.
        if (input_tensor.logical_volume() > 0) {
            ttnn::copy(input_tensor, preallocated_tensor);
        }
        return preallocated_tensor;
    }

    TT_FATAL(
        ((dim >= -static_cast<decltype(dim)>(input_shape.rank())) &&
         (dim < static_cast<decltype(dim)>(input_shape.rank()))),
        "The requested accumulation axis is {}, while the input tensor has rank {}.",
        dim,
        input_tensor.padded_shape().rank());

    // Normalize negative dim
    const int32_t cum_axis = (dim < 0) ? (dim + input_rank) : dim;

    // preprocess_input_tensor / postprocess_output_tensor only reshape-to-4D and permute the tensor (changing its
    // logical shape) when this predicate holds. Otherwise the tensor keeps its original shape end-to-end, so
    // wip_tensor and optional_out share the same logical shape and the op can write straight into optional_out.
    const bool shape_is_transformed = (input_rank - cum_axis < FOUR_DIMENSIONS);

    Tensor wip_tensor = input_tensor;
    ttsl::SmallVector<int64_t> permutation;
    int32_t accumulation_axis;
    wip_tensor = common::preprocess_input_tensor(wip_tensor, cum_axis, permutation, accumulation_axis, dtype);

    if (optional_out.has_value() && !shape_is_transformed) {
        // Fast path: the shape is preserved, so hand optional_out to the prim as its output buffer and let the
        // device op write directly into the caller's preallocated tensor (no extra allocation, no copy).
        wip_tensor = ttnn::prim::accumulation(
            wip_tensor,
            accumulation_axis,
            dtype,
            reverse_order,
            std::move(optional_out),
            memory_config.has_value() ? memory_config.value() : wip_tensor.memory_config(),
            op);
        // No-op here (original_rank - dim >= FOUR_DIMENSIONS), called for uniformity.
        return common::postprocess_output_tensor(wip_tensor, cum_axis, permutation, input_shape, input_rank);
    }

    if (optional_out.has_value()) {
        // General path: preprocessing reshaped/permuted the tensor, so the accumulation op cannot write into
        // optional_out directly (its shape no longer matches). Rather than add a full-tensor copy on top, the
        // *unavoidable* permute-back that postprocessing performs is redirected to write its result straight into
        // optional_out's own buffer -- the caller's preallocated tensor is genuinely reused, with no extra pass.
        Tensor& preallocated = *optional_out;

        // Produce the accumulation result in optional_out's dtype. ttnn semantics give a preallocated output's
        // dtype precedence, and it also means the dtype-preserving permute below lands correctly without a
        // separate typecast (the device op converts on write, exactly like the fast path above).
        wip_tensor = ttnn::prim::accumulation(
            wip_tensor,
            accumulation_axis,
            preallocated.dtype(),
            reverse_order,
            std::nullopt,
            memory_config.has_value() ? memory_config.value() : wip_tensor.memory_config(),
            op);

        // Mirror postprocess_output_tensor, but target optional_out's storage. postprocess permutes the tensor
        // back (always) and, when the input rank was < 4, reshapes off the padded leading dims (a 0-cost relabel
        // that shares the buffer). We therefore point the permute-back at optional_out (viewed with the
        // intermediate 4D shape when needed) so it writes through to the caller's buffer.
        //
        // The permute's output shape equals wip_tensor's shape with `permutation` applied, which is precisely the
        // 4D shape preprocess reshaped the input into (`permutation` swaps axis 0 with the cumulation axis, so it
        // is its own inverse).
        const ttsl::SmallVector<uint32_t> perm(permutation.begin(), permutation.end());
        const auto& wip_shape = wip_tensor.logical_shape();
        ttsl::SmallVector<uint32_t> permuted_dims(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            permuted_dims[i] = wip_shape[perm[i]];
        }
        const ttnn::Shape intermediate_shape(std::move(permuted_dims));

        // For rank < 4 the intermediate shape only left-pads optional_out with unit dims (preprocess reshaped the
        // input the same way, and `permutation` is its own inverse), so this is a pure metadata relabel that must
        // share optional_out's buffer. ttnn::experimental::view guarantees a zero-cost reinterpret (unlike reshape,
        // which may fall back to a data-moving path). We pass optional_out's own tile-aligned padded shape,
        // left-padded with unit dims, so the view stays tile-aligned and physically identical to optional_out. For
        // rank >= 4 preprocess only permutes (no reshape), so wip and optional_out already share this shape and we
        // permute into optional_out directly. Either way the permute below writes through to the caller's buffer.
        Tensor permute_target = preallocated;
        if (input_rank < static_cast<int32_t>(FOUR_DIMENSIONS)) {
            // Left-pad optional_out's OWN padded shape with unit dims up to 4D. We index by the padded shape's own
            // rank, not input_rank: a TILE tensor's padded shape is rank max(logical_rank, 2) (tile alignment has
            // size 2), so a logical rank-1 tensor [W] already has a rank-2 padded shape [32, round_up(W,32)].
            // Indexing by input_rank would drop the real padded width for rank-1 inputs and mis-size the view.
            const auto& out_padded = preallocated.padded_shape();
            const int32_t out_padded_rank = static_cast<int32_t>(out_padded.rank());
            ttsl::SmallVector<uint32_t> padded_dims(FOUR_DIMENSIONS, 1);
            const int32_t pad = static_cast<int32_t>(FOUR_DIMENSIONS) - out_padded_rank;
            for (int32_t i = 0; i < out_padded_rank; ++i) {
                padded_dims[pad + i] = out_padded[i];
            }
            const ttnn::Shape intermediate_padded_shape(std::move(padded_dims));
            permute_target = ttnn::experimental::view(preallocated, intermediate_shape, intermediate_padded_shape);
        }

        // prim::permute does not validate that its permuted output shape matches the provided output tensor; the
        // zero-copy correctness here rests on preprocess_input_tensor producing exactly a unit-dim left-pad plus a
        // self-inverse axis swap, so permute_target's shape must equal wip permuted by perm. Guard it explicitly so
        // a future change to that helper fails loudly instead of silently writing into a mismatched buffer.
        TT_FATAL(
            permute_target.logical_shape() == intermediate_shape,
            "Accumulation optional_out reuse: permute target shape {} does not match the expected permuted output "
            "shape {}. preprocess_input_tensor's shape transform may have changed.",
            permute_target.logical_shape(),
            intermediate_shape);

        ttnn::prim::permute(wip_tensor, perm, permute_target.memory_config(), permute_target, /*pad_value=*/0.0f);
        return preallocated;
    }

    wip_tensor = ttnn::prim::accumulation(
        wip_tensor,
        accumulation_axis,
        dtype,
        reverse_order,
        std::nullopt,
        memory_config.has_value() ? memory_config.value() : wip_tensor.memory_config(),
        op);
    return common::postprocess_output_tensor(wip_tensor, cum_axis, permutation, input_shape, input_rank);
}

}  // namespace ttnn::operations::reduction::accumulation::common
