// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/experimental/quasar/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/operations/experimental/quasar/pad/device/pad_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/experimental/quasar/typecast/typecast.hpp"
#include "ttnn/operation.hpp"
#include <numeric>
#include <ttnn/tensor/types.hpp>

#include "pad.hpp"

namespace ttnn::operations::experimental::quasar::detail {

// Shared data-movement helpers (defined in data_movement/common/common.hpp) used by this op.
// They previously resolved via the enclosing data_movement namespace; bring them in explicitly
// now that this op lives under the experimental::quasar namespace.
using ttnn::operations::data_movement::compute_padded_shape;
using ttnn::operations::data_movement::create_sharded_memory_config;
using ttnn::operations::data_movement::ShardStrategy;
using ttnn::operations::data_movement::squeeze_from_ND_to_4D;
using ttnn::operations::data_movement::squeeze_or_unsqueeze_shape_to_ND;

namespace {

bool eq_spans(const auto a, const auto b) { return std::equal(a.begin(), a.end(), b.begin(), b.end()); }

ttnn::Shape compute_padded_logical_shape(const ttnn::Shape& input_shape, const ttsl::SmallVector<PadSpecDim>& padding) {
    ttsl::SmallVector<uint32_t> output_shape(input_shape.rank());
    for (size_t i = 0; i < input_shape.rank(); i++) {
        output_shape[i] = padding[i].before_elements + input_shape[i] + padding[i].after_elements;
    }
    return ttnn::Shape(std::move(output_shape));
}

// For rank > 4, leading dimensions [0, rank-3) are merged into a single axis by squeeze_from_ND_to_4D and
// cannot be padded through the 4D kernel directly. Reshape to 4D, pad the collapsed axis, reshape back.
ttnn::Tensor pad_leading_dimension_via_reshape(
    const ttnn::Tensor& input_tensor,
    int dim,
    const PadSpecDim& pad_spec,
    const float value,
    const bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    const auto& shape = input_tensor.logical_shape();
    const int rank = static_cast<int>(shape.rank());
    TT_FATAL(
        rank > 4 && dim >= 0 && dim < rank - 3,
        "pad_leading_dimension_via_reshape: expected dim in [0, rank-3), got dim {} for rank {}",
        dim,
        rank);

    const auto shape_view = shape.view();
    const uint32_t before =
        std::accumulate(shape_view.begin(), shape_view.begin() + dim, 1u, std::multiplies<uint32_t>());
    const uint32_t middle =
        std::accumulate(shape_view.begin() + dim, shape_view.begin() + rank - 2, 1u, std::multiplies<uint32_t>());

    auto reshaped = ttnn::operations::experimental::quasar::reshape(
        input_tensor, ttnn::Shape({before, middle, shape[-2], shape[-1]}));

    const uint32_t inner_extent =
        std::accumulate(shape_view.begin() + dim + 1, shape_view.begin() + rank - 2, 1u, std::multiplies<uint32_t>());

    ttsl::SmallVector<PadSpecDim> padding_4d = {
        {0, 0}, {pad_spec.before_elements * inner_extent, pad_spec.after_elements * inner_extent}, {0, 0}, {0, 0}};

    auto padded = ttnn::operations::experimental::quasar::pad(
        reshaped, padding_4d, value, use_multicore, std::nullopt, sub_core_grids);

    ttsl::SmallVector<uint32_t> output_shape(shape.view().begin(), shape.view().end());
    output_shape[dim] += pad_spec.before_elements + pad_spec.after_elements;
    return ttnn::operations::experimental::quasar::reshape(padded, ttnn::Shape(output_shape));
}

ttnn::Tensor pad_leading_dimensions(
    const ttnn::Tensor& input_tensor,
    ttsl::SmallVector<PadSpecDim>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    auto result = input_tensor;
    const int rank = static_cast<int>(input_tensor.logical_shape().rank());
    const int leading_dims_end = rank - 3;
    const int defer_dim = rank - 4;
    for (int dim = 0; dim < leading_dims_end; dim++) {
        if (padding[dim].before_elements == 0 && padding[dim].after_elements == 0) {
            continue;
        }
        if (dim == defer_dim) {
            const auto shape_view = result.logical_shape().view();
            const uint32_t squeezed_axis0 = std::accumulate(
                shape_view.begin(), shape_view.begin() + leading_dims_end, 1u, std::multiplies<uint32_t>());
            if (squeezed_axis0 == shape_view[defer_dim]) {
                continue;
            }
        }
        result = pad_leading_dimension_via_reshape(result, dim, padding[dim], value, use_multicore, sub_core_grids);
        padding[dim] = {0, 0};
    }
    return result;
}

}  // namespace

ttnn::Tensor apply_leading_dimension_padding(
    const ttnn::Tensor& input_tensor,
    ttsl::SmallVector<PadSpecDim>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return pad_leading_dimensions(input_tensor, padding, value, use_multicore, sub_core_grids);
}

ttnn::Tensor pad_impl(
    const ttnn::Tensor& input_tensor,
    std::span<const uint32_t> output_padded_shape,
    std::span<const uint32_t> input_tensor_start,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    auto input_logical_shape = input_tensor.logical_shape().view();
    // on host
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        if (eq_spans(input_logical_shape, output_padded_shape)) {
            return input_tensor;
        }
        return input_tensor.pad(ttnn::Shape(output_padded_shape), ttnn::Shape{input_tensor_start}, value);
    }

    // on device
    auto input_tensor_shape = input_tensor.logical_shape();
    const auto rank = input_tensor_shape.rank();

    TT_FATAL(rank == 4, "ttnn.pad: input tensor passed to pad_impl must have rank == 4, but got rank {}.", rank);
    bool input_output_same = true;
    for (size_t i = 0; i < rank; i++) {
        if (input_tensor_shape[i] != output_padded_shape[i]) {
            input_output_same = false;
            break;
        }
    }
    if (input_output_same) {
        log_debug(tt::LogOp, "Pad Input and Output Shapes are the same. Skipping pad and returning input tensor.");
        return input_tensor;
    }
    using ShardStrategy = ttnn::operations::data_movement::ShardStrategy;
    using ShardOrientation = tt::tt_metal::ShardOrientation;

    auto output_memory_config = memory_config_arg.value_or(input_tensor.memory_config());

    if (input_tensor.is_sharded() && input_tensor.memory_config().memory_layout() != TensorMemoryLayout::ND_SHARDED &&
        output_memory_config.memory_layout() != TensorMemoryLayout::ND_SHARDED &&
        output_memory_config.memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        auto total_height = [](const auto& shape) {
            return std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<uint32_t>());
        };

        auto height_distinct = [&total_height](const auto& shape, const auto& other_shape) {
            return total_height(shape) != total_height(other_shape);
        };

        auto width_distinct = [](const auto& shape, const auto& other_shape) { return shape[3] != other_shape[3]; };

        uint32_t output_w = output_padded_shape[3];

        if (width_distinct(input_logical_shape, output_padded_shape)) {
            std::array<uint32_t, 4> output_shape_width_padded{
                input_logical_shape[0], input_logical_shape[1], input_logical_shape[2], output_w};
            log_warning(
                tt::LogOp,
                "ttnn.pad: Input is HEIGHT_SHARDED and width padding is required. "
                "Ignoring the provided output memory config and recomputing a HEIGHT_SHARDED config "
                "with shard width equal to the padded output width.");
            auto width_pad_memory_config = create_sharded_memory_config(
                ttnn::Shape{output_shape_width_padded},
                input_tensor.shard_spec()->grid,  // reuse input cores for now: FIXME: can we do better?
                                                  // it's complicated because we need the input shards to be local
                                                  // to the core holding the output shard currently.
                ShardStrategy::HEIGHT,            // stay height sharded
                ShardOrientation::ROW_MAJOR);
            output_memory_config = width_pad_memory_config;

            if (height_distinct(input_logical_shape, output_padded_shape)) {
                // we will decompose the padding into two parts and run two
                // separate pads.
                ttsl::SmallVector<uint32_t> adjusted_input_tensor_start{0, 0, 0, input_tensor_start[3]};

                TT_FATAL(
                    not(height_distinct(input_logical_shape, output_shape_width_padded) and
                        width_distinct(input_logical_shape, output_shape_width_padded)),
                    "infinite recursion");

                // pad width
                auto output_tensor_width_padded = pad_impl(
                    input_tensor,
                    output_shape_width_padded,
                    adjusted_input_tensor_start,
                    value,
                    use_multicore,
                    width_pad_memory_config);

                TT_FATAL(
                    not(height_distinct(output_padded_shape, output_shape_width_padded) and
                        width_distinct(output_padded_shape, output_shape_width_padded)),
                    "infinite recursion");

                auto height_pad_memory_config = create_sharded_memory_config(
                    ttnn::Shape{output_padded_shape},
                    input_tensor.shard_spec()->grid,
                    ShardStrategy::HEIGHT,
                    ShardOrientation::ROW_MAJOR);

                // then pad height
                auto output_tensor_height_padded = pad_impl(
                    output_tensor_width_padded,
                    output_padded_shape,
                    input_tensor_start,
                    value,
                    use_multicore,
                    memory_config_arg.value_or(height_pad_memory_config));
                output_tensor_width_padded.deallocate();  // dealloc temporary width padded tensor
                return output_tensor_height_padded;
            }
        }
    }

    ttnn::Shape output_shape{output_padded_shape};
    return ttnn::prim::qsr::pad(
        input_tensor,
        output_shape,
        output_shape,
        ttnn::Shape{input_tensor_start},
        value,
        output_memory_config,
        use_multicore,
        std::nullopt,
        sub_core_grids);
}

ttnn::Tensor pad_impl(
    const ttnn::Tensor& input_tensor,
    ttsl::SmallVector<PadSpecDim> padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    if (input_tensor.dtype() == DataType::BFLOAT8_B && input_tensor.layout() == Layout::TILE) {
        auto bfloat16_tensor = ttnn::operations::experimental::quasar::typecast(input_tensor, DataType::BFLOAT16);
        auto padded_tensor =
            pad_impl(bfloat16_tensor, padding, value, use_multicore, memory_config_arg, sub_core_grids);
        return ttnn::operations::experimental::quasar::typecast(padded_tensor, DataType::BFLOAT8_B);
    }
    const int original_rank = input_tensor.logical_shape().rank();

    TT_FATAL(padding.size() == original_rank, "ttnn.pad: padding must be the same length as the input tensor rank");

    // Unsqueeze Tensor to 4D if it is not already
    ttnn::Tensor input_tensor_4D;
    if (input_tensor.logical_shape().rank() < 4) {
        input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    } else if (input_tensor.logical_shape().rank() > 4) {
        input_tensor_4D = squeeze_from_ND_to_4D(input_tensor);
    } else {
        input_tensor_4D = input_tensor;
    }
    size_t padding_size = 4;
    size_t extra_index = input_tensor.logical_shape().rank() - 4;
    if (input_tensor.logical_shape().rank() < 4) {
        padding.insert(padding.begin(), 4 - original_rank, {0, 0});
        padding_size = padding.size();
        extra_index = 0;
    }
    if (original_rank > 4) {
        // Only padding[extra_index .. extra_index + 3] reaches the 4D kernel. Dims below extra_index have no
        // slot in the 4D spec, and axis 0 of the squeezed tensor may fold several original dims together.
        const auto is_unpadded = [](const PadSpecDim& p) { return p.before_elements == 0 && p.after_elements == 0; };
        const bool axis0_is_merged =
            input_tensor_4D.logical_shape()[0] != input_tensor.logical_shape()[static_cast<int>(extra_index)];
        TT_FATAL(
            std::all_of(padding.begin(), padding.begin() + static_cast<std::ptrdiff_t>(extra_index), is_unpadded) &&
                (!axis0_is_merged || is_unpadded(padding[extra_index])),
            "ttnn.pad: padding on the leading dimensions of a rank {} tensor must be consumed before the squeeze "
            "to 4D; got a non-zero pad on a dimension that the 4D kernel cannot address",
            original_rank);
    }
    auto input_shape_with_tile_padding =
        (input_tensor_4D.layout() == Layout::TILE) ? input_tensor_4D.padded_shape() : input_tensor_4D.logical_shape();
    // For tilized tensors, we want the shape padded to the nearest tile. For row major, we just want the row size
    // (logical shape).
    std::vector<uint32_t> pad_front_array(padding_size, 0);
    std::vector<uint32_t> output_padded_shape(padding_size, 0);
    for (size_t i = 0; i < padding_size; i++) {
        pad_front_array[i] = padding[i + extra_index].before_elements;
        output_padded_shape[i] = padding[i + extra_index].before_elements + input_shape_with_tile_padding[i] +
                                 padding[i + extra_index].after_elements;
    }

    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        const int target_height = output_padded_shape[padding_size - 2];
        const int target_width = output_padded_shape[padding_size - 1];
        TT_FATAL(
            target_height % ttnn::TILE_SIZE == 0 && target_width % ttnn::TILE_SIZE == 0,
            "ttnn.pad: for tiled tensors padding end must be a multiple of the tile size on height and width for a "
            "tensor in tile layout");
    }

    return pad_impl(
        input_tensor_4D, output_padded_shape, pad_front_array, value, use_multicore, memory_config_arg, sub_core_grids);
}

std::tuple<ttnn::Shape, ttnn::Shape> compute_requested_shape(
    const ttnn::Shape& input_logical_shape, const ttsl::SmallVector<PadSpecDim>& pad_spec) {
    if (std::all_of(pad_spec.begin(), pad_spec.end(), [](auto& p) {
            return p.before_elements == 0 && p.after_elements == 0;
        })) {
        return std::make_tuple(compute_padded_shape(input_logical_shape), compute_padded_shape(input_logical_shape));
    }

    const auto rank = input_logical_shape.rank();
    ttsl::SmallVector<uint32_t> requested_logical_shape_vec(rank, 0);

    std::transform(
        input_logical_shape.cbegin(),
        input_logical_shape.cend(),
        pad_spec.cbegin(),
        requested_logical_shape_vec.begin(),
        [](auto& a, auto& b) { return a + b.after_elements; });

    const ttnn::Shape logical_shape(requested_logical_shape_vec);
    return std::make_tuple(logical_shape, compute_padded_shape(logical_shape));
}

ttnn::Tensor invoke_rm(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding_vec,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    const int original_rank = input_tensor.logical_shape().rank();

    ttnn::Tensor output_tensor =
        pad_impl(input_tensor, padding_vec, value, use_multicore, memory_config_arg, sub_core_grids);

    // output_tensor is currently 4D. We have to squeeze back to the original rank
    if (original_rank <= 4) {
        auto to_vec = [](const auto& span) { return ttsl::SmallVector<uint32_t>{span.begin(), span.end()}; };
        auto output_shape = to_vec(output_tensor.padded_shape().view());
        auto padded_shape = to_vec(output_tensor.padded_shape().view());
        if (const auto rank_diff = output_shape.size() - original_rank; rank_diff) {
            auto remove_prefix = [](auto& source, size_t n) { source.erase(source.begin(), source.begin() + n); };
            remove_prefix(output_shape, rank_diff);
            remove_prefix(padded_shape, rank_diff);
            output_tensor = ttnn::operations::experimental::quasar::reshape(
                output_tensor, ttnn::Shape(output_shape), ttnn::Shape(padded_shape));
            output_tensor = ttnn::operations::experimental::quasar::reshape(output_tensor, ttnn::Shape(padded_shape));
        }
    } else {
        // invoke_rm only ever sees row-major tensors, and pad_impl sizes the 4D output from the *logical*
        // shape in that case (input_shape_with_tile_padding). The padded output therefore carries no
        // alignment padding of its own, so the restored rank-N padded shape is the padded logical shape.
        // Deriving it from input_tensor.padded_shape() instead would disagree with the buffer whenever an
        // input arrives with padded_shape != logical_shape (e.g. an explicit two-shape reshape).
        const auto output_logical_shape = compute_padded_logical_shape(input_tensor.logical_shape(), padding_vec);
        TT_FATAL(
            output_logical_shape.volume() == output_tensor.padded_shape().volume(),
            "ttnn.pad: restored shape {} does not match the padded output volume of {}",
            output_logical_shape,
            output_tensor.padded_shape());
        output_tensor =
            ttnn::operations::experimental::quasar::reshape(output_tensor, output_logical_shape, output_logical_shape);
    }
    return output_tensor;
}

ttnn::Tensor invoke_tile(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding_vec,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    const bool front_padding_is_zero =
        std::all_of(padding_vec.begin(), padding_vec.end(), [](auto& x) { return x.before_elements == 0; });
    TT_FATAL(front_padding_is_zero, "ttnn.pad: on device tile padding does not support front padding");

    const auto& input_logical_shape = input_tensor.logical_shape();
    const auto& input_padded_shape = input_tensor.padded_shape();
    const auto [requested_logical_shape, requested_padded_shape] =
        compute_requested_shape(input_logical_shape, padding_vec);
    const auto requested_rank = requested_logical_shape.rank();

    // Consistent with behavior expected by callers
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        ttnn::Shape zeros(ttsl::SmallVector<uint32_t>(input_logical_shape.rank(), 0));
        return input_tensor.pad(requested_padded_shape, zeros, value);
    }

    const bool pad_upper_dims = !std::equal(
        requested_logical_shape.view().rbegin() + 2,
        requested_logical_shape.view().rend(),
        input_logical_shape.view().rbegin() + 2,
        input_logical_shape.view().rend());

    auto pad_current_tile_dim = [&requested_padded_shape, &input_padded_shape](const int i) {
        return requested_padded_shape[i] == input_padded_shape[i];
    };

    ttnn::Tensor output_tensor = ttnn::fill_implicit_tile_padding(input_tensor, value, memory_config_arg);
    if (requested_rank == 1 || (!pad_upper_dims && pad_current_tile_dim(-1) && pad_current_tile_dim(-2))) {
        output_tensor = ttnn::experimental::view(output_tensor, requested_logical_shape, requested_padded_shape);

    } else {
        // need to align the requested padding to tile size. Note that begin padding is not supported so now just
        // set to zero
        ttsl::SmallVector<PadSpecDim> padded_padding_vec;
        padded_padding_vec.reserve(requested_rank);
        std::transform(
            requested_padded_shape.cbegin(),
            requested_padded_shape.cend(),
            input_padded_shape.cbegin(),
            std::back_inserter(padded_padding_vec),
            [](auto& a, auto& b) { return PadSpecDim{0, a - b}; });

        // this tensor will be 4D
        output_tensor = pad_impl(
            output_tensor, std::move(padded_padding_vec), value, use_multicore, memory_config_arg, sub_core_grids);

        // this is the padded shape
        const auto output_shape = squeeze_or_unsqueeze_shape_to_ND(output_tensor.logical_shape(), requested_rank);

        // "slice" down to logical shape
        output_tensor = ttnn::experimental::view(output_tensor, requested_logical_shape, requested_padded_shape);
    }
    if (output_tensor.memory_config().shard_spec().has_value() !=
        memory_config_arg.value_or(input_tensor.memory_config()).shard_spec().has_value()) {
        if (memory_config_arg.has_value()) {
            output_tensor = ttnn::operations::experimental::quasar::to_memory_config(
                output_tensor, memory_config_arg.value(), std::nullopt);
        } else {
            // memory_config_arg is nullopt → condition can only be true if input is sharded
            // (interleaved input + nullopt config → both sides false → condition false)
            // so input_tensor.shard_spec()->grid is safe here.
            const auto sharded_mem_config = create_sharded_memory_config(
                ttnn::Shape{requested_logical_shape},
                input_tensor.shard_spec()->grid,
                ShardStrategy::HEIGHT,
                ShardOrientation::ROW_MAJOR);
            output_tensor = ttnn::operations::experimental::quasar::to_memory_config(
                output_tensor, sharded_mem_config, std::nullopt);
        }
    }
    return output_tensor;
}
}  // namespace ttnn::operations::experimental::quasar::detail

namespace ttnn::operations::experimental::quasar {

// This function signature is similar to pytorch's signature
// Any rank tensor supported

ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<operations::experimental::quasar::PadSpecDim>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using PadSpecDim = operations::experimental::quasar::PadSpecDim;
    const int original_rank = input_tensor.logical_shape().rank();

    ttsl::SmallVector<PadSpecDim> working_padding = padding;

    if (int diff = original_rank - padding.size(); diff != 0) {
        TT_FATAL(diff > 0, "ttnn.pad: padding len can't be larger than input tensor rank");

        working_padding.insert(working_padding.begin(), diff, {0, 0});
    }

    if (std::all_of(working_padding.begin(), working_padding.end(), [](auto& p) {
            return p.before_elements == 0 && p.after_elements == 0;
        })) {
        return input_tensor;
    }

    ttnn::Tensor working_tensor = input_tensor;
    if (original_rank > 4) {
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            const auto first_pad_idx = static_cast<int>(
                std::find_if(
                    working_padding.begin(),
                    working_padding.end(),
                    [](const PadSpecDim& p) { return p.after_elements != 0; }) -
                working_padding.begin());
            TT_FATAL(
                first_pad_idx >= original_rank - 3,
                "ttnn::pad only supports padding on the lowest 3 dimensions for host tensors with rank > 4");
        } else {
            working_tensor = operations::experimental::quasar::detail::apply_leading_dimension_padding(
                working_tensor, working_padding, value, use_multicore, sub_core_grids);
            if (std::all_of(working_padding.begin(), working_padding.end(), [](auto& p) {
                    return p.before_elements == 0 && p.after_elements == 0;
                })) {
                if (memory_config_arg.has_value()) {
                    return ttnn::operations::experimental::quasar::to_memory_config(
                        working_tensor, memory_config_arg.value());
                }
                return working_tensor;
            }
        }
    }

    if (working_tensor.layout() == ttnn::TILE_LAYOUT) {
        return operations::experimental::quasar::detail::invoke_tile(
            working_tensor, working_padding, value, use_multicore, memory_config_arg, sub_core_grids);
    }
    return operations::experimental::quasar::detail::invoke_rm(
        working_tensor, working_padding, value, use_multicore, memory_config_arg, sub_core_grids);
}

ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using PadSpecDim = operations::experimental::quasar::PadSpecDim;
    ttsl::SmallVector<PadSpecDim> padding_impl;
    std::transform(padding.begin(), padding.end(), std::back_inserter(padding_impl), [](auto& p) {
        return PadSpecDim(p[0], p[1]);
    });

    return pad(input_tensor, padding_impl, value, use_multicore, memory_config_arg, sub_core_grids);
}

ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttnn::Array4D& output_padded_shape,
    const ttnn::Array4D& input_tensor_start,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using PadSpecDim = operations::experimental::quasar::PadSpecDim;
    ttsl::SmallVector<PadSpecDim> padding_impl;
    const auto& log_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < output_padded_shape.size(); ++i) {
        padding_impl.emplace_back(
            input_tensor_start.at(i), output_padded_shape.at(i) - log_shape[i] - input_tensor_start.at(i));
    }

    return pad(input_tensor, padding_impl, value, use_multicore, memory_config_arg, sub_core_grids);
}

}  // namespace ttnn::operations::experimental::quasar
