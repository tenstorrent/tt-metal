// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/fill_pad/device/fill_pad_device_operation.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_device_operation.hpp"
#include "ttnn/operations/data_movement/pad/codegen/pad_codegen_supported.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_device_operation.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operation.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <ttnn/tensor/types.hpp>

#include "pad.hpp"
#include "pad_force.hpp"

namespace ttnn::operations::data_movement::detail {

enum class PadRoute { Auto, Native, Codegen };

// Shared body of every entry; recursive paths re-enter here with the same route, never through ttnn::pad.
ttnn::Tensor pad_with_route(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding,
    float value,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids,
    PadRoute route);

namespace {

inline bool is_width_or_block_sharded(const MemoryConfig& mc) {
    const auto layout = mc.memory_layout();
    return mc.is_sharded() &&
           (layout == TensorMemoryLayout::BLOCK_SHARDED || layout == TensorMemoryLayout::WIDTH_SHARDED);
}

// Route through to_memory_config composite only when native kernels cannot handle the input.
//
// All L1 WIDTH/BLOCK sharded inputs are handled natively:
//   - TILE layout: tile factories use TensorAccessor page_id addressing, resolved transparently
//     by the Device 2.0 API for any sharded buffer.
//   - RM layout: noc_async_*_sharded in the default RM factory iterates across per-row pages
//     regardless of tile alignment; the front_padding kernel branch reads into a temp buffer
//     and memmoves to the correct L1 offset, so width front-pad is also handled natively.
//
// DRAM-sharded W/B inputs use to_memory_config → pad → (optional) to_memory_config/interleaved_to_sharded.
// sharded_to_interleaved is L1-only, so to_memory_config is used instead (mirrors repeat.cpp).
inline bool needs_pad_composite_fallback(const ttnn::Tensor& input_tensor) {
    if (!is_width_or_block_sharded(input_tensor.memory_config())) {
        return false;
    }
    return !input_tensor.memory_config().is_l1();  // DRAM-sharded edge case only
}

ttnn::Tensor pad_via_interleaved_composite(
    const ttnn::Tensor& input_tensor,
    std::span<const uint32_t> output_padded_shape,
    std::span<const uint32_t> input_tensor_start,
    const float value,
    const bool use_multicore,
    const MemoryConfig& output_memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    MemoryConfig interleaved_config{TensorMemoryLayout::INTERLEAVED, input_tensor.memory_config().buffer_type()};
    // sharded_to_interleaved is L1-only; DRAM-sharded W/B inputs use to_memory_config (mirrors repeat.cpp).
    auto interleaved_input = ttnn::to_memory_config(input_tensor, interleaved_config, std::nullopt);

    MemoryConfig working_output =
        output_memory_config.is_sharded()
            ? MemoryConfig{TensorMemoryLayout::INTERLEAVED, output_memory_config.buffer_type()}
            : output_memory_config;

    ttnn::Shape out_shape{output_padded_shape};
    ttnn::Shape start{input_tensor_start};
    auto padded = ttnn::prim::pad(
        interleaved_input,
        out_shape,
        out_shape,
        start,
        value,
        working_output,
        use_multicore,
        std::nullopt,
        sub_core_grids);

    if (output_memory_config.is_sharded()) {
        if (output_memory_config.is_l1()) {
            return ttnn::interleaved_to_sharded(padded, output_memory_config, std::nullopt);
        }
        return ttnn::to_memory_config(padded, output_memory_config, std::nullopt);
    }
    return padded;
}

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
    const std::optional<CoreRangeSet>& sub_core_grids,
    const PadRoute route) {
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

    auto reshaped = ttnn::reshape(input_tensor, ttnn::Shape({before, middle, shape[-2], shape[-1]}));

    const uint32_t inner_extent =
        std::accumulate(shape_view.begin() + dim + 1, shape_view.begin() + rank - 2, 1u, std::multiplies<uint32_t>());

    ttsl::SmallVector<PadSpecDim> padding_4d = {
        {0, 0}, {pad_spec.before_elements * inner_extent, pad_spec.after_elements * inner_extent}, {0, 0}, {0, 0}};

    auto padded = pad_with_route(reshaped, padding_4d, value, use_multicore, std::nullopt, sub_core_grids, route);

    ttsl::SmallVector<uint32_t> output_shape(shape.view().begin(), shape.view().end());
    output_shape[dim] += pad_spec.before_elements + pad_spec.after_elements;
    return ttnn::reshape(padded, ttnn::Shape(output_shape));
}

ttnn::Tensor pad_leading_dimensions(
    const ttnn::Tensor& input_tensor,
    ttsl::SmallVector<PadSpecDim>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const PadRoute route) {
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
        result =
            pad_leading_dimension_via_reshape(result, dim, padding[dim], value, use_multicore, sub_core_grids, route);
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
    const std::optional<CoreRangeSet>& sub_core_grids,
    const PadRoute route) {
    return pad_leading_dimensions(input_tensor, padding, value, use_multicore, sub_core_grids, route);
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

    if (needs_pad_composite_fallback(input_tensor)) {
        return pad_via_interleaved_composite(
            input_tensor,
            output_padded_shape,
            input_tensor_start,
            value,
            use_multicore,
            output_memory_config,
            sub_core_grids);
    }

    if (input_tensor.is_sharded() &&
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED &&
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
    return ttnn::prim::pad(
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

ttnn::Tensor to_4d(const ttnn::Tensor& tensor) {
    const auto rank = tensor.logical_shape().rank();
    if (rank < 4) {
        return ttnn::unsqueeze_to_4D(tensor);
    }
    if (rank > 4) {
        return squeeze_from_ND_to_4D(tensor);
    }
    return tensor;
}

// After-padding of the trailing four dims of a rank-`rank` tensor, as the 4D tensor from to_4d sees it. `padding` is
// right-aligned to the dims and may omit leading ones. Dims below rank - 4 are consumed before the squeeze, so they
// have no slot here.
std::array<uint32_t, 4> trailing_4d_after_padding(const ttsl::SmallVector<PadSpecDim>& padding, int rank) {
    const int skipped = rank - static_cast<int>(padding.size());
    std::array<uint32_t, 4> after{};
    for (int d = std::max(0, rank - 4); d < rank; ++d) {
        if (d >= skipped) {
            after[d + 4 - rank] = padding[d - skipped].after_elements;
        }
    }
    return after;
}

ttnn::Tensor pad_impl(
    const ttnn::Tensor& input_tensor,
    ttsl::SmallVector<PadSpecDim> padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt) {
    if (input_tensor.dtype() == DataType::BFLOAT8_B && input_tensor.layout() == Layout::TILE) {
        auto bfloat16_tensor = ttnn::typecast(input_tensor, DataType::BFLOAT16);
        auto padded_tensor =
            pad_impl(bfloat16_tensor, padding, value, use_multicore, memory_config_arg, sub_core_grids);
        return ttnn::typecast(padded_tensor, DataType::BFLOAT8_B);
    }
    const int original_rank = input_tensor.logical_shape().rank();

    TT_FATAL(padding.size() == original_rank, "ttnn.pad: padding must be the same length as the input tensor rank");

    const ttnn::Tensor input_tensor_4D = to_4d(input_tensor);
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
            output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(output_shape), ttnn::Shape(padded_shape));
            output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(padded_shape));
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
        output_tensor = ttnn::reshape(output_tensor, output_logical_shape, output_logical_shape);
    }
    return output_tensor;
}

// Pads a TILE tensor through codegen; nullopt means the caller should take the native path.
std::optional<ttnn::Tensor> pad_tile_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding_vec,
    const ttnn::Shape& requested_logical_shape,
    const ttnn::Shape& requested_padded_shape,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const PadRoute route) {
    namespace pc = ttnn::operations::data_movement::pad_codegen;
    const auto output_mem_config = memory_config_arg.value_or(input_tensor.memory_config());
    // Declined before to_4d: on a sharded tensor below rank 4 it reshards through DRAM, which the native fallback
    // would then pay for nothing, since the gate rejects sharded I/O.
    if (route == PadRoute::Auto && (!pc::supported_execution_controls(use_multicore, sub_core_grids) ||
                                    input_tensor.memory_config().is_sharded() || output_mem_config.is_sharded())) {
        return std::nullopt;
    }
    const auto rank = static_cast<int>(input_tensor.logical_shape().rank());
    const ttnn::Tensor input_4d = to_4d(input_tensor);
    const auto after = trailing_4d_after_padding(padding_vec, rank);

    const auto planned = pc::plan_codegen(input_4d, after, value, output_mem_config);
    if (route == PadRoute::Codegen) {
        TT_FATAL(planned.has_value(), "ttnn.pad: codegen path does not support this input");
    } else if (!planned.has_value() || pc::is_demoted(input_4d, *planned) || !pc::fits_live_l1(input_4d, *planned)) {
        return std::nullopt;
    }
    const auto& params = *planned;

    // Planned on the block-float input because only there does the gate see that the pad composes, and it then
    // budgets the bfloat16 copy and output this branch allocates.
    if (pc::composes_via_bf16(input_4d, params)) {
        const ttnn::Tensor decoded = ttnn::typecast(input_4d, DataType::BFLOAT16);
        const auto bf16_params = pc::make_codegen_params(decoded, after, value, output_mem_config);
        auto padded = ttnn::prim::pad_codegen(decoded, bf16_params);
        auto encoded = ttnn::typecast(padded, input_tensor.dtype());
        return ttnn::experimental::view(encoded, requested_logical_shape, requested_padded_shape);
    }

    // The pad kernel copies whole tiles, so the unused lanes of partial border tiles are filled first, with the value
    // native's fill_implicit_tile_padding writes. Relied on: prim::fill_pad fills in place, so its memory config has no
    // effect; it folds every dim above rank 3 into num_slice_batches, so the 4D view is equivalent; and the gate admits
    // a partial tile only for interleaved non-block-float inputs, for which the wrapper would add just a rank-3 view,
    // so the prim is called directly (measured cheaper on small shapes). test_pad_codegen_matches_native_bitwise pins
    // the result.
    ttnn::Tensor source = input_4d;
    if (pc::has_partial_tile(input_4d)) {
        // fill_pad casts a float straight to a 32-bit integer, which is undefined out of range; the packed word is
        // the clamped value the pad kernel writes, so border lanes and padding hold the same bits.
        const bool int32_input = input_4d.dtype() == DataType::INT32 || input_4d.dtype() == DataType::UINT32;
        const ttnn::PadValue fill_value = int32_input ? ttnn::PadValue{params.packed_pad_value} : ttnn::PadValue{value};
        source = ttnn::prim::fill_pad(input_4d, fill_value, output_mem_config);
    }
    auto output = ttnn::prim::pad_codegen(source, params);
    return ttnn::experimental::view(output, requested_logical_shape, requested_padded_shape);
}

ttnn::Tensor invoke_tile(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding_vec,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const PadRoute route) {
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

    if (route != PadRoute::Native) {
        if (auto output = pad_tile_codegen(
                input_tensor,
                padding_vec,
                requested_logical_shape,
                requested_padded_shape,
                value,
                use_multicore,
                memory_config_arg,
                sub_core_grids,
                route)) {
            return *output;
        }
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
            output_tensor = ttnn::to_memory_config(output_tensor, memory_config_arg.value(), std::nullopt);
        } else {
            // memory_config_arg is nullopt → condition can only be true if input is sharded
            // (interleaved input + nullopt config → both sides false → condition false)
            // so input_tensor.shard_spec()->grid is safe here.
            const auto sharded_mem_config = create_sharded_memory_config(
                ttnn::Shape{requested_logical_shape},
                input_tensor.shard_spec()->grid,
                ShardStrategy::HEIGHT,
                ShardOrientation::ROW_MAJOR);
            output_tensor = ttnn::to_memory_config(output_tensor, sharded_mem_config, std::nullopt);
        }
    }
    return output_tensor;
}

ttnn::Tensor pad_with_route(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const PadRoute route) {
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
            working_tensor = apply_leading_dimension_padding(
                working_tensor, working_padding, value, use_multicore, sub_core_grids, route);
            if (std::all_of(working_padding.begin(), working_padding.end(), [](auto& p) {
                    return p.before_elements == 0 && p.after_elements == 0;
                })) {
                if (memory_config_arg.has_value()) {
                    return ttnn::to_memory_config(working_tensor, memory_config_arg.value());
                }
                return working_tensor;
            }
        }
    }

    if (working_tensor.layout() == ttnn::TILE_LAYOUT) {
        return invoke_tile(
            working_tensor, working_padding, value, use_multicore, memory_config_arg, sub_core_grids, route);
    }
    return invoke_rm(working_tensor, working_padding, value, use_multicore, memory_config_arg, sub_core_grids);
}

namespace {

ttsl::SmallVector<PadSpecDim> to_pad_spec(const ttsl::SmallVector<std::array<uint32_t, 2>>& padding) {
    ttsl::SmallVector<PadSpecDim> spec;
    std::transform(
        padding.begin(), padding.end(), std::back_inserter(spec), [](const auto& p) { return PadSpecDim{p[0], p[1]}; });
    return spec;
}

void require_codegen_supported(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<PadSpecDim>& padding,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    namespace pc = ttnn::operations::data_movement::pad_codegen;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "ttnn.pad: codegen path needs a device tensor");
    TT_FATAL(
        pc::supported_execution_controls(use_multicore, sub_core_grids),
        "ttnn.pad: codegen path does not support use_multicore=False or sub_core_grids");
    TT_FATAL(
        padding.size() <= input_tensor.logical_shape().rank(),
        "ttnn.pad: padding len can't be larger than input tensor rank");
    // L1 fit is checked per dispatched step, so a rank > 4 input can still fail after its leading-dim pads.
    TT_FATAL(
        pc::in_codegen_scope(to_4d(input_tensor), memory_config_arg.value_or(input_tensor.memory_config())),
        "ttnn.pad: codegen path does not support this input");
}

}  // namespace

ttnn::Tensor pad_force_native(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return pad_with_route(
        input_tensor, to_pad_spec(padding), value, use_multicore, memory_config_arg, sub_core_grids, PadRoute::Native);
}

ttnn::Tensor pad_force_codegen(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    const auto spec = to_pad_spec(padding);
    require_codegen_supported(input_tensor, spec, use_multicore, memory_config_arg, sub_core_grids);
    return pad_with_route(
        input_tensor, spec, value, use_multicore, memory_config_arg, sub_core_grids, PadRoute::Codegen);
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

// This function signature is similar to pytorch's signature
// Any rank tensor supported
ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<operations::data_movement::PadSpecDim>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return operations::data_movement::detail::pad_with_route(
        input_tensor,
        padding,
        value,
        use_multicore,
        memory_config_arg,
        sub_core_grids,
        operations::data_movement::detail::PadRoute::Auto);
}

ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttsl::SmallVector<std::array<uint32_t, 2>>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using PadSpecDim = operations::data_movement::PadSpecDim;
    ttsl::SmallVector<PadSpecDim> padding_impl;
    std::transform(padding.begin(), padding.end(), std::back_inserter(padding_impl), [](auto& p) {
        return PadSpecDim(p[0], p[1]);
    });

    return ttnn::pad(input_tensor, padding_impl, value, use_multicore, memory_config_arg, sub_core_grids);
}

ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const ttnn::Array4D& output_padded_shape,
    const ttnn::Array4D& input_tensor_start,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using PadSpecDim = operations::data_movement::PadSpecDim;
    ttsl::SmallVector<PadSpecDim> padding_impl;
    const auto& log_shape = input_tensor.logical_shape();
    for (uint32_t i = 0; i < output_padded_shape.size(); ++i) {
        padding_impl.emplace_back(
            input_tensor_start.at(i), output_padded_shape.at(i) - log_shape[i] - input_tensor_start.at(i));
    }

    return ttnn::pad(input_tensor, padding_impl, value, use_multicore, memory_config_arg, sub_core_grids);
}

}  // namespace ttnn
