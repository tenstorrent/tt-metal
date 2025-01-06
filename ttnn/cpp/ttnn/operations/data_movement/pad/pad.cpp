// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_op.hpp"

namespace ttnn::operations::data_movement {

namespace {

template <typename ArrayType>
bool eq_spans(const ArrayType& a, const ArrayType& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end());
}

ttnn::Shape update_original_shape(ttnn::Shape padded_shape, ttnn::Shape input_shape) {
    std::vector<uint32_t> updated_shape;
    size_t input_rank = input_shape.rank();
    for (size_t i = 0; i < input_rank - 2; i++) {
        updated_shape.push_back(input_shape[i]);
    }
    updated_shape.push_back(padded_shape[-2]);
    updated_shape.push_back(padded_shape[-1]);
    return ttnn::Shape(updated_shape);
}

static ttnn::Tensor pad_impl(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    std::span<const uint32_t> output_padded_shape,
    std::span<const uint32_t> input_tensor_start,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    auto input_logical_shape = input_tensor.logical_shape().view();
    // on host
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        if (eq_spans(input_logical_shape, output_padded_shape)) {
            return input_tensor;
        } else {
            return input_tensor.pad(
                ttnn::SimpleShape(output_padded_shape), ttnn::SimpleShape{input_tensor_start}, value);
        }
    }

    // on device
    else {
        auto input_tensor_shape = input_tensor.get_shape();
        const auto rank = input_tensor_shape.rank();

        TT_FATAL(rank == 4, "ttnn.pad: input tensor passed to pad_impl must have rank == 4, but got rank {}.", rank);

        using ShardStrategy = ttnn::operations::data_movement::ShardStrategy;
        using ShardOrientation = tt::tt_metal::ShardOrientation;
        using Layout = tt::tt_metal::Layout;

        auto output_memory_config = memory_config_arg.value_or(input_tensor.memory_config());

        if (input_tensor.is_sharded()) {
            auto total_height = [](const auto& shape) {
                return std::accumulate(shape.begin(), shape.end() - 1, 1, std::multiplies<uint32_t>());
            };

            auto height_distinct = [&total_height](const auto& shape, const auto& other_shape) {
                return total_height(shape) != total_height(other_shape);
            };

            auto width_distinct = [](const auto& shape, const auto& other_shape) { return shape[3] != other_shape[3]; };

            uint32_t input_w = input_logical_shape[3];
            uint32_t output_w = output_padded_shape[3];

            if (width_distinct(input_logical_shape, output_padded_shape)) {
                std::array<uint32_t, 4> output_shape_width_padded{
                    input_logical_shape[0], input_logical_shape[1], input_logical_shape[2], output_w};
                auto width_pad_memory_config = create_sharded_memory_config(
                    ttnn::SimpleShape{output_shape_width_padded},
                    input_tensor.shard_spec()->grid,  // reuse input cores for now: FIXME: can we do better?
                                                      // it's complicated because we need the input shards to be local
                                                      // to the core holding the output shard currently.
                    ShardStrategy::HEIGHT,            // stay height sharded
                    ShardOrientation::ROW_MAJOR);
                output_memory_config = width_pad_memory_config;

                if (height_distinct(input_logical_shape, output_padded_shape)) {
                    // we will decompose the padding into two parts and run two
                    // separate pads.
                    ttnn::SmallVector<uint32_t> adjusted_input_tensor_start{0, 0, 0, input_tensor_start[3]};

                    TT_ASSERT(
                        not(height_distinct(input_logical_shape, output_shape_width_padded) and
                            width_distinct(input_logical_shape, output_shape_width_padded)),
                        "infinite recursion");

                    // pad width
                    auto output_tensor_width_padded = pad_impl(
                        queue_id,
                        input_tensor,
                        output_shape_width_padded,
                        adjusted_input_tensor_start,
                        value,
                        use_multicore,
                        width_pad_memory_config);

                    TT_ASSERT(
                        not(height_distinct(output_padded_shape, output_shape_width_padded) and
                            width_distinct(output_padded_shape, output_shape_width_padded)),
                        "infinite recursion");

                    auto height_pad_memory_config = create_sharded_memory_config(
                        ttnn::SimpleShape{output_padded_shape},
                        input_tensor.shard_spec()->grid,
                        ShardStrategy::HEIGHT,
                        ShardOrientation::ROW_MAJOR);

                    // then pad height
                    auto output_tensor_height_padded = pad_impl(
                        queue_id,
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

        auto output_w = output_padded_shape[3];
        TT_ASSERT(
            !input_tensor.is_sharded() || output_w == output_memory_config.shard_spec->shape[1],
            "output_w != output_memory_config.shard_spec().shape[1]");

        ttnn::SimpleShape output_shape{output_padded_shape};
        auto output_tensor = operation::run(
                                 Pad{output_shape,
                                     output_shape,
                                     ttnn::SimpleShape{input_tensor_start},
                                     value,
                                     output_memory_config,
                                     use_multicore},
                                 {input_tensor},
                                 {},
                                 {},
                                 queue_id)
                                 .front();

        return output_tensor;
    }
}

static ttnn::Tensor pad_impl(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    const int original_rank = input_tensor.get_shape().rank();
    if (int diff = original_rank - padding.size(); diff != 0) {
        TT_FATAL(diff > 0, "ttnn.pad: padding len can't be larger than input tensor rank");

        padding.insert(padding.begin(), diff, {0, 0});
    }

    TT_FATAL(padding.size() == original_rank, "ttnn.pad: padding must be the same length as the input tensor rank");

    // Unsqueeze Tensor to 4D if it is not already
    ttnn::Tensor input_tensor_4D;
    if (input_tensor.get_shape().rank() < 4) {
        input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
    } else if (input_tensor.get_shape().rank() > 4) {
        input_tensor_4D = squeeze_from_ND_to_4D(input_tensor);
    } else {
        input_tensor_4D = input_tensor;
    }
    size_t padding_size = 4;
    size_t extra_index = input_tensor.get_shape().rank() - 4;
    if (input_tensor.get_shape().rank() < 4) {
        padding.insert(padding.begin(), 4 - original_rank, {0, 0});
        padding_size = padding.size();
        extra_index = 0;
    }
    auto input_shape_with_tile_padding = input_tensor_4D.get_shape().with_tile_padding();
    std::vector<uint32_t> output_padded_shape(padding_size, 0);
    for (size_t i = 0; i < padding_size; i++) {
        output_padded_shape[i] =
            padding[i + extra_index].first + input_shape_with_tile_padding[i] + padding[i + extra_index].second;
    }

    auto pad_front = padding | std::views::transform([](const auto& p) { return p.first; });
    auto pad_back = padding | std::views::transform([](const auto& p) { return p.second; });

    const bool front_padding_is_zero = std::accumulate(pad_front.begin(), pad_front.end(), 0) == 0;
    if (input_tensor.get_layout() == ttnn::TILE_LAYOUT) {
        TT_FATAL(front_padding_is_zero, "ttnn.pad: on device tile padding does not support front padding");
    }

    if (input_tensor.get_layout() == ttnn::TILE_LAYOUT) {
        const int target_height = output_padded_shape[padding_size - 2];
        const int target_width = output_padded_shape[padding_size - 1];
        TT_FATAL(
            target_height % ttnn::TILE_SIZE == 0 || target_width % ttnn::TILE_SIZE == 0,
            "ttnn.pad: for tiled tensors padding end must be a multiple of the tile size on height and width for a "
            "tensor in tile layout");
    }

    // Performing actual padding
    std::vector<uint32_t> pad_front_array(padding_size, 0);
    for (size_t i = 0; i < pad_front.size(); i++) {
        pad_front_array[i] = pad_front[i];
    }

    return pad_impl(
        queue_id, input_tensor_4D, output_padded_shape, pad_front_array, value, use_multicore, memory_config_arg);
}

}  // anonymous namespace

// This function signature is similar to pytorch's signature
// Any rank tensor supported

ttnn::Tensor ExecutePad::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    tt::stl::Span<const std::pair<uint32_t, uint32_t>> padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    const int original_rank = input_tensor.get_shape().rank();
    ttnn::SmallVector<std::pair<uint32_t, uint32_t>> padding_vec(padding.begin(), padding.end());

    ttnn::Tensor output_tensor =
        pad_impl(queue_id, input_tensor, std::move(padding_vec), value, use_multicore, memory_config_arg);
    // output_tensor is currently 4D. We have to squeeze back to the original rank
    if (original_rank <= 4) {
        auto to_vec = [](const auto& arr) { return ttnn::SmallVector<uint32_t>{arr.begin(), arr.end()}; };
        auto output_shape = to_vec(output_tensor.get_shape().value);
        auto padded_shape = to_vec(output_tensor.get_shape().with_tile_padding().value);
        if (const auto rank_diff = output_shape.size() - original_rank; rank_diff) {
            auto remove_prefix = [](auto& source, size_t n) { source.erase(source.begin(), source.begin() + n); };
            remove_prefix(output_shape, rank_diff);
            remove_prefix(padded_shape, rank_diff);
            auto squeezedShape = ttnn::Shape(tt::tt_metal::LegacyShape(output_shape, padded_shape));
            output_tensor = ttnn::reshape(output_tensor, squeezedShape);
            output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(padded_shape));
        }
    } else {
        auto to_vec = [](const auto& arr) { return ttnn::SmallVector<uint32_t>{arr.begin(), arr.end()}; };
        auto output_shape = to_vec(output_tensor.get_shape().value);
        auto padded_shape = to_vec(output_tensor.get_shape().with_tile_padding().value);
        auto squeezedShape = ttnn::Shape(tt::tt_metal::LegacyShape(output_shape, padded_shape));
        output_tensor = ttnn::reshape(output_tensor, update_original_shape(squeezedShape, input_tensor.get_shape()));
    }

    // Padding always turns the intended shape to the shape with tile padding. For simplicity of the operation

    return output_tensor;
}

#define PAD_OVERLOAD_DIM_IMPL(ShapeType)                                                                               \
    ttnn::Tensor ExecutePad::invoke(                                                                                   \
        uint8_t queue_id,                                                                                              \
        const ttnn::Tensor& input_tensor,                                                                              \
        const ShapeType& output_padded_shape,                                                                          \
        const ShapeType& input_tensor_start,                                                                           \
        const float value,                                                                                             \
        const bool use_multicore,                                                                                      \
        const std::optional<MemoryConfig>& memory_config_arg) {                                                        \
        return pad_impl(                                                                                               \
            queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config_arg); \
    }                                                                                                                  \
                                                                                                                       \
    ttnn::Tensor ExecutePad::invoke(                                                                                   \
        const ttnn::Tensor& input_tensor,                                                                              \
        const ShapeType& output_padded_shape,                                                                          \
        const ShapeType& input_tensor_start,                                                                           \
        const float value,                                                                                             \
        const std::optional<MemoryConfig>& memory_config_arg) {                                                        \
        return pad_impl(                                                                                               \
            DefaultQueueId, input_tensor, output_padded_shape, input_tensor_start, value, false, memory_config_arg);   \
    }                                                                                                                  \
                                                                                                                       \
    ttnn::Tensor ExecutePad::invoke(                                                                                   \
        const ttnn::Tensor& input_tensor,                                                                              \
        const ShapeType& output_padded_shape,                                                                          \
        const ShapeType& input_tensor_start,                                                                           \
        const float value) {                                                                                           \
        return pad_impl(                                                                                               \
            DefaultQueueId, input_tensor, output_padded_shape, input_tensor_start, value, false, std::nullopt);        \
    }

PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array1D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array2D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array3D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array4D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array5D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array6D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array7D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array8D)

}  // namespace ttnn::operations::data_movement
