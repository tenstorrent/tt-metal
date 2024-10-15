// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"

#include "ttnn/operations/data_movement/pad/device/pad_op.hpp"

namespace ttnn::operations::data_movement {

namespace {

template <typename ShapeType>
static ttnn::Tensor pad_impl(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ShapeType & output_padded_shape,
    const ShapeType & input_tensor_start,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {

    // on host
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        if (input_tensor.get_legacy_shape() == output_padded_shape) {
            return input_tensor;
        }
        else {
            return input_tensor.pad(tt::tt_metal::LegacyShape(output_padded_shape), ttnn::SimpleShape(input_tensor_start), value);
        }
    }
    // on device
    else {
        const auto input_tensor_shape = input_tensor.get_shape();
        const auto rank = input_tensor_shape.rank();

        TT_FATAL(rank == 4, "Tensor rank is not 4");

        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        auto output_tensor = operation::run(
            Pad{tt::tt_metal::LegacyShape(output_padded_shape), ttnn::SimpleShape(input_tensor_start), value, memory_config, use_multicore},
            {input_tensor}, {}, {}, queue_id).front();

        return output_tensor;
    }
}

template <typename ShapeType>
static ttnn::Tensor pad_impl(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    std::vector<std::pair<uint32_t, uint32_t>> padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {

    const int original_rank = input_tensor.get_shape().rank();
    if(int diff = original_rank - padding.size(); diff != 0) {
        TT_FATAL(diff > 0, "ttnn.pad: padding len can't be larger than input tensor rank");

        padding.insert(padding.begin(), diff, {0, 0});
    }

    TT_FATAL(
        padding.size() == original_rank,
        "ttnn.pad: padding must be the same length as the input tensor rank");

    // Unsqueeze Tensor to 4D if it is not already
    ttnn::Tensor input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);


    padding.insert(padding.begin(), 4 - original_rank, {0, 0});
    auto input_shape_with_tile_padding = input_tensor_4D.get_shape().with_tile_padding();

    ShapeType output_padded_shape;
    for(size_t i = 0; i < padding.size(); i++) {
        output_padded_shape[i] = padding[i].first + input_shape_with_tile_padding[i] + padding[i].second;
    }

    auto pad_front = padding | std::views::transform([](const auto& p) { return p.first; });
    auto pad_back = padding | std::views::transform([](const auto& p) { return p.second; });

    const bool front_padding_is_zero = std::accumulate(pad_front.begin(), pad_front.end(), 0) == 0;
    if (input_tensor.get_layout() == ttnn::TILE_LAYOUT) {
        TT_FATAL(
            front_padding_is_zero,
            "ttnn.pad: on device tile padding does not support front padding");
    }

    if (input_tensor.get_layout() == ttnn::TILE_LAYOUT) {
        const int target_height = output_padded_shape[padding.size() - 2];
        const int target_width = output_padded_shape[padding.size() - 1];
        TT_FATAL(
            target_height % ttnn::TILE_SIZE == 0 || target_width % ttnn::TILE_SIZE == 0,
            "ttnn.pad: for tiled tensors padding end must be a multiple of the tile size on height and width for a "
            "tensor in tile layout");
    }

    // Performing actual padding
    ShapeType pad_front_array;
    for(size_t i = 0; i < pad_front.size(); i++) {
        pad_front_array[i] = pad_front[i];
    }

    return pad_impl<ShapeType>(queue_id, input_tensor_4D, output_padded_shape, pad_front_array, value, use_multicore, memory_config_arg);
}

} // anonymous namespace

// This function signature is similar to pytorch's signature
// Any rank tensor supported
ttnn::Tensor ExecutePad::invoke(
    uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const std::vector<std::pair<uint32_t, uint32_t>>& padding,
    const float value,
    const bool use_multicore,
    const std::optional<MemoryConfig>& memory_config_arg) {
    const int original_rank = input_tensor.get_shape().rank();

    ttnn::Tensor output_tensor;
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        switch (original_rank) {
            case 1: output_tensor = pad_impl<tt::tt_metal::Array1D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg); break;
            case 2: output_tensor = pad_impl<tt::tt_metal::Array2D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg); break;
            case 3: output_tensor = pad_impl<tt::tt_metal::Array3D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg); break;
            case 4: output_tensor = pad_impl<tt::tt_metal::Array4D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg); break;
            case 5: output_tensor = pad_impl<tt::tt_metal::Array5D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg); break;
            case 6: output_tensor = pad_impl<tt::tt_metal::Array6D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg); break;
            case 7: output_tensor = pad_impl<tt::tt_metal::Array7D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg); break;
            case 8: output_tensor = pad_impl<tt::tt_metal::Array8D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg); break;
            default: TT_THROW("Unsupported tensor rank of {}. Needs to be between 1 and 8 inclusively.", original_rank);
        }
    }
    else {
        output_tensor =  pad_impl<tt::tt_metal::Array4D>(queue_id, input_tensor, padding, value, use_multicore, memory_config_arg);
    }
    // output_tensor is currently 4D. We have to squeeze back to the original rank
    auto to_vec = [](const auto& arr) {return std::vector<uint32_t>(arr.begin(), arr.end());};
    auto shape = to_vec(output_tensor.get_shape().value);
    auto padded_shape = to_vec(output_tensor.get_shape().with_tile_padding().value);
    if (auto rank_diff = shape.size() - original_rank; rank_diff) {
        auto remove_first_elements = [](auto& source, size_t n) {
            source.erase(source.begin(), source.begin() + n);
        };
        remove_first_elements(shape, rank_diff);
        remove_first_elements(padded_shape, rank_diff);
        auto squeezedShape = ttnn::Shape(tt::tt_metal::LegacyShape(shape, padded_shape));
        output_tensor = ttnn::reshape(output_tensor, squeezedShape);
    }

    // Padding always turns the intended shape to the shape with tile padding. For simplicity of the operation
    output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(padded_shape));

    return output_tensor;
}

#define PAD_OVERLOAD_DIM_IMPL(ShapeType) ttnn::Tensor ExecutePad::invoke(\
    uint8_t queue_id,\
    const ttnn::Tensor& input_tensor,\
    const ShapeType& output_padded_shape,\
    const ShapeType& input_tensor_start,\
    const float value,\
    const bool use_multicore,\
    const std::optional<MemoryConfig>& memory_config_arg) {\
    return pad_impl<ShapeType>(\
        queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config_arg);\
}\
\
ttnn::Tensor ExecutePad::invoke(\
    const ttnn::Tensor& input_tensor,\
    const ShapeType& output_padded_shape,\
    const ShapeType& input_tensor_start,\
    const float value,\
    const std::optional<MemoryConfig>& memory_config_arg) {\
\
    return pad_impl<ShapeType>(DefaultQueueId, input_tensor, output_padded_shape, input_tensor_start, value, false, memory_config_arg);\
}\
\
ttnn::Tensor ExecutePad::invoke(\
    const ttnn::Tensor& input_tensor,\
    const ShapeType& output_padded_shape,\
    const ShapeType& input_tensor_start,\
    const float value) {\
\
    return pad_impl<ShapeType>(DefaultQueueId, input_tensor, output_padded_shape, input_tensor_start, value, false, std::nullopt);\
}

PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array1D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array2D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array3D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array4D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array5D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array6D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array7D)
PAD_OVERLOAD_DIM_IMPL(tt::tt_metal::Array8D)

} // ttnn::operations::data_movement
