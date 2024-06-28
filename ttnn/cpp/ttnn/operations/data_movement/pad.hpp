// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/types.hpp"
#include "tt_eager/tt_dnn/op_library/concat/concat_op.hpp"
#include "tt_eager/tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_eager/tt_dnn/op_library/permute/permute_op.hpp"
#include "tt_eager/tt_dnn/op_library/repeat/repeat_op.hpp"
#include "tt_eager/tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_eager/tt_dnn/op_library/upsample/upsample_op.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {


struct Pad {
    static inline const std::array<TensorSchema, 1> input_tensor_schemas() {
        return {ttnn::TensorSchema{
            2,  // min rank
            4,  // max rank
            {ttnn::bfloat16, ttnn::bfloat8_b, ttnn::uint16, ttnn::int32, ttnn::uint32},
            {ttnn::TILE_LAYOUT},
            true,   // can_be_on_device
            false,  // can_be_on_cpu
            false,  // can_be_scalar
            false   // is_optional}
        }};
    }

    template <typename... Args>
    static auto input_tensors_to_validate(const ttnn::Tensor& input_tensor, Args&&... args) {
        return std::make_tuple(input_tensor);
    }

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        std::vector<std::pair<uint32_t, uint32_t>> padding, //intentionally not const&
        const float value,
        const std::optional<MemoryConfig>& memory_config_arg) {

        const int original_rank = input_tensor.get_shape().rank();
        if(int diff = original_rank - padding.size(); diff != 0) {
            TT_FATAL(diff > 0, "ttnn.pad: padding len can't be larger than input tensor rank");

            padding.insert(padding.begin(), diff, {0, 0});
        }

        TT_FATAL(
            padding.size() == original_rank,
            "ttnn.pad: padding must be the same length as the input tensor rank");
        TT_FATAL(
            input_tensor.get_layout() != ttnn::ROW_MAJOR_LAYOUT,
            "ttnn.pad: row-major tensors have to use fallback because the kernel currently causes a PCC error");

        // Unsqueeze Tensor to 4D if it is not already
        ttnn::Tensor input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);        
        padding.insert(padding.begin(), 4 - original_rank, {0, 0});
        auto input_shape_with_tile_padding = input_tensor_4D.get_shape().with_tile_padding();
        std::vector<uint32_t> output_padded_shape(padding.size());
        for(size_t i = 0; i < padding.size(); i++) {
            output_padded_shape[i] = input_shape_with_tile_padding[i] + padding[i].second;
        }

        // Due to the strangeness of tt::tt_metal::pad, we need to split front and back pad
        // Front will be passed separately. And pad_back is retrieved -> output_padded_shape - pad_front
        auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
        auto pad_front = padding | std::views::transform([](const auto& p) { return p.first; });
        auto pad_back = padding | std::views::transform([](const auto& p) { return p.second; });

        const bool front_padding_is_zero = std::accumulate(pad_front.begin(), pad_front.end(), 0) == 0;
        TT_FATAL(
            front_padding_is_zero,
            "ttnn.pad: on device padding does not support front padding");

        const int target_height = output_padded_shape[output_padded_shape.size() - 2];
        const int target_width = output_padded_shape[output_padded_shape.size() - 1];
        TT_FATAL(
            target_height % ttnn::TILE_SIZE == 0 || target_width % ttnn::TILE_SIZE == 0,
            "ttnn.pad: for tiled tensors padding end must be a multiple of the tile size on height and width for a "
            "tensor in tile layout");

        // Performing actual padding        
        std::vector<uint32_t> pad_front_vec(pad_front.begin(), pad_front.end());
        auto output_tensor = operation::run(
            tt::tt_metal::Pad{
                .output_tensor_shape=tt::tt_metal::Shape(output_padded_shape),
                .input_tensor_start=tt::tt_metal::Shape(pad_front_vec),
                .pad_value=value,
                .output_mem_config=memory_config,
                .use_multicore=true
            },
            {input_tensor_4D}).front();


        // output_tensor is currently 4D. We have to squeeze back to the original rank
        auto to_vec = [](const auto& arr) {return std::vector<uint32_t>(arr.begin(), arr.end());};
        auto shape = to_vec(output_tensor.get_shape().value());
        auto padded_shape = to_vec(output_tensor.get_shape().with_tile_padding().value());
        if (auto rank_diff = shape.size() - original_rank; rank_diff) {
            auto remove_first_elements = [](auto& source, size_t n) {
                source.erase(source.begin(), source.begin() + n);
            };
            remove_first_elements(shape, rank_diff);
            remove_first_elements(padded_shape, rank_diff);
            auto squeezedShape = ttnn::Shape(tt::tt_metal::Shape(shape, padded_shape));
            output_tensor = ttnn::reshape(output_tensor, squeezedShape);
        }

        // Padding always turns the intended shape to the shape with tile padding. For simplicity of the operation
        output_tensor = ttnn::reshape(output_tensor, ttnn::Shape(padded_shape));

        return output_tensor;
    }
};

}  // namespace data_movement
}  // namespace operations

constexpr auto pad = ttnn::register_operation<ttnn::operations::data_movement::Pad>("ttnn::pad");

}  // namespace ttnn
