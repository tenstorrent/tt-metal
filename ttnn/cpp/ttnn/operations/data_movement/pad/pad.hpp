// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_eager/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/core.hpp"

#include "tt_eager/tt_dnn/op_library/run_operation.hpp"

#include "device/pad_op.hpp"

#include <ranges>


namespace ttnn {
namespace operations {
namespace data_movement {

constexpr uint8_t DefaultQueueId = 0;
constexpr std::size_t NUM_DIMENSIONS = 4;


struct ExecutePad {


    static ttnn::Tensor _execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const std::array<uint32_t, NUM_DIMENSIONS> & output_padded_shape,
        const std::array<uint32_t, NUM_DIMENSIONS> & input_tensor_start,
        const float value,
        const bool use_multicore,
        const std::optional<MemoryConfig>& memory_config_arg) {

        uint32_t rank = input_tensor.shape().rank();
        // on host
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            if (input_tensor.get_legacy_shape() == output_padded_shape) {
                return input_tensor;
            }
            else {
                return input_tensor.pad(tt::tt_metal::Shape(output_padded_shape, rank), tt::tt_metal::Shape(input_tensor_start, rank), value);
            }
        }
        // on device
        else {
            const auto input_tensor_shape = input_tensor.get_shape();
            const auto rank = input_tensor_shape.rank();
            if (rank != 4) {
                TT_FATAL("Tensor rank is not 4");
            }

            auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());
            auto output_tensor = operation::run(
                Pad{tt::tt_metal::Shape(output_padded_shape, rank), tt::tt_metal::Shape(input_tensor_start, rank), value, memory_config, use_multicore},
                {input_tensor}, {}, {}, queue_id).front();

            return output_tensor;
        }
    }

    static ttnn::Tensor execute_on_worker_thread(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const std::array<uint32_t, NUM_DIMENSIONS> & output_padded_shape,
        const std::array<uint32_t, NUM_DIMENSIONS> & input_tensor_start,
        const float value,
        const bool use_multicore,
        const std::optional<MemoryConfig>& memory_config_arg) {

        return _execute_on_worker_thread(queue_id, input_tensor, output_padded_shape, input_tensor_start, value, use_multicore, memory_config_arg);

    }


    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const std::array<uint32_t, NUM_DIMENSIONS> & output_padded_shape,
        const std::array<uint32_t, NUM_DIMENSIONS> & input_tensor_start,
        const float value
        ) {

        return _execute_on_worker_thread(0, input_tensor, output_padded_shape, input_tensor_start, value, false, std::nullopt);

    }


    // This function signature is similar to pytorch's signature
    // Any rank tensor supported
    static ttnn::Tensor execute_on_worker_thread(
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
        TT_FATAL(
            input_tensor.get_layout() != ttnn::ROW_MAJOR_LAYOUT,
            "ttnn.pad: row-major tensors have to use fallback because the kernel currently causes a PCC error");

        // Unsqueeze Tensor to 4D if it is not already
        ttnn::Tensor input_tensor_4D = ttnn::unsqueeze_to_4D(input_tensor);
        padding.insert(padding.begin(), 4 - original_rank, {0, 0});
        auto input_shape_with_tile_padding = input_tensor_4D.get_shape().with_tile_padding();
        std::array<uint32_t, NUM_DIMENSIONS> output_padded_shape;
        for(size_t i = 0; i < padding.size(); i++) {
            output_padded_shape[i] = input_shape_with_tile_padding[i] + padding[i].second;
        }

        auto pad_front = padding | std::views::transform([](const auto& p) { return p.first; });
        auto pad_back = padding | std::views::transform([](const auto& p) { return p.second; });

        const bool front_padding_is_zero = std::accumulate(pad_front.begin(), pad_front.end(), 0) == 0;
        TT_FATAL(
            front_padding_is_zero,
            "ttnn.pad: on device padding does not support front padding");

        const int target_height = output_padded_shape[padding.size() - 2];
        const int target_width = output_padded_shape[padding.size() - 1];
        TT_FATAL(
            target_height % ttnn::TILE_SIZE == 0 || target_width % ttnn::TILE_SIZE == 0,
            "ttnn.pad: for tiled tensors padding end must be a multiple of the tile size on height and width for a "
            "tensor in tile layout");

        // Performing actual padding
        std::array<uint32_t, NUM_DIMENSIONS> pad_front_array;
        for(size_t i = 0; i < pad_front.size(); i++) {
            pad_front_array[i] = pad_front[i];
        }


        auto output_tensor = _execute_on_worker_thread(queue_id, input_tensor_4D, output_padded_shape, pad_front_array, value, use_multicore, memory_config_arg);


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

constexpr auto pad = ttnn::register_operation<ttnn::operations::data_movement::ExecutePad>("ttnn::pad");

}  // namespace ttnn
