// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/tensor/types.hpp"
#include "ttnn/experimental/tt_dnn/op_library/concat/concat_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/pad/pad_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/repeat/repeat_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/composite/composite_ops.hpp"
#include "ttnn/experimental/tt_dnn/op_library/upsample/upsample_op.hpp"

#include "ttnn/cpp/ttnn/operations/core.hpp"

#include <ranges>

namespace ttnn {
namespace operations {
namespace data_movement {

struct UpSample {

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        std::variant<int, std::array<int, 2>, std::array<int, 3>, std::array<int, 4>> scale_factor,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(ttnn::DRAM_MEMORY_CONFIG);

        int scale_h = 1;
        int scale_w = 1;
        std::visit(
            [&scale_h, &scale_w](auto&& sf) {
                using T = std::decay_t<decltype(sf)>;
                if constexpr (std::is_same_v<T, int>) {
                    scale_h = sf;
                    scale_w = sf;
                } else if constexpr (std::is_same_v<T, std::array<int, 2>>) {
                    scale_w = sf.at(0);
                    int scale_c = sf.at(1);
                    TT_FATAL(scale_c == 1);
                } else if constexpr (std::is_same_v<T, std::array<int, 3>>) {
                    scale_h = sf.at(0);
                    scale_w = sf.at(1);
                    int scale_c = sf.at(2);
                    TT_FATAL(scale_c == 1);
                } else if constexpr (std::is_same_v<T, std::array<int, 4>>) {
                    int scale_n = sf.at(0);
                    scale_h = sf.at(1);
                    scale_w = sf.at(2);
                    int scale_c = sf.at(3);
                    TT_FATAL(scale_n == 1);
                    TT_FATAL(scale_c == 1);
                } else {
                    // static_assert(false, "Unsupported scale factor");
                    static_assert(sizeof(T) != 0, "Type check failed.");
                }
            },
            scale_factor);

        // DEBUG
        // fmt::print("scale_h: {}, scale_w: {}\n", scale_h, scale_w);

        if (input_tensor.is_sharded()) {
            // TT_FATAL(not input_tensor.is_sharded());
            int shard_height = input_tensor.memory_config().shard_spec.value().shape[0];
            const auto batch_size = input_tensor.get_shape()[0];
            const auto input_h = input_tensor.get_shape()[1];
            const auto input_w = input_tensor.get_shape()[2];
            const auto num_channels = input_tensor.get_shape()[3];
            if (shard_height % input_w != 0) {
                TT_FATAL(shard_height % input_w != 0);
            }
        }

        return tt::tt_metal::upsample(input_tensor, scale_h, scale_w, mem_config);
    }
};

struct Repeat {

    static ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        const ttnn::Shape& shape,
        std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat(input_tensor, shape.value(), mem_config);
        return output_tensor;
    }
};

struct RepeatInterleave {

    // # This operation does not support the following cases:
    // #   - Shape([2[32], 2[32]]) -> repeats = 2, dim = 0
    // #   - Shape([2[32], 2[32]]) -> repeats = Tensor[1,2], dim = 1
    static ttnn::Tensor execute_on_worker_thread(const ttnn::Tensor& input_tensor,
                                                 uint32_t repeats,
                                                 int32_t dim,
                                                 std::optional<MemoryConfig> output_mem_config = std::nullopt) {
        MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
        auto output_tensor = tt::tt_metal::repeat_interleave(input_tensor, repeats, dim, mem_config);
        return output_tensor;
    }
};


<<<<<<< HEAD
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
=======
>>>>>>> origin/main

}  // namespace data_movement
}  // namespace operations

constexpr auto upsample = ttnn::register_operation<ttnn::operations::data_movement::UpSample>("ttnn::upsample");
constexpr auto repeat = ttnn::register_operation<ttnn::operations::data_movement::Repeat>("ttnn::repeat");
constexpr auto repeat_interleave = ttnn::register_operation<ttnn::operations::data_movement::RepeatInterleave>("ttnn::repeat_interleave");

}  // namespace ttnn
