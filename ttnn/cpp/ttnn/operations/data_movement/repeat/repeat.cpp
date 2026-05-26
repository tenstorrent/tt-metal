// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "device/repeat_device_operation.hpp"
#include "repeat.hpp"

namespace ttnn::operations::data_movement::detail {

struct UpperRepeatDims {
    static constexpr uint32_t collapsed_upper = 0;
    static constexpr uint32_t repeat = 1;
    static constexpr uint32_t collapsed_lower = 2;
    static constexpr uint32_t page_size = 3;
};

ttnn::Tensor repeat_upper_dims_rm(
    const ttnn::Tensor& tensor, const uint32_t dim, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
    const auto& input_shape = tensor.logical_shape();
    ttnn::SmallVector<uint32_t> collapsed_shape_vector(4);

    collapsed_shape_vector[UpperRepeatDims::collapsed_upper] =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + dim, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[UpperRepeatDims::repeat] = input_shape[dim];
    collapsed_shape_vector[UpperRepeatDims::collapsed_lower] =
        std::accumulate(input_shape.cbegin() + dim + 1, input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[UpperRepeatDims::page_size] = input_shape[-1];

    auto input_tensor = ttnn::view(tensor, ttnn::Shape(collapsed_shape_vector));

    constexpr bool is_final_dim = false;
    auto out_tensor = ttnn::prim::repeat(input_tensor, repetitions, is_final_dim, output_mem_config);
    auto expected_shape = input_shape;
    expected_shape[dim] *= repetitions;

    return ttnn::view(out_tensor, ttnn::Shape(expected_shape));
}

ttnn::Tensor repeat_last_dim_rm(
    const ttnn::Tensor& tensor, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
    const auto& input_shape = tensor.logical_shape();
    ttnn::SmallVector<uint32_t> collapsed_shape_vector(2);

    collapsed_shape_vector[0] =
        std::accumulate(input_shape.cbegin(), input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[1] = input_shape[-1];

    auto input_tensor = ttnn::view(tensor, ttnn::Shape(collapsed_shape_vector));

    constexpr bool is_final_dim = true;
    auto out_tensor = ttnn::prim::repeat(input_tensor, repetitions, is_final_dim, output_mem_config);

    auto expected_shape = input_shape;
    expected_shape[-1] *= repetitions;

    return ttnn::view(out_tensor, ttnn::Shape(expected_shape));
}

std::tuple<ttnn::Tensor, ttnn::SmallVector<uint32_t>> match_input_rank(
    const ttnn::Tensor& tensor, const SmallVector<uint32_t>& repetition_vector) {
    auto working_tensor = tensor;
    const auto& input_shape = working_tensor.logical_shape();
    SmallVector<uint32_t> working_repetition_vector;

    const auto total_reps =
        std::accumulate(repetition_vector.cbegin(), repetition_vector.cend(), 1, std::multiplies<uint_fast32_t>());

    if (input_shape.rank() < repetition_vector.size()) {
        ttnn::SmallVector<uint32_t> new_shape_vec(repetition_vector.size(), 1);
        std::copy_backward(input_shape.cbegin(), input_shape.cend(), new_shape_vec.end());
        working_tensor = ttnn::view(working_tensor, ttnn::Shape(new_shape_vec));
        working_repetition_vector = repetition_vector;
    }
    // torch actually throws an error if the repetition rank is smaller than the tensor rank but it seems reasonable to
    // handle it
    else if (repetition_vector.size() < input_shape.rank()) {
        working_repetition_vector.resize(input_shape.rank(), 1);
        std::copy_backward(repetition_vector.cbegin(), repetition_vector.cend(), working_repetition_vector.end());
    }

    else {
        working_repetition_vector = repetition_vector;
    }

    TT_ASSERT(working_tensor.logical_volume() == tensor.logical_volume());
    TT_ASSERT(
        std::accumulate(
            working_repetition_vector.cbegin(),
            working_repetition_vector.cend(),
            1,
            std::multiplies<uint_fast32_t>()) == total_reps);

    return std::tie(working_tensor, working_repetition_vector);
}

bool is_tile_repeat_eligible(const ttnn::Tensor& tensor) {
    if (tensor.layout() != ttnn::TILE_LAYOUT) {
        return false;
    }
    const auto& shape = tensor.logical_shape();
    if (shape.rank() < 2) {
        return false;
    }
    return (shape[-1] % tt::constants::TILE_WIDTH == 0) && (shape[-2] % tt::constants::TILE_HEIGHT == 0);
}

ttnn::Tensor repeat_dim_tile(
    const ttnn::Tensor& tensor, const uint32_t dim, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
    const auto& shape = tensor.logical_shape();
    const auto rank = shape.rank();

    uint32_t h_tiles = shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t w_tiles = shape[-1] / tt::constants::TILE_WIDTH;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    uint32_t tile_page_size = tt::tile_size(cb_data_format);

    uint32_t higher, rep_dim_pages, lower;

    if (dim == rank - 1) {
        // W dimension: each tile-row's w_tiles get repeated
        higher = std::accumulate(shape.cbegin(), shape.cend() - 2, 1u, std::multiplies<uint32_t>()) * h_tiles;
        rep_dim_pages = w_tiles;
        lower = 1;
    } else if (dim == rank - 2) {
        // H dimension: tile-rows get repeated
        higher = std::accumulate(shape.cbegin(), shape.cend() - 2, 1u, std::multiplies<uint32_t>());
        rep_dim_pages = h_tiles;
        lower = w_tiles;
    } else {
        // Upper dimensions (batch, channel, etc.): groups of tiles get repeated
        higher = std::accumulate(shape.cbegin(), shape.cbegin() + dim, 1u, std::multiplies<uint32_t>());
        uint32_t lower_elements =
            std::accumulate(shape.cbegin() + dim + 1, shape.cend() - 2, 1u, std::multiplies<uint32_t>());
        rep_dim_pages = shape[dim];
        lower = lower_elements * h_tiles * w_tiles;
    }

    return ttnn::prim::repeat_tile(
        tensor, repetitions, dim, output_mem_config, higher, rep_dim_pages, lower, tile_page_size);
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

ttnn::Tensor repeat(
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<uint32_t>& repetition_vector,
    const std::optional<MemoryConfig>& memory_config) {
    auto [working_tensor, working_repetition_vector] =
        operations::data_movement::detail::match_input_rank(input_tensor, repetition_vector);
    MemoryConfig output_mem_config = memory_config.value_or(input_tensor.memory_config());
    auto working_output_mem_config = output_mem_config;

    if (std::any_of(
            working_repetition_vector.cbegin(), working_repetition_vector.cend(), [](auto x) { return x == 0; })) {
        const auto& shape = working_tensor.logical_shape();
        std::transform(
            shape.cbegin(),
            shape.cend(),
            working_repetition_vector.cbegin(),
            working_repetition_vector.begin(),
            std::multiplies<uint32_t>());
        return ttnn::reshape(input_tensor, ttnn::Shape(working_repetition_vector));
    }

    TT_FATAL(working_tensor.logical_shape().rank() > 0, "repeat does not support rank 0 tensors");

    // nothing to do!
    if (std::all_of(
            working_repetition_vector.cbegin(), working_repetition_vector.cend(), [](auto x) { return x == 1; })) {
        return input_tensor;
    }

    // Sharded -> interleaved
    if (input_tensor.memory_config().is_sharded()) {
        MemoryConfig working_memory_config{TensorMemoryLayout::INTERLEAVED, input_tensor.memory_config().buffer_type()};
        working_tensor = ttnn::sharded_to_interleaved(input_tensor, working_memory_config, std::nullopt);
    }
    if (working_output_mem_config.is_sharded()) {
        working_output_mem_config =
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, working_output_mem_config.buffer_type()};
    }

    if (operations::data_movement::detail::is_tile_repeat_eligible(working_tensor)) {
        // Tile-native path: operate directly on tiles, skip TILE->RM->TILE conversion
        for (auto it = working_repetition_vector.crbegin(); it != working_repetition_vector.crend(); ++it) {
            if (*it == 1) {
                continue;
            }
            auto dim = working_repetition_vector.crend() - it - 1;
            working_tensor =
                operations::data_movement::detail::repeat_dim_tile(working_tensor, dim, *it, working_output_mem_config);
        }
    } else {
        // ROW_MAJOR path: convert TILE->RM if needed, repeat, convert back
        if (working_tensor.layout() == ttnn::TILE_LAYOUT) {
            working_tensor = ttnn::to_layout(working_tensor, ttnn::ROW_MAJOR_LAYOUT);
        }

        for (auto it = working_repetition_vector.crbegin(); it != working_repetition_vector.crend(); ++it) {
            if (*it == 1) {
                continue;
            }
            if (it == working_repetition_vector.crbegin()) {
                working_tensor = operations::data_movement::detail::repeat_last_dim_rm(
                    working_tensor, *it, working_output_mem_config);
            } else {
                auto i = working_repetition_vector.crend() - it - 1;
                working_tensor = operations::data_movement::detail::repeat_upper_dims_rm(
                    working_tensor, i, *it, working_output_mem_config);
            }
        }

        if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
            working_tensor = ttnn::to_layout(working_tensor, ttnn::TILE_LAYOUT, input_tensor.dtype());
        }
    }

    // Interleaved to OG mem layout
    if (output_mem_config.is_sharded()) {
        working_tensor = ttnn::interleaved_to_sharded(working_tensor, output_mem_config, std::nullopt);
    }

    return working_tensor;
}

ttnn::Tensor repeat(const ttnn::Tensor& input_tensor, const ttnn::Shape& repeat_dims) {
    return ttnn::repeat(input_tensor, SmallVector<uint32_t>(repeat_dims.cbegin(), repeat_dims.cend()), std::nullopt);
}

}  // namespace ttnn
