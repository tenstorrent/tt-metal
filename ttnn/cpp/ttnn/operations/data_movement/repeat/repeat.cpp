// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>

#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/sharded_to_interleaved.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "device/repeat_device_operation.hpp"
#include "repeat.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

struct UpperRepeatDims {
    static constexpr uint32_t collapsed_upper = 0;
    static constexpr uint32_t repeat = 1;
    static constexpr uint32_t collapsed_lower = 2;
    static constexpr uint32_t page_size = 3;
};
struct LastRepeatDims {
    static constexpr uint32_t collapsed_upper = 0;
    static constexpr uint32_t repeat = 1;
};

ttnn::Tensor repeat_upper_dims_rm(
    const ttnn::Tensor& tensor, const uint32_t dim, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
    // collapse upper dims to 4D or append 1s
    // collapse lower dims or insert 1s
    // op
    // un-collaps to expected size

    // figure out the shape of the input tensor for the op. dims before and after rep dim get collapsed, not including
    // page size.
    const auto& input_shape = tensor.logical_shape();
    ttnn::SmallVector<uint32_t> collapsed_shape_vector(4);

    collapsed_shape_vector[UpperRepeatDims::collapsed_upper] =
        std::accumulate(input_shape.cbegin(), input_shape.cbegin() + dim, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[UpperRepeatDims::repeat] = input_shape[dim];
    collapsed_shape_vector[UpperRepeatDims::collapsed_lower] =
        std::accumulate(input_shape.cbegin() + dim + 1, input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[UpperRepeatDims::page_size] = input_shape[-1];

    // use ttnn::view to check logic
    auto input_tensor = ttnn::view(tensor, ttnn::Shape(collapsed_shape_vector));

    constexpr bool is_final_dim = false;
    auto out_tensor = ttnn::prim::repeat(input_tensor, repetitions, is_final_dim, output_mem_config);
    auto expected_shape = input_shape;
    expected_shape[dim] *= repetitions;

    return ttnn::view(out_tensor, ttnn::Shape(expected_shape));
}

ttnn::Tensor repeat_last_dim_rm(
    const ttnn::Tensor& tensor, const uint32_t repetitions, const MemoryConfig& output_mem_config) {
    // collapse to 2D
    // op
    // un-collapse
    const auto& input_shape = tensor.logical_shape();
    ttnn::SmallVector<uint32_t> collapsed_shape_vector(2);

    collapsed_shape_vector[0] =
        std::accumulate(input_shape.cbegin(), input_shape.cend() - 1, 1, std::multiplies<uint32_t>());
    collapsed_shape_vector[1] = input_shape[-1];

    // use ttnn:view
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
}  // namespace detail

ttnn::Tensor RepeatOperation::invoke(
    const ttnn::Tensor& tensor,
    const ttnn::SmallVector<uint32_t>& provided_repetition_vector,
    const std::optional<MemoryConfig>& provided_output_mem_config) {
    auto [working_tensor, repetition_vector] = detail::match_input_rank(tensor, provided_repetition_vector);
    MemoryConfig output_mem_config = provided_output_mem_config.value_or(tensor.memory_config());
    auto working_output_mem_config = output_mem_config;

    if (std::any_of(repetition_vector.cbegin(), repetition_vector.cend(), [](auto x) { return x == 0; })) {
        const auto& shape = working_tensor.logical_shape();
        std::transform(
            shape.cbegin(),
            shape.cend(),
            repetition_vector.cbegin(),
            repetition_vector.begin(),
            std::multiplies<uint32_t>());
        return ttnn::reshape(tensor, ttnn::Shape(repetition_vector));
    }

    TT_FATAL(working_tensor.logical_shape().rank() > 0, "repeat does not support rank 0 tensors");

    // nothing to do!
    if (std::all_of(repetition_vector.cbegin(), repetition_vector.cend(), [](auto x) { return x == 1; })) {
        return tensor;
    }

    // Sharded -> interleaved
    if (tensor.memory_config().is_sharded()) {
        MemoryConfig working_memory_config{TensorMemoryLayout::INTERLEAVED, tensor.memory_config().buffer_type()};
        working_tensor = ttnn::sharded_to_interleaved(tensor, working_memory_config, std::nullopt);
    }
    if (working_output_mem_config.is_sharded()) {
        working_output_mem_config =
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, working_output_mem_config.buffer_type()};
    }

    // tiled -> RM
    if (working_tensor.layout() == ttnn::TILE_LAYOUT) {
        working_tensor = ttnn::to_layout(working_tensor, ttnn::ROW_MAJOR_LAYOUT);
    }

    // loop over dims in repetition vector, backwards because repeat pages first is faster
    for (auto it = repetition_vector.crbegin(); it != repetition_vector.crend(); ++it) {
        // no op for unit repetitions
        if (*it == 1) {
            continue;
        }
        // if last dim
        if (it == repetition_vector.crbegin()) {
            working_tensor = detail::repeat_last_dim_rm(working_tensor, *it, working_output_mem_config);
        }
        // if not last dim
        else {
            auto i = repetition_vector.crend() - it - 1;  // forward index
            working_tensor = detail::repeat_upper_dims_rm(working_tensor, i, *it, working_output_mem_config);
        }
    }

    // RM -> OG page layout
    if (tensor.layout() == ttnn::TILE_LAYOUT) {
        working_tensor = ttnn::to_layout(working_tensor, ttnn::TILE_LAYOUT, tensor.dtype());
    }

    // Interleaved to OG mem layout
    if (output_mem_config.is_sharded()) {
        working_tensor = ttnn::interleaved_to_sharded(working_tensor, output_mem_config, std::nullopt);
    }

    return working_tensor;
}

ttnn::Tensor RepeatOperation::invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& repeat_dims) {
    return RepeatOperation::invoke(
        input_tensor, SmallVector<uint32_t>(repeat_dims.cbegin(), repeat_dims.cend()), std::nullopt);
}

}  // namespace ttnn::operations::data_movement
