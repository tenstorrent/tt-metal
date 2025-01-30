// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/constants.hpp"
// #include "ttnn/operations/core/core.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

std::vector<ttnn::Tensor> split_with_slice_impl(
    const uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int64_t>& split_sizes,
    const int32_t dim,
    const MemoryConfig& memory_config) {
    const auto& input_shape = input_tensor.get_logical_shape();

    // torch requires split size to sum to dim size but since we are using slice we can be more permissive.
    TT_FATAL(
        std::accumulate(split_sizes.begin(), split_sizes.end(), 0) >= input_shape[dim],
        "Split sizes should sum to at least dimension size");
    std::vector<ttnn::Tensor> results;
    results.reserve(split_sizes.size());

    const ttnn::SmallVector<const int32_t> steps(input_shape.rank(), 1);
    ttnn::SmallVector<int32_t> begins(input_shape.rank(), 0), ends(input_shape.cbegin(), input_shape.cend());
    const tt::stl::Span<const int32_t> sbegins(begins), ssteps(steps), sends(ends);

    ends[dim] = 0;
    for (const auto& s : split_sizes) {
        ends[dim] = std::min(static_cast<uint32_t>(ends[dim] + s), input_shape[dim]);
        results.emplace_back(ttnn::slice(queue_id, input_tensor, sbegins, sends, ssteps, memory_config));
        begins[dim] += s;
    }

    return results;
}
}  // namespace detail

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& split_sizes,
    const int64_t& dim = 0,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());

    TT_FATAL(
        std::all_of(split_sizes.begin(), split_sizes.end(), [](const auto& x) { return x > 0; }),
        "split_size should be greater than 0 ");

    return detail::split_with_slice_impl(queue_id, input_tensor, split_sizes, dim, memory_config);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& split_sizes,
    const int64_t& dim = 0,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
    return SplitOperation::invoke(DefaultQueueId, input_tensor, split_sizes, dim, memory_config_arg);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const uint8_t queue_id,
    const ttnn::Tensor& input_tensor,
    const int64_t& split_size,
    const int64_t& dim = 0,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());

    const auto num_chunks =
        std::ceil(static_cast<float>(input_tensor.get_logical_shape()[dim]) / static_cast<float>(split_size));
    const ttnn::SmallVector<int64_t> split_sizes(num_chunks, split_size);
    return SplitOperation::invoke(queue_id, input_tensor, split_sizes, dim, memory_config);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const int64_t& split_size,
    const int64_t& dim = 0,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
    return SplitOperation::invoke(DefaultQueueId, input_tensor, split_size, dim, memory_config_arg);
}

}  // namespace ttnn::operations::data_movement
