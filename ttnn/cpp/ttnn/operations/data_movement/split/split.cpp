// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/split/split.hpp"
#include "ttnn/operations/data_movement/split/device/split_op.hpp"
#include "ttnn/operations/data_movement/view/view.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

constexpr auto TWO_CHUNKS = 2;
constexpr auto RANK_FOUR = 4;

std::vector<Tensor> impl_split_last_dim_two_chunks_tiled(const Tensor& input_tensor, const MemoryConfig& mem_config) {
    const auto& input_shape = input_tensor.padded_shape();
    auto padded_input_shape = ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(input_shape);
    ttnn::operations::experimental::auto_format::FormatParams input_format_params = {
        .pad_shape = padded_input_shape, .pad_value = 0.0, .target_layout = Layout::TILE};
    return tt::tt_metal::operation::run_with_autoformat(
        SplitDeviceOperation{2, 3, mem_config}, {input_tensor}, {input_format_params}, {Layout::TILE, Layout::TILE});
}

std::vector<Tensor> split_last_dim_two_chunks_tiled(const Tensor& input_tensor, const MemoryConfig& mem_config) {
    const auto& shape = input_tensor.padded_shape();
    const bool pre_post_reshape = shape[0] > 1;

    if (!pre_post_reshape) {
        return impl_split_last_dim_two_chunks_tiled(input_tensor, mem_config);
    }

    const int W = 1, Z = shape[0] * shape[1], Y = shape[2], X = shape[3];
    const Tensor& reshaped_tensor =
        ttnn::reshape_on_device(input_tensor, ttnn::SmallVector<int32_t>{1, -1, Y, X}, mem_config);

    auto part_reshaped = impl_split_last_dim_two_chunks_tiled(reshaped_tensor, mem_config);

    std::vector<Tensor> results;
    results.reserve(part_reshaped.size());
    for (auto& part : part_reshaped) {
        results.emplace_back(
            ttnn::reshape_on_device(part, ttnn::SmallVector<int32_t>{-1, (int32_t)shape[1], Y, X / 2}, mem_config));
    }

    return results;
}

std::vector<ttnn::Tensor> split_with_slice_impl(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const ttnn::SmallVector<int64_t>& split_sizes,
    const int32_t dim,
    const MemoryConfig& memory_config) {
    const auto& input_shape = input_tensor.logical_shape();

    // torch requires split size to sum to dim size but since we are using slice we can be more permissive.
    TT_FATAL(
        std::accumulate(split_sizes.begin(), split_sizes.end(), 0L) >= input_shape[dim],
        "Split sizes should sum to at least dimension size. Split sizes: {} dimension {}",
        split_sizes,
        input_shape[dim]);
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
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& split_sizes,
    const int64_t dim = 0,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());

    TT_FATAL(
        std::all_of(split_sizes.begin(), split_sizes.end(), [](const auto& x) { return x > 0; }),
        "split_size should be greater than 0, instead got: {}",
        split_sizes);
    const auto& input_shape = input_tensor.logical_shape();

    // special case to use hardcoded kernel for two chunks sometimes
    if (split_sizes.size() == detail::TWO_CHUNKS && dim == input_shape.rank() - 1 &&
        input_tensor.layout() == Layout::TILE && input_shape.rank() >= 2 &&
        input_shape[-2] / tt::constants::TILE_HEIGHT >= 2 && input_shape[-1] / tt::constants::TILE_WIDTH >= 2) {
        ttnn::Tensor input_tensor_4d;
        if (input_shape.rank() > detail::RANK_FOUR) {
            input_tensor_4d = squeeze_from_ND_to_4D(input_tensor);
        } else if (input_shape.rank() < detail::RANK_FOUR) {
            input_tensor_4d = core::unsqueeze_to_4D(input_tensor);
        } else {
            input_tensor_4d = std::move(input_tensor);
        }
        const auto outputs_4d = detail::split_last_dim_two_chunks_tiled(input_tensor_4d, memory_config);
        std::vector<ttnn::Tensor> outputs;
        outputs.reserve(detail::TWO_CHUNKS);
        for (const auto& t : outputs_4d) {
            ttnn::SmallVector<uint32_t> final_shape(input_shape.cbegin(), input_shape.cend());
            final_shape.back() = t.logical_shape()[-1];
            outputs.emplace_back(ttnn::view(t, ttnn::Shape(final_shape)));
        }
        return outputs;

    } else {
        return detail::split_with_slice_impl(queue_id, input_tensor, split_sizes, dim, memory_config);
    }
}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& split_sizes,
    const int64_t dim = 0,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
    return SplitOperation::invoke(DefaultQueueId, input_tensor, split_sizes, dim, memory_config_arg);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const int64_t split_size,
    const int64_t dim = 0,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
    auto memory_config = memory_config_arg.value_or(input_tensor.memory_config());

    const auto num_chunks =
        std::ceil(static_cast<float>(input_tensor.logical_shape()[dim]) / static_cast<float>(split_size));

    const ttnn::SmallVector<int64_t> split_sizes(num_chunks, split_size);
    return SplitOperation::invoke(queue_id, input_tensor, split_sizes, dim, memory_config);
}

std::vector<ttnn::Tensor> SplitOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const int64_t split_size,
    const int64_t dim = 0,
    const std::optional<MemoryConfig>& memory_config_arg = std::nullopt) {
    return SplitOperation::invoke(DefaultQueueId, input_tensor, split_size, dim, memory_config_arg);
}

}  // namespace ttnn::operations::data_movement
