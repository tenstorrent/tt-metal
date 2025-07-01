// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include <tt-metalium/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

#include <ranges>
#include <vector>
#include <cstdint>

namespace ttnn::ccl::cmd::builder {

std::vector<ttnn::ccl::v2::TensorSlice> generate_tensor_slices(
    const size_t num_slices, const Tensor& tensor, size_t split_dim) {
    return compute_page_aligned_slices(num_slices, tensor, split_dim);
}

ttnn::ccl::v2::TensorSlice convert_to_whole_tensor_slice(const Tensor& tensor) {
    return compute_page_aligned_slices(1, tensor, 0).at(0);
}

// TENSOR MANIP
// Pairs of slice size and slice offset
std::vector<std::pair<size_t, size_t>> compute_evenly_split_sizes(size_t size, size_t num_slices) {
    const int64_t num_larger_slices_total = size % num_slices;
    const bool evenly_divisible = num_larger_slices_total == 0;
    const int64_t smaller_slice_size = size / num_slices;
    const int64_t larger_slice_size = smaller_slice_size + !evenly_divisible;

    auto compute_slice_dim_size =
        [larger_slice_size, smaller_slice_size, num_larger_slices_total](int64_t slice_index) {
            bool is_larger_slice = slice_index < num_larger_slices_total;
            return is_larger_slice ? larger_slice_size : smaller_slice_size;
        };

    auto compute_slice_offset = [num_larger_slices_total, larger_slice_size, smaller_slice_size](int64_t slice_index) {
        int64_t num_larger_slices = std::min(slice_index, num_larger_slices_total);
        int64_t num_smaller_slices = std::min(slice_index - num_larger_slices, 0L);
        return num_larger_slices * larger_slice_size + (slice_index - num_larger_slices) * smaller_slice_size;
    };

    auto compute_slice_size_and_offset = [compute_slice_dim_size,
                                          compute_slice_offset](size_t slice_index) -> std::pair<size_t, size_t> {
        return {compute_slice_dim_size(slice_index), compute_slice_offset(slice_index)};
    };
    auto result = std::vector<std::pair<size_t, size_t>>{};
    result.reserve(num_slices);
    for (size_t i = 0; i < num_slices; i++) {
        result.push_back(compute_slice_size_and_offset(i));
    }
    return std::vector<std::pair<size_t, size_t>>(result.begin(), result.end());
}

// // Outer vector = per worker command stream, inner vector = commands
std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> split_tensor_slices_across_workers_page_aligned(
    size_t num_workers, std::vector<ttnn::ccl::v2::TensorSlice> const& tensor_slices) {
    TT_FATAL(tensor_slices.size() > 0, "Number of slices must be greater than 0");
    // not split up across workers yet

    auto worker_slices_streams = std::vector<std::vector<ttnn::ccl::v2::TensorSlice>>(num_workers);
    std::ranges::for_each(
        worker_slices_streams, [&tensor_slices](auto& worker_slices) { worker_slices.reserve(tensor_slices.size()); });

    for (auto const& tensor_slice : tensor_slices) {
        auto const worker_slices = split_tensor_slice_across_workers_wrapped_page_aligned(tensor_slice, num_workers);
        TT_FATAL(
            worker_slices.size() == num_workers,
            "Expected {} worker slices for tensor slice but got {}",
            num_workers,
            worker_slices.size());
        for (size_t i = 0; i < num_workers; i++) {
            worker_slices_streams[i].push_back(worker_slices[i]);
        }
    }
    for (size_t i = 0; i < num_workers; i++) {
        TT_FATAL(
            worker_slices_streams[i].size() == tensor_slices.size(),
            "Mismatch in tensor slices. Expected {} but got {}",
            tensor_slices.size(),
            worker_slices_streams[i].size());
    }

    return worker_slices_streams;
};

static ttnn::ccl::Shape4D<uint32_t> shape_to_shape_in_tiles(const Shape& shape) {
    TT_FATAL(shape.rank() == 4, "Expected 4D shape but got {}", shape.rank());
    ttnn::ccl::Shape4D<uint32_t> shape_in_tiles = {
        shape[0], shape[1], shape[-2] / tt::constants::TILE_HEIGHT, shape[-1] / tt::constants::TILE_WIDTH};
    return shape_in_tiles;
}

std::vector<ttnn::ccl::v2::TensorSlice> split_tensor_slice_across_workers_wrapped_page_aligned(
    ttnn::ccl::v2::TensorSlice const& tensor_slice, size_t num_workers) {
    const size_t num_pages = tensor_slice.tensor_slice_shape.volume();

    auto to_cmd_tensor = [&tensor_slice](std::pair<size_t, size_t> size_offset) {
        auto worker_slice = tensor_slice;
        worker_slice.worker_slice_shape = {1, 1, 1, size_offset.first};
        worker_slice.worker_slice_offset = {0, 0, 0, size_offset.second};
        return worker_slice;
    };

    const auto evenly_split_sizes = compute_evenly_split_sizes(num_pages, num_workers);
    std::vector<ttnn::ccl::v2::TensorSlice> worker_slices;
    worker_slices.reserve(num_workers);
    std::transform(
        evenly_split_sizes.begin(),
        evenly_split_sizes.end(),
        std::back_inserter(worker_slices),
        [to_cmd_tensor](auto size_offset) { return to_cmd_tensor(size_offset); });

    TT_FATAL(
        worker_slices.size() == num_workers, "Expected {} worker slices but got {}", num_workers, worker_slices.size());
    return worker_slices;
}

// Assumed that the tensor_slice shape is in terms of pages, not elements
std::vector<ttnn::ccl::v2::TensorSlice> compute_page_aligned_slices(
    size_t const num_slices, const Tensor& input_tensor, size_t split_dim) {
    TT_FATAL(num_slices > 0, "Number of slices must be greater than 0");
    std::vector<ttnn::ccl::v2::TensorSlice> tensor_slices;

    const auto input_tensor_shape_in_tiles = shape_to_shape_in_tiles(input_tensor.logical_shape());
    tensor_slices.reserve(num_slices);

    // split the input tensor, by shape, into pieces
    ttnn::ccl::v2::TensorSlice reference_tensor = {
        input_tensor_shape_in_tiles,
        input_tensor_shape_in_tiles,
        {0, 0, 0, 0},
        input_tensor_shape_in_tiles,
        {0, 0, 0, 0}};
    auto to_cmd_tensor = [&reference_tensor, split_dim](std::pair<size_t, size_t> size_offset) {
        auto cmd_tensor = reference_tensor;
        cmd_tensor.tensor_slice_shape[split_dim] = size_offset.first;
        cmd_tensor.tensor_slice_offset[split_dim] = size_offset.second;
        return cmd_tensor;
    };

    const auto evenly_split_sizes = compute_evenly_split_sizes(input_tensor_shape_in_tiles[split_dim], num_slices);
    tensor_slices.reserve(evenly_split_sizes.size());
    std::transform(
        evenly_split_sizes.begin(),
        evenly_split_sizes.end(),
        std::back_inserter(tensor_slices),
        [to_cmd_tensor](auto size_offset) { return to_cmd_tensor(size_offset); });

    TT_FATAL(
        tensor_slices.size() == num_slices, "Expected {} tensor slices but got {}", num_slices, tensor_slices.size());

    return tensor_slices;
}

std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> generate_worker_tensor_slices(
    const size_t num_slices, const Tensor& tensor, const size_t num_workers, size_t split_dim) {
    auto tensor_slices = compute_page_aligned_slices(num_slices, tensor, split_dim);
    return split_tensor_slices_across_workers_page_aligned(num_workers, tensor_slices);
}

};  // namespace ttnn::ccl::cmd::builder
