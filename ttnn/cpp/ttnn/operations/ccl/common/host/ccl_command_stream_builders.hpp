// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0
///

#pragma once

#include "ttnn/operations/ccl/common/uops/ccl_command.hpp"

#include <vector>
// #include <cstdint>

namespace tt::tt_metal {
class Tensor;
};

namespace ttnn::ccl::cmd::builder {

std::vector<ttnn::ccl::v2::TensorSlice> generate_tensor_slices(
    size_t num_slices, const tt::tt_metal::Tensor& tensor, size_t split_dim);

ttnn::ccl::v2::TensorSlice convert_to_whole_tensor_slice(const tt::tt_metal::Tensor& tensor);

std::vector<ttnn::ccl::v2::TensorSlice> compute_page_aligned_slices(
    size_t num_slices, const tt::tt_metal::Tensor& input_tensor, size_t split_dim);

// Pairs of slice size and slice offset
std::vector<std::pair<size_t, size_t>> compute_evenly_split_sizes(size_t size, size_t num_slices);

// // Outer vector = per worker command stream, inner vector = commands
std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> split_tensor_slices_across_workers_page_aligned(
    size_t num_workers, std::vector<ttnn::ccl::v2::TensorSlice> const& tensor_slices);

std::vector<ttnn::ccl::v2::TensorSlice> split_tensor_slice_across_workers_wrapped_page_aligned(
    ttnn::ccl::v2::TensorSlice const& tensor_slice, size_t num_workers);

std::vector<std::vector<ttnn::ccl::v2::TensorSlice>> generate_worker_tensor_slices(
    size_t num_slices, const tt::tt_metal::Tensor& tensor, size_t num_workers, size_t split_dim);

};  // namespace ttnn::ccl::cmd::builder
