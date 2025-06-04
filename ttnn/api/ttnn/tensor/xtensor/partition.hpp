// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <xtensor/views/xstrided_view.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::xtensor {
namespace detail {

// Helper to compute the type of an unowned strided view over an `xt::xexpression`.
template <typename T>
auto compute_strided_view() -> decltype(xt::strided_view(
    std::declval<const xt::xexpression<T>&>().derived_cast(), std::declval<xt::xstrided_slice_vector>()));

}  // namespace detail

template <typename T>
using StridedView = decltype(detail::compute_strided_view<T>());

template <typename T>
using StridedViews = std::vector<StridedView<T>>;

// IMPORTANT: `chunk` and `concatenate` are not yet part of the public ttnn API.
//
// Splits an xtensor expression into chunks along the specified dimension, and returns a vector of un-owned strided
// views.
// The value of `dim` must not be negative.
template <typename T>
StridedViews<T> chunk(const xt::xexpression<T>& expr, int num_chunks, int dim = 0);

// Overload that performs multi-dimensional chunking.
// Chunking is done in row-major order relative to the supplied `dims`, and returned in the vector in the same order.
// `num_chunks` and `dims` must have the same length. When empty, no chunking is performed, and the entire tensor is
// returned as a single chunk.
// The values of `dims` must not be negative.
template <typename T>
StridedViews<T> chunk_ndim(
    const xt::xexpression<T>& expr, tt::stl::SmallVector<int> num_chunks, tt::stl::SmallVector<int> dims);

// Concatenates a list of tensors along the specified dimension.
tt::tt_metal::Tensor concat(const std::vector<tt::tt_metal::Tensor>& tensors, int dim = 0);

// Overload for `xt::xarray`.
template <typename T>
xt::xarray<T> concat(const std::vector<xt::xarray<T>>& v, int dim = 0);

}  // namespace ttnn::experimental::xtensor
