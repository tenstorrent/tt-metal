// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/small_vector.hpp>
#include <ttnn/tensor/xtensor/conversion_utils.hpp>
#include <vector>
#include <xtensor/views/xstrided_view.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::xtensor {
namespace detail {

// Helper to compute the type of an unowned strided view over an `xt::xexpression`.
template <typename Expression>
auto compute_strided_view() -> decltype(xt::strided_view(
    std::declval<const xt::xexpression<Expression>&>().derived_cast(), std::declval<xt::xstrided_slice_vector>()));

}  // namespace detail

template <typename Expression>
using StridedView = decltype(detail::compute_strided_view<Expression>());

template <typename Expression>
using StridedViews = std::vector<StridedView<Expression>>;

// IMPORTANT: `chunk` and `concatenate` are not yet part of the public ttnn API.
//
// Splits an xtensor expression into chunks along the specified dimension, and returns a vector of un-owned strided
// views.
template <typename Expression>
StridedViews<Expression> chunk(const xt::xexpression<Expression>& expr, int num_chunks, int dim = 0);

// Overload that performs multi-dimensional chunking.
// Chunking is done in row-major order relative to the supplied `dims`, and returned in the vector in the same order.
// `num_chunks` and `dims` must have the same length. When empty, no chunking is performed, and the entire tensor is
// returned as a single chunk.
template <typename Expression>
StridedViews<Expression> chunk_ndim(
    const xt::xexpression<Expression>& expr,
    const tt::stl::SmallVector<int>& num_chunks,
    const tt::stl::SmallVector<int>& dims);

// Concatenates a list of tensors along the specified dimension.
template <typename Expression>
XtensorAdapter<typename Expression::value_type> concat(const std::vector<Expression>& v, int dim = 0);

// Overload that performs multi-dimensional concatenation.
// `expressions` are assumed to be laid out in row-major order relative to the supplied `dims`.
template <typename Expression>
XtensorAdapter<typename Expression::value_type> concat_ndim(
    const std::vector<Expression>& expressions,
    const tt::stl::SmallVector<int>& num_chunks,
    const tt::stl::SmallVector<int>& dims);

// Overload in terms of `Tensor`.
// Deprecated: Use high-level APIs defined in distributed_tensor.hpp
tt::tt_metal::Tensor concat(const std::vector<tt::tt_metal::Tensor>& tensors, int dim = 0);

}  // namespace ttnn::experimental::xtensor
