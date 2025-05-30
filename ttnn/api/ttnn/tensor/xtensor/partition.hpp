// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <xtensor/views/xstrided_view.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/xtensor/xtensor_all_includes.hpp"

namespace ttnn::experimental::xtensor {
namespace detail {

template <typename XtExpr>
auto chunk_xexpression(XtExpr& expr, int num_chunks, int dim) {
    using StridedView = decltype(xt::strided_view(expr, std::declval<xt::xstrided_slice_vector>()));

    TT_FATAL(num_chunks > 0, "num_chunks must be > 0; got num_chunks: {}", num_chunks);
    TT_FATAL(
        dim >= 0 && dim < expr.dimension(),
        "invalid dimension index; got dim: {}, tensor dimension: {}",
        dim,
        expr.dimension());

    if (num_chunks == 1) {
        xt::xstrided_slice_vector indices(expr.dimension(), xt::all());
        return std::vector<StridedView>{xt::strided_view(expr, indices)};
    }

    const int size_along_dim = static_cast<int>(expr.shape()[dim]);
    const int chunk_size = (size_along_dim + num_chunks - 1) / num_chunks;
    int remaining_size = size_along_dim;

    std::vector<StridedView> chunk_views;
    chunk_views.reserve(static_cast<std::size_t>(num_chunks));
    int start = 0;
    int end = 0;
    for (int i = 0; i < num_chunks && end < size_along_dim; ++i) {
        int current_chunk_size = std::min(chunk_size, remaining_size);
        remaining_size -= current_chunk_size;
        end = start + current_chunk_size;

        // Build indices for slicing
        xt::xstrided_slice_vector indices(expr.dimension(), xt::all());
        indices[dim] = xt::range(start, end);

        auto chunk_view = xt::strided_view(expr, indices);
        chunk_views.push_back(chunk_view);
        start = end;
    }

    return chunk_views;
}

}  // namespace detail

// IMPORTANT: `chunk` and `concatenate` are not yet part of the public ttnn API.
//
// Splits a tensor into chunks along the specified dimension.
std::vector<tt::tt_metal::Tensor> chunk(const tt::tt_metal::Tensor& tensor, int num_chunks, int dim = 0);

// Overload for `xt::xexpression`.
template <typename T>
auto chunk(const xt::xexpression<T>& expr, int num_chunks, int dim = 0) {
    return detail::chunk_xexpression(expr.derived_cast(), num_chunks, dim);
}

// Concatenates a list of tensors along the specified dimension.
tt::tt_metal::Tensor concat(const std::vector<tt::tt_metal::Tensor>& tensors, int dim = 0);

// Overload for `xt::xarray`.
template <typename T>
xt::xarray<T> concat(const std::vector<xt::xarray<T>>& v, int dim = 0);

}  // namespace ttnn::experimental::xtensor
