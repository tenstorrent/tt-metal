// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>

#include <xtensor/views/xstrided_view.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/xtensor/xtensor_all_includes.hpp"

namespace ttnn::experimental::xtensor {
namespace detail {

template <typename XtExpr>
auto chunk_xexpression(XtExpr& expr, tt::stl::SmallVector<int> chunk_sizes, tt::stl::SmallVector<int> dims) {
    using StridedView = decltype(xt::strided_view(expr, std::declval<xt::xstrided_slice_vector>()));
    if (chunk_sizes.empty()) {
        xt::xstrided_slice_vector indices(expr.dimension(), xt::all());
        return std::vector<StridedView>{xt::strided_view(expr, indices)};
    }

    TT_FATAL(chunk_sizes.size() == dims.size(), "chunk_sizes and dims must have the same size");
    TT_FATAL(std::is_sorted(dims.begin(), dims.end()), "dims must be sorted");
    TT_FATAL(std::unique(dims.begin(), dims.end()) == dims.end(), "dims must be unique");
    TT_FATAL(
        std::all_of(chunk_sizes.begin(), chunk_sizes.end(), [](size_t size) { return size > 0; }),
        "chunk_sizes must be > 0; got chunk_sizes: {}",
        chunk_sizes);
    TT_FATAL(
        std::all_of(dims.begin(), dims.end(), [&expr](size_t dim) { return dim >= 0 && dim < expr.dimension(); }),
        "invalid dimension index; got dims: {}, tensor dimension: {}",
        dims,
        expr.dimension());

    tt::stl::SmallVector<std::vector<std::pair<int, int>>> dim_ranges;
    tt::stl::SmallVector<size_t> num_chunks_per_dim;
    for (size_t i = 0; i < dims.size(); ++i) {
        int dim = dims[i];
        int num_chunks = chunk_sizes[i];
        int size_along_dim = static_cast<int>(expr.shape()[dim]);
        int chunk_size = (size_along_dim + num_chunks - 1) / num_chunks;

        std::vector<std::pair<int, int>> ranges;
        ranges.reserve(num_chunks);

        int start = 0;
        for (int chunk_idx = 0; chunk_idx < num_chunks && start < size_along_dim; ++chunk_idx) {
            int current_chunk_size = std::min(chunk_size, size_along_dim - start);
            int end = start + current_chunk_size;
            ranges.emplace_back(start, end);
            start = end;
        }
        num_chunks_per_dim.push_back(ranges.size());
        dim_ranges.push_back(std::move(ranges));
    }

    const size_t total_chunks =
        std::accumulate(num_chunks_per_dim.begin(), num_chunks_per_dim.end(), 1, std::multiplies<size_t>());

    std::vector<StridedView> chunk_views;
    tt::stl::SmallVector<size_t> current_indices(dims.size(), 0);
    for (size_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        xt::xstrided_slice_vector indices(expr.dimension(), xt::all());
        for (size_t i = 0; i < dims.size(); ++i) {
            int dim = dims[i];
            auto [start, end] = dim_ranges[i][current_indices[i]];
            indices[dim] = xt::range(start, end);
        }

        auto chunk_view = xt::strided_view(expr, indices);
        chunk_views.push_back(chunk_view);

        for (size_t i = 0; i < dims.size(); ++i) {
            if (++current_indices[i] < num_chunks_per_dim[i]) {
                break;
            }
            current_indices[i] = 0;
        }
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
    return detail::chunk_xexpression(expr.derived_cast(), {num_chunks}, {dim});
}

template <typename T>
auto chunk_ndim(const xt::xexpression<T>& expr, tt::stl::SmallVector<int> chunk_sizes, tt::stl::SmallVector<int> dims) {
    return detail::chunk_xexpression(expr.derived_cast(), chunk_sizes, dims);
}

// Concatenates a list of tensors along the specified dimension.
tt::tt_metal::Tensor concat(const std::vector<tt::tt_metal::Tensor>& tensors, int dim = 0);

// Overload for `xt::xarray`.
template <typename T>
xt::xarray<T> concat(const std::vector<xt::xarray<T>>& v, int dim = 0);

}  // namespace ttnn::experimental::xtensor
