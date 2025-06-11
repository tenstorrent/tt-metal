// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"
#include <ttnn/tensor/xtensor/xtensor_all_includes.hpp>

namespace ttnn::experimental::xtensor {

// Returns the shape of the xtensor as `ttnn::Shape`.
template <typename E>
ttnn::Shape get_shape_from_xarray(const E& xarr) {
    ttnn::SmallVector<uint32_t> shape_dims;
    for (size_t i = 0; i < xarr.shape().size(); ++i) {
        shape_dims.push_back(xarr.shape()[i]);
    }
    return ttnn::Shape(shape_dims);
}

// Converts a span to an xtensor view.
// IMPORTANT: the lifetime of the returned xtensor view is tied to the lifetime of the underlying buffer.
template <typename T>
xt::xarray<T> span_to_xtensor_view(tt::stl::Span<const T> buffer, const ttnn::Shape& shape) {
    std::vector<size_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(buffer.data(), buffer.size(), xt::no_ownership(), shape_vec);
}

// Converts a span to an xtensor view.
// IMPORTANT: the lifetime of the returned xtensor view is tied to the lifetime of the underlying buffer.
template <typename T>
xt::xarray<T> span_to_xtensor_view(std::span<T> buffer, const ttnn::Shape& shape) {
    std::vector<size_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(buffer.data(), buffer.size(), xt::no_ownership(), shape_vec);
}

// Converts an xtensor to a span.
// IMPORTANT: the lifetime of the returned span is tied to the lifetime of the underlying xtensor.
template <typename T>
auto xtensor_to_span(const xt::xarray<T>& xtensor) {
    auto adaptor = xt::adapt(xtensor.data(), xtensor.size(), xt::no_ownership());
    return tt::stl::Span<const T>(adaptor.data(), adaptor.size());
}

// Converts an xtensor to a Tensor.
// IMPORTANT: this copies the data into the returned Tensor, which can be an expensive operation.
template <typename T>
tt::tt_metal::Tensor from_xtensor(const xt::xarray<T>& buffer, const TensorSpec& spec) {
    auto shape = get_shape_from_xarray(buffer);
    TT_FATAL(shape == spec.logical_shape(), "xtensor has a different shape than the supplied TensorSpec");
    auto buffer_view = xtensor_to_span(buffer);
    return tt::tt_metal::Tensor::from_span<T>(buffer_view, spec);
}

// Converts a Tensor to an xtensor.
// IMPORTANT: this copies the data into the returned Tensor, which can be an expensive operation.
template <typename T>
xt::xarray<T> to_xtensor(const tt::tt_metal::Tensor& tensor) {
    auto vec = tensor.to_vector<T>();
    auto shape = tensor.logical_shape();
    return xt::xarray<T>(span_to_xtensor_view(tt::stl::Span<const T>(vec.data(), vec.size()), shape));
}

}  // namespace ttnn::experimental::xtensor
