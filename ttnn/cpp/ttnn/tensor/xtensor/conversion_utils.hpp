// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/shape/small_vector.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <ttnn/tensor/xtensor/xtensor_all_includes.hpp>

namespace ttnn::experimental::xtensor {

template <typename E>
ttnn::SimpleShape get_shape_from_xarray(const E& xarr) {
    ttnn::SmallVector<uint32_t> shape_dims;
    for (size_t i = 0; i < xarr.shape().size(); ++i) {
        shape_dims.push_back(xarr.shape()[i]);
    }
    return ttnn::SimpleShape(shape_dims);
}

template <typename T>
tt::tt_metal::Tensor from_span(
    tt::stl::Span<const T> buffer, const ttnn::SimpleShape& shape, tt::tt_metal::DataType dtype);

template <typename T>
tt::tt_metal::Tensor from_vector(
    const std::vector<T>& buffer, const ttnn::SimpleShape& shape, tt::tt_metal::DataType dtype) {
    return from_span(tt::stl::Span<const T>(buffer.data(), buffer.size()), shape, dtype);
}

template <typename T>
std::vector<T> to_vector(const tt::tt_metal::Tensor& tensor);

template <typename T>
xt::xarray<T> tt_span_to_xtensor(tt::stl::Span<const T> vec, const ttnn::SimpleShape& shape) {
    std::vector<size_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(vec.data(), vec.size(), xt::no_ownership(), shape_vec);
}

// TODO: make the usage of std::span / tt::stl::Span consistent.
template <typename T>
xt::xarray<T> span_to_xtensor(std::span<T> vec, const ttnn::SimpleShape& shape) {
    std::vector<size_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(vec.data(), vec.size(), xt::no_ownership(), shape_vec);
}

template <typename T>
auto xtensor_to_tt_span(const xt::xarray<T>& xtensor) {
    auto adaptor = xt::adapt(xtensor.data(), xtensor.size(), xt::no_ownership());
    return tt::stl::Span<const T>(adaptor.data(), adaptor.size());
}

// TODO: make the usage of std::span / tt::stl::Span consistent.
template <typename T>
auto xtensor_to_span(const xt::xarray<T>& xtensor) {
    auto adaptor = xt::adapt(xtensor.data(), xtensor.size(), xt::no_ownership());
    return std::span(adaptor.data(), adaptor.size());
}

template <typename T>
tt::tt_metal::Tensor from_xtensor(const xt::xarray<T>& buffer, tt::tt_metal::DataType dtype) {
    auto shape = get_shape_from_xarray(buffer);
    auto buffer_view = xtensor_to_tt_span(buffer);
    return from_span<T>(buffer_view, shape, dtype);
}

template <typename T>
xt::xarray<T> to_xtensor(const tt::tt_metal::Tensor& tensor) {
    auto vec = to_vector<T>(tensor);
    auto shape = tensor.get_shape().logical_shape();
    return tt_span_to_xtensor(tt::stl::Span<const T>(vec.data(), vec.size()), shape);
}

}  // namespace ttnn::experimental::xtensor
