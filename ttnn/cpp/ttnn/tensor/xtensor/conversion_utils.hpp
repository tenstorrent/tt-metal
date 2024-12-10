// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/shape/small_vector.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <ttnn/tensor/xtensor/xtensor_all_includes.hpp>

namespace ttnn::experimental::xtensor {

template <typename E>
ttnn::Shape get_shape_from_xarray(const E& expr) {
    ttnn::SmallVector<uint32_t> shape_dims;
    for (size_t i = 0; i < expr.shape().size(); ++i) {
        shape_dims.push_back(expr.shape()[i]);
    }
    return ttnn::Shape(shape_dims);
}

template <class VectorType = float, DataType TensorType = DataType::BFLOAT16>
tt::tt_metal::Tensor from_vector(const std::vector<VectorType>& buffer, const ttnn::Shape& shape);

template <class T = float>
std::vector<T> to_vector(const tt::tt_metal::Tensor& tensor);

template <class T>
xt::xarray<T> span_to_xtensor(std::span<T> vec, const ttnn::SimpleShape& shape) {
    std::vector<size_t> shape_vec(shape.cbegin(), shape.cend());
    return xt::adapt(vec.data(), vec.size(), xt::no_ownership(), shape_vec);
}
template <class T>
auto xtensor_to_span(const xt::xarray<T>& xtensor) {
    auto adaptor = xt::adapt(xtensor.data(), xtensor.size(), xt::no_ownership());
    return std::span(adaptor.data(), adaptor.size());
}

template <class T = float, DataType TensorType = DataType::BFLOAT16>
tt::tt_metal::Tensor from_xtensor(const xt::xarray<T>& buffer) {
    auto shape = get_shape_from_xarray(buffer);
    auto buffer_view = xtensor_to_span(buffer);
    return from_vector<T, TensorType>(std::vector<T>(buffer_view.begin(), buffer_view.end()), shape);
}

template <class T = float>
xt::xarray<T> to_xtensor(const tt::tt_metal::Tensor& tensor) {
    auto vec = to_vector<T>(tensor);
    auto shape = tensor.get_shape().logical_shape();
    return span_to_xtensor(std::span<T>(vec.data(), vec.size()), shape);
}

}  // namespace ttnn::experimental::xtensor
