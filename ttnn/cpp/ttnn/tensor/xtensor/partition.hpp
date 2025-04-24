// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/xtensor/xtensor_all_includes.hpp"

namespace ttnn::experimental::xtensor {

// IMPORTANT: `chunk` and `concatenate` are not yet part of the public ttnn API. Internally, they rely on the `xtensor`
// library for efficient host-side operations.

// Splits a tensor into chunks along the specified dimension.
std::vector<tt::tt_metal::Tensor> chunk(const tt::tt_metal::Tensor& tensor, int num_chunks, int dim = 0);

// Overload for `xt::xarray`.
template <typename T>
std::vector<xt::xarray<T>> chunk(const xt::xarray<T>& tensor, int num_chunks, int dim = 0);

// Overload for `tt::stl::Span` that performs zero-copy chunking on the outermost dimension (dim = 0).
template <typename T>
std::vector<std::pair<tt::stl::Span<T>, ttnn::Shape>> chunk(
    tt::stl::Span<T> span, const ttnn::Shape& shape, int num_chunks);

// Concatenates a list of tensors along the specified dimension.
tt::tt_metal::Tensor concat(const std::vector<tt::tt_metal::Tensor>& tensors, int dim = 0);

// Overload for `xt::xarray`.
template <typename T>
xt::xarray<T> concat(const std::vector<xt::xarray<T>>& v, int dim = 0);

}  // namespace ttnn::experimental::xtensor
