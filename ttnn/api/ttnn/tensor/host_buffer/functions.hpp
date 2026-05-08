// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

namespace ttnn::host_buffer {

tt::tt_metal::HostBuffer get_host_buffer(const Tensor& tensor);

template <typename T>
ttsl::Span<const T> get_as(const Tensor& tensor);

template <typename T>
ttsl::Span<T> get_as(Tensor& tensor);

}  // namespace ttnn::host_buffer

// Compatibility bridges - ttnn tensor infrastructure has moved to the ttnn namespace.
namespace tt::tt_metal::host_buffer {

template <int = 0>
[[deprecated("use ttnn::host_buffer::get_host_buffer instead. This alias may be removed after Jun 2026.")]]
inline tt::tt_metal::HostBuffer get_host_buffer(const ttnn::Tensor& tensor) {
    return ttnn::host_buffer::get_host_buffer(tensor);
}

template <typename T, int = 0>
[[deprecated("use ttnn::host_buffer::get_as instead. This alias may be removed after Jun 2026.")]]
inline ttsl::Span<const T> get_as(const ttnn::Tensor& tensor) {
    return ttnn::host_buffer::get_as<T>(tensor);
}

template <typename T, int = 0>
[[deprecated("use ttnn::host_buffer::get_as instead. This alias may be removed after Jun 2026.")]]
inline ttsl::Span<T> get_as(ttnn::Tensor& tensor) {
    return ttnn::host_buffer::get_as<T>(tensor);
}

}  // namespace tt::tt_metal::host_buffer
