// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// NOLINTBEGIN(misc-include-cleaner)
// Needs shape.hpp to export ttnn::Shape alias to tt_metal::Shape.
#include "ttnn/tensor/shape/shape.hpp"
// Forward include - re-exports tt-metalium host_buffer APIs for TTNN users.
#include <tt-metalium/experimental/tensor/host_buffer/functions.hpp>
// NOLINTEND(misc-include-cleaner)

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::host_buffer {

HostBuffer get_host_buffer(const Tensor& tensor);

template <typename T>
tt::stl::Span<const T> get_as(const Tensor& tensor) {
    validate_datatype<Tensor, T>(tensor);
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

template <typename T>
tt::stl::Span<T> get_as(Tensor& tensor) {
    validate_datatype<Tensor, T>(tensor);
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

}  // namespace tt::tt_metal::host_buffer
