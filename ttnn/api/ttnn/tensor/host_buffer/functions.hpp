// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt_stl/overloaded.hpp>
#include <type_traits>

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/tensor/tensor_utils.hpp>

namespace tt::tt_metal::host_buffer {

// validate_datatype, get_host_buffer(HostTensor), and get_as for HostBuffer/HostTensor
// are now defined in tt-metalium/experimental/tensor/tensor_utils.hpp

HostBuffer get_host_buffer(const Tensor& tensor);

template <typename T>
tt::stl::Span<const T> get_as(const Tensor& tensor) {
    return get_as<T>(tensor.host_tensor());
}

template <typename T>
tt::stl::Span<T> get_as(Tensor& tensor) {
    return get_as<T>(tensor.host_tensor());
}

}  // namespace tt::tt_metal::host_buffer
