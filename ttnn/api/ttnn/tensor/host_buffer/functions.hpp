// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

namespace tt::tt_metal::host_buffer {

HostBuffer get_host_buffer(const ttnn::Tensor& tensor);

template <typename T>
ttsl::Span<const T> get_as(const ttnn::Tensor& tensor);

template <typename T>
ttsl::Span<T> get_as(ttnn::Tensor& tensor);

}  // namespace tt::tt_metal::host_buffer
