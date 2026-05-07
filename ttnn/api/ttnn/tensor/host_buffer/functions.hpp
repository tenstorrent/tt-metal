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
