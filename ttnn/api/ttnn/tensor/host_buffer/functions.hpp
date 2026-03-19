// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::host_buffer {

HostBuffer get_host_buffer(const Tensor& tensor);

template <typename T>
ttsl::Span<const T> get_as(const HostBuffer& buffer);

template <typename T>
ttsl::Span<T> get_as(HostBuffer& buffer);

template <typename T>
ttsl::Span<const T> get_as(const Tensor& tensor);

template <typename T>
ttsl::Span<T> get_as(Tensor& tensor);

}  // namespace tt::tt_metal::host_buffer
