// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/experimental/tensor/tensor_apis.hpp>

// NOTE: These Tensor overloads intentionally remain in tt::tt_metal::host_buffer alongside the
// metalium HostBuffer/HostTensor overloads. Moving only the Tensor overloads into ttnn::host_buffer
// causes enclosing-namespace lookup from within ttnn::* to hide the metalium overloads.
// See NAMESPACE_MIGRATION_ADL.md.

namespace tt::tt_metal::host_buffer {

HostBuffer get_host_buffer(const Tensor& tensor);

template <typename T>
ttsl::Span<const T> get_as(const Tensor& tensor);

template <typename T>
ttsl::Span<T> get_as(Tensor& tensor);

}  // namespace tt::tt_metal::host_buffer
