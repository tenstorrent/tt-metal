// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include <type_traits>
#include <vector>

#include <tt-metalium/experimental/tensor/host_tensor.hpp>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

namespace tt::tt_metal::host_buffer {

HostBuffer get_host_buffer(const Tensor& tensor) {
    TT_FATAL(is_cpu_tensor(tensor), "Tensor must have on host");
    return get_host_buffer(tensor.host_tensor());
}

template <typename T>
ttsl::Span<const T> get_as(const Tensor& tensor) {
    return get_as<T>(tensor.host_tensor());
}

template <typename T>
ttsl::Span<T> get_as(Tensor& tensor) {
    return get_as<T>(tensor.host_storage().host_tensor());
}

// Explicit template instantiations
#define INSTANTIATE_HOST_BUFFER_FUNCTIONS(T)                     \
    template ttsl::Span<const T> get_as<T>(const Tensor&);       \
    template ttsl::Span<const T> get_as<const T>(const Tensor&); \
    template ttsl::Span<T> get_as<T>(Tensor&);                   \
    template ttsl::Span<const T> get_as<const T>(Tensor&);

INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(int32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(float)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(bfloat16)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint16_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint8_t)

#undef INSTANTIATE_HOST_BUFFER_FUNCTIONS

}  // namespace tt::tt_metal::host_buffer
