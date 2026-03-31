// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

template <typename T>
void validate_datatype(DataType dtype) {
    using BaseType = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<BaseType, uint32_t>) {
        TT_FATAL(
            dtype == DataType::UINT32 or dtype == DataType::BFLOAT8_B or dtype == DataType::BFLOAT4_B,
            "Incorrect data type {}",
            dtype);
    } else if constexpr (std::is_same_v<BaseType, int32_t>) {
        TT_FATAL(dtype == DataType::INT32, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, float>) {
        TT_FATAL(dtype == DataType::FLOAT32, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, bfloat16>) {
        TT_FATAL(dtype == DataType::BFLOAT16, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, uint16_t>) {
        TT_FATAL(dtype == DataType::UINT16, "Incorrect data type {}", dtype);
    } else if constexpr (std::is_same_v<BaseType, uint8_t>) {
        TT_FATAL(dtype == DataType::UINT8, "Incorrect data type {}", dtype);
    } else {
        static_assert(sizeof(BaseType) == 0, "Unsupported DataType");
    }
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

HostBuffer get_host_buffer(const HostTensor& tensor) {
    std::vector<HostBuffer> buffers;
    tensor.buffer().apply([&buffers](const HostBuffer& shard) { buffers.push_back(shard); });
    TT_FATAL(
        buffers.size() == 1,
        "Can't get a single buffer from host storage distributed over mesh shape {}",
        tensor.buffer().shape());
    return buffers.front();
}

HostBuffer get_host_buffer(const Tensor& tensor) {
    TT_FATAL(is_cpu_tensor(tensor), "Tensor must have on host");
    return get_host_buffer(tensor.host_tensor());
}

template <typename T>
ttsl::Span<const T> get_as(const HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
ttsl::Span<T> get_as(HostBuffer& buffer) {
    return buffer.view_as<T>();
}

template <typename T>
ttsl::Span<const T> get_as(const HostTensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

template <typename T>
ttsl::Span<const T> get_as(const Tensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

template <typename T>
ttsl::Span<T> get_as(Tensor& tensor) {
    CMAKE_UNIQUE_NAMESPACE::validate_datatype<T>(tensor.dtype());
    HostBuffer buffer = get_host_buffer(tensor);
    return buffer.template view_as<T>();
}

// Explicit template instantiations
#define INSTANTIATE_HOST_BUFFER_FUNCTIONS(T)                         \
    template ttsl::Span<const T> get_as<T>(const HostBuffer&);       \
    template ttsl::Span<const T> get_as<const T>(const HostBuffer&); \
    template ttsl::Span<T> get_as<T>(HostBuffer&);                   \
    template ttsl::Span<const T> get_as<const T>(HostBuffer&);       \
    template ttsl::Span<const T> get_as<T>(const HostTensor&);       \
    template ttsl::Span<const T> get_as<const T>(const HostTensor&); \
    template ttsl::Span<const T> get_as<T>(const Tensor&);           \
    template ttsl::Span<const T> get_as<const T>(const Tensor&);     \
    template ttsl::Span<T> get_as<T>(Tensor&);                       \
    template ttsl::Span<const T> get_as<const T>(Tensor&);

INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(int32_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(float)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(bfloat16)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint16_t)
INSTANTIATE_HOST_BUFFER_FUNCTIONS(uint8_t)

#undef INSTANTIATE_HOST_BUFFER_FUNCTIONS

}  // namespace tt::tt_metal::host_buffer
