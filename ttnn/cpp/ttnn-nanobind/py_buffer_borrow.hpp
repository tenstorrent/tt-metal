// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>

// Borrowing a Python buffer as a HostBuffer, shared by the Tensor constructor (pytensor.cpp) and
// host_tensor_from_bytes (tensor.cpp) so that the pin, the GIL rule and the data type table live in one place.
namespace ttnn::tensor::py_buffer {

// The element size of a data type a Python buffer can be borrowed as, or nullopt: block-float, fp8 and MX formats
// are sized by the device runtime and cannot be reinterpreted from fixed-width bytes.
inline std::optional<size_t> borrowable_element_size(tt::tt_metal::DataType dtype) {
    using tt::tt_metal::DataType;
    switch (dtype) {
        case DataType::BFLOAT16: return sizeof(bfloat16);
        case DataType::FLOAT32: return sizeof(float);
        case DataType::UINT32: return sizeof(uint32_t);
        case DataType::INT32: return sizeof(int32_t);
        case DataType::UINT16: return sizeof(uint16_t);
        case DataType::UINT8: return sizeof(uint8_t);
        case DataType::INT8: return sizeof(int8_t);
        default: return std::nullopt;
    }
}

// A HostBuffer over the memory of `buffer`: `elements` elements of `dtype` starting at its data pointer, no copy.
// The caller has checked that the buffer is contiguous and holds exactly `elements * borrowable_element_size(dtype)`
// bytes; this checks the alignment the element type needs and refuses data types that cannot be borrowed.
//
// The HostBuffer pins the Python object: a heap copy of the ndarray handle lives in the pin and is released with the
// last copy of the HostBuffer. nanobind guarantees the GIL for the thread that calls into C++, but that last release
// may run on a thread without it, so the deleter reacquires the GIL before dropping the Python reference.
//
// A read-only array (nb::ro) is accepted: HostBuffer has no read-only flavour, so the span is writable in type only,
// and nothing may write through it.
template <typename... Args>
tt::tt_metal::HostBuffer borrow_py_buffer(
    const nanobind::ndarray<Args...>& buffer, tt::tt_metal::DataType dtype, size_t elements) {
    using Array = nanobind::ndarray<Args...>;
    using tt::tt_metal::DataType;
    using tt::tt_metal::HostBuffer;

    tt::tt_metal::MemoryPin pin(std::shared_ptr<Array>(new Array(buffer), [](Array* pinned) {
        nanobind::gil_scoped_acquire gil;
        delete pinned;
    }));
    std::byte* data = const_cast<std::byte*>(static_cast<const std::byte*>(buffer.data()));

    auto borrow = [&]<typename T>() -> HostBuffer {
        TT_FATAL(
            reinterpret_cast<uintptr_t>(data) % alignof(T) == 0,
            "Cannot borrow a Python buffer as {}: its data starts at address {:#x}, which is not aligned to {} bytes",
            dtype,
            reinterpret_cast<uintptr_t>(data),
            alignof(T));
        return HostBuffer(ttsl::Span<T>(reinterpret_cast<T*>(data), elements), pin);
    };
    switch (dtype) {
        case DataType::BFLOAT16: return borrow.template operator()<bfloat16>();
        case DataType::FLOAT32: return borrow.template operator()<float>();
        case DataType::UINT32: return borrow.template operator()<uint32_t>();
        case DataType::INT32: return borrow.template operator()<int32_t>();
        case DataType::UINT16: return borrow.template operator()<uint16_t>();
        case DataType::UINT8: return borrow.template operator()<uint8_t>();
        case DataType::INT8: return borrow.template operator()<int8_t>();
        default: TT_THROW("Cannot borrow a Python buffer as {}: only fixed-width data types can be borrowed", dtype);
    }
}

}  // namespace ttnn::tensor::py_buffer
