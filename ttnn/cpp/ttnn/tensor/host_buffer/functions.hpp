// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/tensor/host_buffer/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

namespace tt {

namespace tt_metal {

namespace borrowed_buffer {

template <typename T>
void validate_datatype(const Tensor& tensor) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::UINT32);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::INT32);
    } else if constexpr (std::is_same_v<T, float>) {
        TT_FATAL(tensor.get_dtype() == DataType::FLOAT32);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        TT_FATAL(tensor.get_dtype() == DataType::BFLOAT16);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::UINT16);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::UINT8);
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported DataType");
    }
}

template <typename T>
Buffer<T> get_as(BorrowedBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template <typename T>
Buffer<T> get_as(const BorrowedBuffer& buffer) {
    TT_ASSERT(std::holds_alternative<Buffer<T>>(buffer), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(buffer),__FILE__, __LINE__));
    return std::get<Buffer<T>>(buffer);
}

template <typename T>
Buffer<T> get_as(Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [](auto&& storage) -> Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have BorrowedStorage");
            }
        },
        tensor.get_storage());
}

template <typename T>
Buffer<T> get_as(const Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [](auto&& storage) -> Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have BorrowedStorage");
            }
        },
        tensor.get_storage());
}

}  // namespace borrowed_buffer

namespace owned_buffer {

template <typename T>
Buffer<T> create(std::vector<T>&& storage) {
    return Buffer<T>{std::make_shared<std::vector<T>>(std::forward<std::vector<T>>(storage))};
}

template <typename T>
Buffer<T> create(std::size_t size) {
    return create(std::vector<T>(size, 0));
}

template <typename T>
void validate_datatype(const Tensor& tensor) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        TT_FATAL(
            tensor.get_dtype() == DataType::UINT32 or tensor.get_dtype() == DataType::BFLOAT8_B or
            tensor.get_dtype() == DataType::BFLOAT4_B);
    } else if constexpr (std::is_same_v<T, int32_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::INT32);
    } else if constexpr (std::is_same_v<T, float>) {
        TT_FATAL(tensor.get_dtype() == DataType::FLOAT32);
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        TT_FATAL(tensor.get_dtype() == DataType::BFLOAT16);
    } else if constexpr (std::is_same_v<T, uint16_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::UINT16);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        TT_FATAL(tensor.get_dtype() == DataType::UINT8);
    } else {
        static_assert(tt::stl::concepts::always_false_v<T>, "Unsupported DataType");
    }
}

template <typename T>
Buffer<T> get_as(OwnedBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template <typename T>
Buffer<T> get_as(const OwnedBuffer& buffer) {
    return std::get<Buffer<T>>(buffer);
}

template <typename T>
Buffer<T> get_as(Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [](auto&& storage) -> Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have OwnedStorage");
            }
        },
        tensor.get_storage());
}

template <typename T>
Buffer<T> get_as(const Tensor& tensor) {
    validate_datatype<T>(tensor);
    return std::visit(
        [](auto&& storage) -> Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                return get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have OwnedStorage");
            }
        },
        tensor.get_storage());
}

}  // namespace owned_buffer

namespace host_buffer {

template <typename T>
borrowed_buffer::Buffer<T> get_as(OwnedBuffer& buffer) {
    TT_ASSERT(std::holds_alternative<owned_buffer::Buffer<T>>(buffer), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(buffer),__FILE__, __LINE__));
    auto& owned_buffer = std::get<owned_buffer::Buffer<T>>(buffer);
    return borrowed_buffer::Buffer<T>(owned_buffer.begin(), owned_buffer.size());
}

template <typename T>
borrowed_buffer::Buffer<T> get_as(const OwnedBuffer& buffer) {
    TT_ASSERT(std::holds_alternative<owned_buffer::Buffer<T>>(buffer), fmt::format("Unexpected type {} in {}:{} ",tt::stl::get_active_type_name_in_variant(buffer),__FILE__, __LINE__));
    auto owned_buffer = std::get<owned_buffer::Buffer<T>>(buffer);
    return borrowed_buffer::Buffer<T>(owned_buffer.begin(), owned_buffer.size());
}

template <typename T>
borrowed_buffer::Buffer<T> get_as(OwnedBuffer&& buffer) = delete;
template <typename T>
borrowed_buffer::Buffer<T> get_as(const OwnedBuffer&& buffer) = delete;

template <typename T>
borrowed_buffer::Buffer<T> get_as(BorrowedBuffer& buffer) {
    return borrowed_buffer::get_as<T>(buffer);
}

template <typename T>
borrowed_buffer::Buffer<T> get_as(const BorrowedBuffer& buffer) {
    return borrowed_buffer::get_as<T>(buffer);
}

template <typename T>
borrowed_buffer::Buffer<T> get_as(Tensor& tensor) {
    return std::visit(
        [](auto&& storage) -> borrowed_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                return host_buffer::get_as<T>(storage.buffer);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                return host_buffer::get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have OwnedStorage or BorrowedStorage");
            }
        },
        tensor.get_storage());
}

template <typename T>
borrowed_buffer::Buffer<T> get_as(const Tensor& tensor) {
    return std::visit(
        [](auto&& storage) -> borrowed_buffer::Buffer<T> {
            using StorageType = std::decay_t<decltype(storage)>;
            if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
                return host_buffer::get_as<T>(storage.buffer);
            } else if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
                return host_buffer::get_as<T>(storage.buffer);
            } else {
                TT_THROW("Tensor must have OwnedStorage or BorrowedStorage");
            }
        },
        tensor.get_storage());
}

}  // namespace host_buffer

}  // namespace tt_metal

}  // namespace tt
