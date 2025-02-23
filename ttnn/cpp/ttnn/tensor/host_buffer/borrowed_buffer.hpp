// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <vector>

namespace tt::tt_metal::borrowed_buffer {

template <typename T>
struct Buffer {
    using value_type = T;

    Buffer() = default;
    Buffer(T* data_ptr, std::size_t size) : data_ptr_(data_ptr), size_(size) {}

    std::size_t size() const { return this->size_; }

    T& operator[](std::size_t index) noexcept { return this->data_ptr_[index]; }
    const T& operator[](std::size_t index) const noexcept { return this->data_ptr_[index]; }

    T* begin() noexcept { return this->data_ptr_; }
    T* end() noexcept { return this->data_ptr_ + this->size(); }

    const T* begin() const noexcept { return this->data_ptr_; }
    const T* end() const noexcept { return this->data_ptr_ + this->size(); }

    void* data() noexcept { return static_cast<void*>(this->data_ptr_); }
    const void* data() const noexcept { return static_cast<void*>(this->data_ptr_); }

private:
    T* data_ptr_ = nullptr;
    std::size_t size_ = 0;
};

template <typename T>
bool operator==(const Buffer<T>& buffer_a, const Buffer<T>& buffer_b) noexcept {
    if (buffer_a.size() != buffer_b.size()) {
        return false;
    }
    for (auto index = 0; index < buffer_a.size(); index++) {
        if (buffer_a[index] != buffer_b[index]) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool operator!=(const Buffer<T>& buffer_a, const Buffer<T>& buffer_b) noexcept {
    return not(buffer_a == buffer_b);
}

}  // namespace tt::tt_metal::borrowed_buffer
