// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <tt-metalium/bfloat16.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/overloaded.hpp>

#include <functional>
#include <memory>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

#include "ttnn/tensor/host_buffer/host_buffer.hpp"
#include "ttnn/tensor/host_buffer/memory_pin.hpp"

namespace tt::tt_metal {

HostBuffer::HostBuffer() : pin_(), view_(tt::stl::Span<std::byte>()), type_info_(nullptr), is_borrowed_(false) {}

HostBuffer::HostBuffer(const HostBuffer& other) = default;

HostBuffer& HostBuffer::operator=(const HostBuffer& other) {
    HostBuffer temp(other);
    swap(temp);
    return *this;
}

HostBuffer::HostBuffer(HostBuffer&& other) noexcept : HostBuffer() { swap(other); }

HostBuffer& HostBuffer::operator=(HostBuffer&& other) noexcept {
    swap(other);
    return *this;
}

void HostBuffer::swap(HostBuffer& other) noexcept {
    using std::swap;
    swap(pin_, other.pin_);
    swap(view_, other.view_);
    swap(type_info_, other.type_info_);
    swap(is_borrowed_, other.is_borrowed_);
}

tt::stl::Span<std::byte> HostBuffer::view_bytes() & noexcept { return view_; }

tt::stl::Span<const std::byte> HostBuffer::view_bytes() const& noexcept { return view_; }

bool HostBuffer::is_allocated() const { return pin_ != nullptr; }

bool HostBuffer::is_borrowed() const { return is_borrowed_; }

HostBuffer HostBuffer::deep_copy() const {
    auto copied_data = std::make_shared<std::vector<std::byte>>(view_bytes().begin(), view_bytes().end());
    HostBuffer copy;
    copy.view_ = tt::stl::Span<std::byte>(copied_data->data(), copied_data->size());
    copy.pin_ = MemoryPin(copied_data);
    copy.type_info_ = type_info_;
    copy.is_borrowed_ = false;
    return copy;
}

MemoryPin HostBuffer::pin() const { return pin_; }

void HostBuffer::deallocate() {
    pin_ = MemoryPin();
    view_ = tt::stl::Span<std::byte>();
    type_info_ = nullptr;
    is_borrowed_ = false;
}

bool operator==(const HostBuffer& a, const HostBuffer& b) noexcept {
    auto a_view = a.view_bytes();
    auto b_view = b.view_bytes();
    if (a_view.size() != b_view.size()) {
        return false;
    }
    for (auto i = 0; i < a_view.size(); i++) {
        if (a_view[i] != b_view[i]) {
            return false;
        }
    }
    return true;
}

bool operator!=(const HostBuffer& a, const HostBuffer& b) noexcept { return !(a == b); }

void swap(HostBuffer& lhs, HostBuffer& rhs) noexcept { lhs.swap(rhs); }

}  // namespace tt::tt_metal
