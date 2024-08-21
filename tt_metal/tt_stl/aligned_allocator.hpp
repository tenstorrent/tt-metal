// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <new>

namespace tt::stl {

template <typename T, std::size_t Alignment = alignof(T)>
class aligned_allocator {
   public:
    static_assert(Alignment >= alignof(T), "Alignment must be at least as strict as alignof(T)");

    constexpr static std::align_val_t alignment{Alignment};

    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    constexpr aligned_allocator() noexcept = default;
    constexpr aligned_allocator(const aligned_allocator&) noexcept = default;

    [[nodiscard]] T* allocate(std::size_t n) const {
        if (n > max_size()) {
            throw std::bad_array_new_length();
        }

        if (n == 0) {
            return nullptr;
        }

        const size_type n_bytes = n * sizeof(T);
        void* const p = ::operator new[](n_bytes, alignment);

        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t size) const noexcept { ::operator delete[](p, size * sizeof(T), alignment); }

    [[nodiscard]] size_type max_size() const noexcept { return std::numeric_limits<size_type>::max() / sizeof(T); }
};

template <typename T, typename U, std::size_t Alignment>
constexpr bool operator==(const aligned_allocator<T, Alignment>&, const aligned_allocator<U, Alignment>&) noexcept {
    return true;
}

}  // namespace tt::stl
