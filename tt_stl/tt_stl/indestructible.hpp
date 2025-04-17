// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <new>
#include <utility>

namespace tt::stl {

// `Indestructible` is a wrapper around `T` that behaves like `T` but does not call the destructor of `T`.
// This is useful for creating objects with static storage duration: `Indestructible` avoids heap allocation, provides
// thread-safe construction, and ensures the destructor is no-op, so does not depend on any other objects.
//
//
// Example usage:
//
// const auto& get_object() {
//     static Indestructible<MyObject> object;
//     return object.get();
// }
//
template <typename T>
class Indestructible {
public:
    template <typename... Args>
    explicit Indestructible(Args&&... args) {
        // Construct T in our aligned storage
        new (&storage_) T(std::forward<Args>(args)...);
    }

    T& get() { return *std::launder(reinterpret_cast<T*>(&storage_)); }

    const T& get() const { return *std::launder(reinterpret_cast<const T*>(&storage_)); }

    // Disable copy and assignment
    Indestructible(const Indestructible&) = delete;
    Indestructible& operator=(const Indestructible&) = delete;

    // Destructor does NOT call T's destructor.
    // This leaves the object "indestructible."
    ~Indestructible() = default;

private:
    // A buffer of std::byte with alignment of T and size of T
    alignas(T) std::byte storage_[sizeof(T)];
};

}  // namespace tt::stl
