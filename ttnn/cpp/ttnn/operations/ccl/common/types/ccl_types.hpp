// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 *    ------   ATTENTION  ATTENTION  ATTENTION  ATTENTION  ATTENTION   ------
 * This file is intended to be useable across both host and device code. Therefore.
 *
 * DO NOT include any headers that are not host/device agnostic.
 * DO NOT use any types that do not have fixed sizes across host and device.
 * e.g. int32_t -> good (always 32 bits), int -> bad (size depends on platform)
 */

#include <cstdint>
#include <cstddef>

namespace ttnn {
namespace ccl {

using address_t = uint32_t;

template <typename T>
struct Shape4D {
    T w;
    T z;
    T y;
    T x;

    Shape4D() = default;
    Shape4D(T w, T z, T y, T x) : w(w), z(z), y(y), x(x) {}
    Shape4D(Shape4D const &rhs) = default;

    Shape4D<T> operator+(const Shape4D<T> &rhs) const {
        return {w + rhs.w, z + rhs.z, y + rhs.y, x + rhs.x};
    }

    bool operator==(const Shape4D<T> &rhs) const {
        return w == rhs.w && z == rhs.z && y == rhs.y && x == rhs.x;
    }

    constexpr std::size_t volume() const {
        return w * z * y * x;
    }
};

struct WorkerEdmInterfaceArgs {
    const uint32_t edm_noc_x;
    const uint32_t edm_noc_y;
    const address_t edm_buffer_base_address;
    const address_t edm_semaphore_address;
    const uint32_t num_buffers_per_channel;
};

}  // namespace ccl
}  // namespace ttnn
