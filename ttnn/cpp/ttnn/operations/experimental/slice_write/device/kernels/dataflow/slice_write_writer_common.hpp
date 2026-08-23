// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "experimental/kernel_args.h"

namespace slice_write {

constexpr uint32_t MAX_RANK = 8;

struct Geometry {
    uint32_t num_unpadded[MAX_RANK];
    uint32_t num_padded[MAX_RANK];
    uint32_t id[MAX_RANK];
};

struct StridedGeometry : Geometry {
    uint32_t reverse_stride[MAX_RANK];
};

FORCE_INLINE Geometry load_geometry(uint32_t num_dims) {
    Geometry geometry{};
    for (uint32_t j = 0; j < num_dims; ++j) {
        geometry.num_unpadded[j] = get_vararg(j);
        geometry.num_padded[j] = get_vararg(num_dims + j);
        geometry.id[j] = get_vararg(2 * num_dims + j);
    }
    return geometry;
}

FORCE_INLINE StridedGeometry load_strided_geometry(uint32_t num_dims) {
    StridedGeometry geometry{};
    for (uint32_t j = 0; j < num_dims; ++j) {
        geometry.num_unpadded[j] = get_vararg(j);
        geometry.num_padded[j] = get_vararg(num_dims + j);
        geometry.id[j] = get_vararg(2 * num_dims + j);
        geometry.reverse_stride[j] = get_vararg(3 * num_dims + j);
    }
    return geometry;
}

template <typename GeometryType>
FORCE_INLINE void advance(uint32_t num_dims, GeometryType& geometry, uint32_t& page_id) {
    for (uint32_t j = 0; j < num_dims; ++j) {
        geometry.id[j]++;
        if (geometry.id[j] == geometry.num_unpadded[j]) {
            geometry.id[j] = 0;
            page_id += geometry.num_padded[j];
        } else {
            break;
        }
    }
}

}  // namespace slice_write
