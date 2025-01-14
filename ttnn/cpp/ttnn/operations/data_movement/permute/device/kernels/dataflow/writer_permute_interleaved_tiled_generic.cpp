// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

// ------------------------------------------------------------------
// 1) unflatten_index<N>:
//    Unflatten 'flat_idx' in row-major order for a shape[] of length N.
//    shape[d] is also uint32_t. We store the result into out_multi_idx[].
template <uint32_t N>
inline void unflatten_index(uint32_t flat_idx, const uint32_t (&shape)[N], uint32_t (&out_multi_idx)[N]) {
    // Process from last dimension to first, in row-major unflattening.
    for (int d = N - 1; d >= 0; d--) {
        uint32_t dim_size = shape[d];
        out_multi_idx[d] = flat_idx % dim_size;
        flat_idx /= dim_size;
    }
}

// ------------------------------------------------------------------
// 2) flatten_index_ignore_last_dim<N>:
//    Flatten all N dims in row-major order except ignoring dimension N-1.
template <uint32_t N>
inline uint32_t flatten_index_ignore_last_dim(const uint32_t (&multi_idx)[N], const uint32_t (&shape)[N]) {
    uint32_t offset = 0;
    for (uint32_t d = 0; d < N - 1; d++) {
        offset = offset * shape[d] + multi_idx[d];
    }
    return offset;
}

template <uint32_t N, uint32_t TILE_HEIGHT, uint32_t TILE_WIDTH>
inline uint32_t get_unpadded_linear_row_index_for_tile(
    uint32_t tile,
    const uint32_t (&input_tiled_shape)[N],  // [ ..., X_t, W_t ]
    const uint32_t (&input_shape)[N],        // [ ..., X,   W   ]
    uint32_t (&src_multi_idx)[N]) {
    // 1) Unflatten 'tile' => src_multi_idx in the tiled shape
    unflatten_index<N>(tile, input_tiled_shape, src_multi_idx);

    // 2) Multiply the X-dim by TILE_HEIGHT
    src_multi_idx[N - 2] *= TILE_HEIGHT;

    // 3) Flatten ignoring last dim => linear row offset
    return flatten_index_ignore_last_dim<N>(src_multi_idx, input_shape);
}

void kernel_main() {}
