// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t N = get_arg(args::rank);
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);

    const uint32_t start_tile = get_arg(args::start_tile);
    const uint32_t end_tile = get_arg(args::end_tile);

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::input);
    DataflowBuffer dfb(dfb::cb_in0);
    const uint32_t tile_bytes = dfb.get_tile_size();
    Noc noc;

    // Rank-length arrays (count = N, a CTA) delivered as runtime varargs:
    // output_tiled_shape in varargs [0, N), inv_perm in [N, 2N), src_strides in [2N, 3N).
    uint32_t output_tiled_shape[N], inv_perm[N], src_strides[N];
    for (uint32_t i = 0; i < N; i++) {
        output_tiled_shape[i] = get_vararg(i);
        inv_perm[i] = get_vararg(i + N);
        src_strides[i] = get_vararg(i + 2 * N);
    }

    for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
        // Compute multi-dimensional index for the source tile
        uint32_t dest_multi_idx[N];
        size_t remaining = tile;
        for (uint32_t i = 0; i < N; ++i) {
            size_t dim = N - 1 - i;
            dest_multi_idx[dim] = remaining % output_tiled_shape[dim];
            remaining /= output_tiled_shape[dim];
        }

        // Apply permutation to get destination multi-dimensional index
        uint32_t src_multi_idx[N];
        for (uint32_t i = 0; i < N; ++i) {
            src_multi_idx[i] = dest_multi_idx[inv_perm[i]];
        }

        // Convert destination multi-dimensional index to linear index
        uint32_t src_linear_idx = 0;
        for (uint32_t i = 0; i < N; ++i) {
            src_linear_idx += src_multi_idx[i] * src_strides[i];
        }

        dfb.reserve_back(onetile);
        noc.async_read(s, dfb, tile_bytes, {.page_id = src_linear_idx}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb.push_back(onetile);
    }
}
