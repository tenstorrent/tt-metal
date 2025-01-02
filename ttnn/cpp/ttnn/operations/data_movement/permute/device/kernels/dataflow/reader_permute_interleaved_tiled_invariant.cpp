// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr uint32_t N = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(3);

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t end_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = 0;
    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    // start at runtime arg 3 since address/start_block/end_block make up the first 3 args
    uint32_t output_tiled_shape[N], inv_perm[N], src_strides[N];
    for (uint32_t i = 3; i < N + 3; i++) {
        output_tiled_shape[i - 3] = get_arg_val<uint32_t>(i);
        inv_perm[i - 3] = get_arg_val<uint32_t>(i + N);
        src_strides[i - 3] = get_arg_val<uint32_t>(i + 2 * N);
    }

    uint32_t src_buffer_l1_addr = get_write_ptr(tt::CBIndex::c_0);
    uint32_t curr_addr = src_addr;
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

        cb_reserve_back(cb_id_in0, onetile);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read_tile(src_linear_idx, s, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
