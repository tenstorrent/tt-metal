
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t single_block_size_row_arg = get_arg_val<uint32_t>(2);
    uint32_t single_block_size_col_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t num_tiles_per_2d = get_compile_time_arg_val(0);
    constexpr uint32_t third_dim = get_compile_time_arg_val(1);
    constexpr uint32_t total_tiles_per_row = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);

#ifdef BACKWARDS
    for (uint32_t dim = 0; dim > -third_dim; dim--) {
        for (uint32_t c = 0; c > -single_block_size_col_arg; c--) {
            for (uint32_t r = 0; r > -single_block_size_row_arg; r--) {
                uint32_t tile = -start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
#else
    for (uint32_t dim = 0; dim < third_dim; dim++) {
        for (uint32_t c = 0; c < single_block_size_col_arg; c++) {
            for (uint32_t r = 0; r < single_block_size_row_arg; r++) {
                uint32_t tile = start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
#endif
                cb.reserve_back(onetile);
                noc.async_read(s, cb, tile_bytes, {.page_id = tile}, {.offset_bytes = 0});

                noc.async_read_barrier();
                cb.push_back(onetile);
            }
        }
    }
}
