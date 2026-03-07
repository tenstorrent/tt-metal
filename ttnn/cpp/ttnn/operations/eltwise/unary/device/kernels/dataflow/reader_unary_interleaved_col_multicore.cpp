
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t core_number = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;
    const uint32_t num_tiles_per_2d = get_compile_time_arg_val(0);
    const uint32_t third_dim = get_compile_time_arg_val(1);
    const uint32_t number_blocks_per_core = get_compile_time_arg_val(2);
    constexpr auto src_args = TensorAccessorArgs<3>();

    constexpr uint32_t onetile = 1;

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);

#ifdef OUT_SHARDED
    cb.wait_front(onetile);
#else

    // single-tile ublocks
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);

    const auto s = TensorAccessor(src_args, src_addr, tile_bytes);

#ifdef BACKWARDS
    uint32_t end_id = -num_tiles_per_2d;
    for (uint32_t dim = 0; dim > -third_dim; dim--) {
        for (uint32_t k = 0; k > -num_blocks; k--) {
            for (uint32_t i = num_tiles_per_2d * dim - number_blocks_per_core * core_number;
                 i > end_id + num_tiles_per_2d * dim;
                 i = i - tiles_per_row) {
#else
    uint32_t end_id = num_tiles_per_2d;
    for (uint32_t dim = 0; dim < third_dim; dim++) {
        for (uint32_t k = 0; k < num_blocks; k++) {
            for (uint32_t i = num_tiles_per_2d * dim + number_blocks_per_core * core_number;
                 i < end_id + num_tiles_per_2d * dim;
                 i = i + tiles_per_row) {
#endif
                cb.reserve_back(onetile);
                noc.async_read(s, cb, tile_bytes, {.page_id = static_cast<uint32_t>(i + k)}, {.offset_bytes = 0});

                noc.async_read_barrier();
                cb.push_back(onetile);
            }
        }
    }
#endif
}
