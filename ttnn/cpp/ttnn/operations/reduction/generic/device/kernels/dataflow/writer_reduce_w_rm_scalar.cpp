// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

// Writes one scalar (datum_bytes) per reduced row from the first bytes of each packed output tile.
void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rows = get_arg_val<uint32_t>(1);
    const uint32_t start_page = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_tile = tt::CBIndex::c_3;
    constexpr uint32_t datum_bytes = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    constexpr uint32_t onetile = 1;

    const auto dst_accessor = TensorAccessor(dst_args, dst_addr);

    experimental::Noc noc;
    experimental::CircularBuffer cb_tile(cb_id_tile);

    const uint32_t tile_bytes = get_tile_size(cb_id_tile);
    (void)tile_bytes;

    for (uint32_t i = 0; i < num_rows; i++) {
        cb_tile.wait_front(onetile);
        noc.async_write(
            cb_tile, dst_accessor, datum_bytes, {.offset_bytes = 0}, {.page_id = start_page + i, .offset_bytes = 0});
        noc.async_write_barrier();
        cb_tile.pop_front(onetile);
    }
}
