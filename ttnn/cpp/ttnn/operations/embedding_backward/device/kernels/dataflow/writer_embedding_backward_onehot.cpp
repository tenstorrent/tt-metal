// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer for the output-stationary embedding backward: every work unit's finished tiles are written once.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

void kernel_main() {
    const uint32_t output_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_unit = get_arg_val<uint32_t>(1);
    const uint32_t unit_stride = get_arg_val<uint32_t>(2);
    const uint32_t total_units = get_arg_val<uint32_t>(3);

    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(2);

    constexpr auto out_args = TensorAccessorArgs<3>();
    const auto out_s = TensorAccessor(out_args, output_tensor_addr);

    constexpr uint32_t cb_out_idx = tt::CBIndex::c_16;
    Noc noc;
    CircularBuffer cb_out(cb_out_idx);
    const uint32_t out_tile_bytes = get_tile_size(cb_out_idx);

    for (uint32_t u = first_unit; u < total_units; u += unit_stride) {
        const uint32_t r = u / num_chunks;
        const uint32_t c = u % num_chunks;
        const uint32_t first_tile = c * chunk_tiles;
        constexpr uint32_t tiles = chunk_tiles;

        cb_out.wait_front(tiles);
        uint32_t l1 = cb_out.get_read_ptr();
        for (uint32_t t = 0; t < tiles; ++t) {
            noc.async_write(
                CoreLocalMem<uint32_t>(l1), out_s, out_tile_bytes, {}, {.page_id = r * hidden_tiles + first_tile + t});
            l1 += out_tile_bytes;
        }
        noc.async_writes_flushed();
        cb_out.pop_front(tiles);
    }
    noc.async_write_barrier();
}
