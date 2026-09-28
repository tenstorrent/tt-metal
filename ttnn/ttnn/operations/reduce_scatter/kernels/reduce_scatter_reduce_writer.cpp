// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — reduce writer (BRISC), core (0,2).
//
// Drains cb_output_tiles in g-tile granules and writes the S summed slice tiles to
// the DENSE output tensor (pages 0..S-1). The compute kernel streams granules in
// the reader's walker order, which IS the output tensor's own row-major tile order
// for the dim=3 scatter (slice_Wt columns of every tile row), so the t-th tile
// drained maps to output page t and this kernel needs no walker. Pure local NoC
// writes; no fabric, no semaphores.
//
// CT args: [cb_output_tiles, S, g] + output TensorAccessorArgs
// RT args: [output_addr, page_size]

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);  // slice tiles = output pages
    constexpr uint32_t g = get_compile_time_arg_val(2);  // granule (divides S)
    constexpr auto output_args = TensorAccessorArgs<3>();

    static_assert(g > 0 && S % g == 0, "reduce_scatter: granule must divide the slice tile count");

    uint32_t ai = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);

    const auto output = TensorAccessor(output_args, output_addr, page_size);

    for (uint32_t chunk = 0; chunk < S / g; ++chunk) {
        cb_wait_front(cb_output_tiles, g);
        uint32_t l1 = get_read_ptr(cb_output_tiles);
        for (uint32_t t = 0; t < g; ++t) {
            noc_async_write(l1, output.get_noc_addr(chunk * g + t), page_size);
            l1 += page_size;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_output_tiles, g);
    }
}
