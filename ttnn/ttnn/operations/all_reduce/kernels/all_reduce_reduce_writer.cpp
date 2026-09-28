// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// all_reduce — reduce writer (BRISC), core (0,2).
//
// Drains cb_summed in g-tile granules and writes the P summed shard tiles to the
// DENSE output tensor (pages 0..P-1). The compute kernel drains granules in the
// reader's dense page order (R11: every contribution streams pages 0..P-1 of its
// block), so the t-th tile drained maps to output page t and this kernel is
// shape-agnostic. Pure local NoC writes; no fabric, no semaphores.
//
// CT args: [cb_summed, P, g] + output TensorAccessorArgs
// RT args: [output_addr, page_size]

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_summed = get_compile_time_arg_val(0);
    constexpr uint32_t P = get_compile_time_arg_val(1);  // shard tiles = output pages
    constexpr uint32_t g = get_compile_time_arg_val(2);  // granule (divides P)
    constexpr auto output_args = TensorAccessorArgs<3>();

    static_assert(g > 0 && P % g == 0, "all_reduce: granule must divide the shard tile count (R5)");

    uint32_t ai = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);

    const auto output = TensorAccessor(output_args, output_addr, page_size);

    for (uint32_t chunk = 0; chunk < P / g; ++chunk) {
        cb_wait_front(cb_summed, g);
        uint32_t l1 = get_read_ptr(cb_summed);
        for (uint32_t t = 0; t < g; ++t) {
            noc_async_write(l1, output.get_noc_addr(chunk * g + t), page_size);
            l1 += page_size;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_summed, g);
    }
}
