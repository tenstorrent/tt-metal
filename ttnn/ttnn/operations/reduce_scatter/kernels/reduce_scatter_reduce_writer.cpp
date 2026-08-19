// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// reduce_scatter — Phase B reduce writer (BRISC). Writes each reduced (summed)
// tile from cb_reduced_slice to its dense output page start_tile + t. Pure local
// NoC writes — the output slice is dense, so the walk is just start_tile + t
// (SequentialTileWalker deliberately not used: no step/channel structure exists —
// see op_design.md "Helpers considered and rejected").

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_reduced_slice = get_compile_time_arg_val(0);
    constexpr auto output_args = TensorAccessorArgs<1>();

    uint32_t ai = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t page_size = get_arg_val<uint32_t>(ai++);
    const uint32_t start_tile = get_arg_val<uint32_t>(ai++);
    const uint32_t num_tiles = get_arg_val<uint32_t>(ai++);

    const auto output = TensorAccessor(output_args, output_addr, page_size);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_reduced_slice, 1);
        const uint32_t l1 = get_read_ptr(cb_reduced_slice);
        noc_async_write(l1, output.get_noc_addr(start_tile + t), page_size);
        noc_async_write_barrier();
        cb_pop_front(cb_reduced_slice, 1);
    }
}
