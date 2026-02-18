// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args: buffer address only
    uint32_t out_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_size = get_tile_size(cb_out);

    // Compile-time args: TensorAccessorArgs + dimensions
    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out = TensorAccessor(out_args, out_addr, tile_size);

    constexpr uint32_t dims_offset = out_args.next_compile_time_args_offset();
    constexpr uint32_t Mt = get_compile_time_arg_val(dims_offset);
    constexpr uint32_t Nt = get_compile_time_arg_val(dims_offset + 1);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            cb_wait_front(cb_out, 1);
            uint32_t l1_addr = get_read_ptr(cb_out);
            noc_async_write_tile(mt * Nt + nt, out, l1_addr);
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
