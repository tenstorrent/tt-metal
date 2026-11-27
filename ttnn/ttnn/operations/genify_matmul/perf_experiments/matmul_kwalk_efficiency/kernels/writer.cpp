// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// K-walk bench writer (BRISC). CHECK_OUT: drain the compute's fp32 copy of the accumulated block
// (tile row-major) to the DRAM output, one tile per page. Perf mode: nothing to do.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
#ifdef CHECK_OUT
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr auto out_args = TensorAccessorArgs<3>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_arg_val<uint32_t>(1);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
    for (uint32_t t = 0; t < block_tiles; ++t) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), out_acc.get_noc_addr(core_idx * block_tiles + t), tile_bytes);
        noc_async_writes_flushed();
        cb_pop_front(cb_out, 1);
    }
    noc_async_write_barrier();
#endif
}
