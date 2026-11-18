// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// t2_raw_kwalk_hxw isolated K-walk bench writer (BRISC): drains the compute's output block (tile row-major).
// CHECK_OUT: writes each tile to the DRAM output; otherwise wait / pop only (no NoC traffic).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
#ifdef CHECK_OUT
    constexpr auto out_args = TensorAccessorArgs<3>();
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_arg_val<uint32_t>(1);
    const auto out_acc = TensorAccessor(out_args, out_addr, tile_bytes);
#endif
    for (uint32_t t = 0; t < block_tiles; ++t) {
        cb_wait_front(cb_out, 1);
#ifdef CHECK_OUT
        noc_async_write(get_read_ptr(cb_out), out_acc.get_noc_addr(core_idx * block_tiles + t), tile_bytes);
        noc_async_writes_flushed();
#endif
        cb_pop_front(cb_out, 1);
    }
#ifdef CHECK_OUT
    noc_async_write_barrier();
#endif
}
