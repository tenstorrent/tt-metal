// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
// s1_pace matmul bench writer (BRISC): waits for every q tile (resident cb_out, like the op's cb_q_slice), then writes
// them to DRAM (row-major tiles: block j row i col c -> page (j*M+i)*cols + c) for the correctness check.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(1);
    constexpr auto o_args = TensorAccessorArgs<2>();
    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t ot = get_tile_size(cb_out);
    const auto oa = TensorAccessor(o_args, o_addr, ot);
    cb_wait_front(cb_out, n_tiles);
    uint32_t l1 = get_read_ptr(cb_out);
    for (uint32_t p = 0; p < n_tiles; ++p) {
        noc_async_write(l1, oa.get_noc_addr(p), ot);
        l1 += ot;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_tiles);
}
