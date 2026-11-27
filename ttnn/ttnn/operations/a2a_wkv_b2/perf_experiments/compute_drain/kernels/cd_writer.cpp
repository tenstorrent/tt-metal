// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compute_drain bench writer = the op's local-head publish: per block wait Nt tiles, write them to the
// (L1 interleaved) output, one barrier.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t out_tile_bytes = get_compile_time_arg_val(3);
    constexpr auto out_args = TensorAccessorArgs<4>();
    const auto out = TensorAccessor(out_args, get_arg_val<uint32_t>(0), out_tile_bytes);
    const uint32_t cb_base = get_read_ptr(cb_out);
    for (uint32_t mb = 0; mb < Mt; ++mb) {
        cb_wait_front(cb_out, (mb + 1) * Nt);
        DeviceZoneScopedN("cd_wr_block");
        uint32_t l1 = cb_base + mb * Nt * out_tile_bytes;
        for (uint32_t t = 0; t < Nt; ++t) {
            noc_async_write(l1, out.get_noc_addr(mb * Nt + t), out_tile_bytes);
            l1 += out_tile_bytes;
        }
        noc_async_write_barrier();
    }
    cb_pop_front(cb_out, Mt * Nt);
}
