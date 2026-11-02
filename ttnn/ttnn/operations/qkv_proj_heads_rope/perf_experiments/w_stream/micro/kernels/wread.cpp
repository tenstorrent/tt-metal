// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// w_stream micro: one RISC reads tiles [lo, hi) of every W round (k_block x n_tiles slice of the N-group's
// columns) from DRAM on this kernel's NoC, one barrier per round. Same source on BRISC (NoC1) and NCRISC (NoC0).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t k_block = get_compile_time_arg_val(1);
    constexpr uint32_t n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t width_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr auto w_args = TensorAccessorArgs<5>();
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_col = get_arg_val<uint32_t>(1);
    const uint32_t lo = get_arg_val<uint32_t>(2);
    const uint32_t hi = get_arg_val<uint32_t>(3);
    const uint32_t rounds = get_arg_val<uint32_t>(4);
    const auto w = TensorAccessor(w_args, w_addr, tile_bytes);
    const uint32_t base = get_write_ptr(cb_scratch);
    for (uint32_t r = 0; r < rounds; ++r) {
        DeviceZoneScopedN("wread_round");
        for (uint32_t t = lo; t < hi; ++t) {
            const uint32_t kt = (r * k_block + t / n_tiles) % 24;
            noc_async_read_page(kt * width_tiles + first_col + t % n_tiles, w, base + t * tile_bytes);
        }
        noc_async_read_barrier();
    }
}
