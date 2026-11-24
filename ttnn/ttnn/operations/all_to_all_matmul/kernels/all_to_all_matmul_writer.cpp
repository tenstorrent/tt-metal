// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// all_to_all_matmul compute-core writer (RISCV_0 / NoC0).
//
// Per head slot, per n-block of the slot's range [nb0, nb1), per row group (out_subblock_h block rows x block_n
// columns, TileRowMajor): write each tile to output page ((b * H + h) * Mq_t + mt) * Nt + n, block row m = b * Mq_t +
// mt. One flush per row group before the pop (the CB slot may be refilled once the data has left L1); one barrier at
// the end.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#if defined(ABLATE_ZONES)
#include "tools/profiler/kernel_profiler.hpp"
#define A2AMM_ZONE(n) DeviceZoneScopedN(n)
#else
#define A2AMM_ZONE(n)
#endif

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t block_m = get_compile_time_arg_val(2);
    constexpr uint32_t Mq_t = get_compile_time_arg_val(3);
    constexpr uint32_t Nt = get_compile_time_arg_val(4);
    constexpr uint32_t block_n = get_compile_time_arg_val(5);
    [[maybe_unused]] constexpr uint32_t num_n_blocks = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(7);
    constexpr uint32_t H = get_compile_time_arg_val(8);
    constexpr auto out_args = TensorAccessorArgs<9>();
    constexpr uint32_t row_group_tiles = out_subblock_h * block_n;
    constexpr uint32_t num_row_groups = block_m / out_subblock_h;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t n_slots = get_arg_val<uint32_t>(0);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    for (uint32_t s = 0; s < n_slots; ++s) {
        const uint32_t h = get_arg_val<uint32_t>(1 + 2 * s);
        const uint32_t nbw = get_arg_val<uint32_t>(2 + 2 * s);  // this core's n-blocks [nb0, nb1) of head h
        A2AMM_ZONE("wr_head");
        for (uint32_t nb = nbw & 0xFFFF; nb < (nbw >> 16); ++nb) {
            for (uint32_t rg = 0; rg < num_row_groups; ++rg) {
                cb_wait_front(cb_out, row_group_tiles);
                uint32_t rp = get_read_ptr(cb_out);
                for (uint32_t r = 0; r < out_subblock_h; ++r) {
                    const uint32_t m = rg * out_subblock_h + r;
                    const uint32_t b = m / Mq_t;
                    const uint32_t mt = m - b * Mq_t;
                    const uint32_t page = ((b * H + h) * Mq_t + mt) * Nt + nb * block_n;
                    // Issue order rotated per head (see the reader): spreads the writers over the DRAM banks.
                    uint32_t n = h % block_n;
                    for (uint32_t i = 0; i < block_n; ++i) {
#ifndef ABLATE_NOWRITE
                        noc_async_write(rp + n * tile_bytes, out.get_noc_addr(page + n), tile_bytes);
#endif
                        n = (n + 1 == block_n) ? 0 : n + 1;
                    }
                    rp += block_n * tile_bytes;
                }
                noc_async_writes_flushed();
                cb_pop_front(cb_out, row_group_tiles);
            }
        }
    }
    noc_async_write_barrier();
}
