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
#include "ttnn/cpp/ttnn/operations/all_to_all_matmul/device/kernels/all_to_all_matmul_args.hpp"

using namespace all_to_all_matmul;

#if defined(ABLATE_ZONES)
#include "api/debug/kernel_profiler.hpp"
#define A2AMM_ZONE(n) DeviceZoneScopedN(n)
#else
#define A2AMM_ZONE(n)
#endif

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(writer_ct::CB_OUT);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(writer_ct::TILE_BYTES);
    constexpr uint32_t block_m = get_compile_time_arg_val(writer_ct::BLOCK_M);
    constexpr uint32_t Mq_t = get_compile_time_arg_val(writer_ct::MQ_T);
    constexpr uint32_t Nt = get_compile_time_arg_val(writer_ct::NT);
    constexpr uint32_t block_n = get_compile_time_arg_val(writer_ct::BLOCK_N);
    [[maybe_unused]] constexpr uint32_t num_n_blocks = get_compile_time_arg_val(writer_ct::NUM_N_BLOCKS);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(writer_ct::OUT_SUBBLOCK_H);
    constexpr uint32_t H = get_compile_time_arg_val(writer_ct::H);
    // Output tiles in flight per core (writer twin of the reader's weight_chunk_tiles): one write barrier per chunk.
    // 0 = off: one flush per row group only.
    constexpr uint32_t out_chunk_tiles = get_compile_time_arg_val(writer_ct::OUT_CHUNK_TILES);
    constexpr auto out_args = TensorAccessorArgs<writer_ct::COUNT>();
    constexpr uint32_t row_group_tiles = out_subblock_h * block_n;
    [[maybe_unused]] constexpr uint32_t num_row_groups = block_m / out_subblock_h;

    const uint32_t out_addr = get_common_arg_val<uint32_t>(writer_common::OUT_ADDR);
    const uint32_t n_slots = get_arg_val<uint32_t>(writer_rt::N_SLOTS);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    [[maybe_unused]] uint32_t in_flight = 0;

    for (uint32_t s = 0; s < n_slots; ++s) {
        const uint32_t rec = writer_rt::SLOTS_BASE + s * writer_slot::COUNT;
        const uint32_t h = get_arg_val<uint32_t>(rec + writer_slot::HEAD);
        // this core's n-blocks [nb0, nb1) and rows [m0, m1) of head h (tail M-split), 8 bits each
        const uint32_t nbw = get_arg_val<uint32_t>(rec + writer_slot::NBW);
        const uint32_t m0 = (nbw >> nbw_bits::M0_SHIFT) & nbw_bits::FIELD_MASK;
        const uint32_t m1 = (nbw >> nbw_bits::M1_SHIFT) & nbw_bits::FIELD_MASK;
        const uint32_t nb0 = (nbw >> nbw_bits::NB0_SHIFT) & nbw_bits::FIELD_MASK;
        const uint32_t nb1 = (nbw >> nbw_bits::NB1_SHIFT) & nbw_bits::FIELD_MASK;
        const uint32_t n_rg = (m1 - m0) / out_subblock_h;
        A2AMM_ZONE("wr_head");
        for (uint32_t nb = nb0; nb < nb1; ++nb) {
            for (uint32_t rg = 0; rg < n_rg; ++rg) {
                cb_wait_front(cb_out, row_group_tiles);
                uint32_t rp = get_read_ptr(cb_out);
                for (uint32_t r = 0; r < out_subblock_h; ++r) {
                    const uint32_t m = m0 + rg * out_subblock_h + r;
                    const uint32_t b = m / Mq_t;
                    const uint32_t mt = m - b * Mq_t;
                    const uint32_t page = ((b * H + h) * Mq_t + mt) * Nt + nb * block_n;
                    // Issue order rotated per head (see the reader): spreads the writers over the DRAM banks.
                    uint32_t n = h % block_n;
                    for (uint32_t i = 0; i < block_n; ++i) {
#ifndef ABLATE_NOWRITE
                        noc_async_write(rp + n * tile_bytes, out.get_noc_addr(page + n), tile_bytes);
#endif
                        if constexpr (out_chunk_tiles > 0) {
                            if (++in_flight == out_chunk_tiles) {
                                in_flight = 0;
                                noc_async_write_barrier();
                            }
                        }
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
