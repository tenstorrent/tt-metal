// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-reader (x_round0_fill experiment copy). Same as the op's x-reader, plus:
//   * chunked delivery: round r is split into nch(r) chunks of round_tiles / nch(r) consecutive tiles
//     (nch = x_chunks_per_round for every round, or only for round 0 when x_chunk_r0_only). Each chunk
//     carries its own transaction id and is pushed as soon as it lands, so the multicaster can send
//     the head of a round while its tail is still being read (read and multicast pipeline within the
//     round). The trid window is `x_read_window` chunks (<= 15).
//   * r0_brisc_chunks: the last r0_brisc_chunks chunks of round 0 are read by the multicaster (BRISC,
//     the other NoC) instead; this kernel pushes them without reading.
// x_chunks_per_round == 1, r0_brisc_chunks == 0 is the op's schedule (window = X_READ_TRIDS rounds).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t x_read_window = get_compile_time_arg_val(5);  // chunks in flight
    constexpr uint32_t first_round = get_compile_time_arg_val(6);
    constexpr uint32_t round_stride = get_compile_time_arg_val(7);
    constexpr uint32_t x_chunks_per_round = get_compile_time_arg_val(8);
    constexpr bool x_chunk_r0_only = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t r0_brisc_chunks = get_compile_time_arg_val(10);
    constexpr bool x_eager_push = get_compile_time_arg_val(11) != 0;
    constexpr auto x_args = TensorAccessorArgs<12>();
    constexpr uint32_t my_rounds = (num_k_blocks - first_round + round_stride - 1) / round_stride;

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;
    static_assert(x_read_window >= 1 && x_read_window <= NOC_MAX_TRANSACTION_ID, "window out of trid range");
    static_assert(round_tiles % x_chunks_per_round == 0, "chunks must divide the round");
    static_assert(r0_brisc_chunks == 0 || first_round == 0, "brisc round-0 reads need the round-0 sender");

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const uint32_t round_bytes = round_tiles * x_tile_bytes;
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);
    const Noc noc;

    cb_reserve_back(cb_x_resident, round_tiles * my_rounds);
    const uint32_t base = get_write_ptr(cb_x_resident);

    // Chunk q's tile count, for the push when it retires (ring of the window's size).
    uint32_t ring[x_read_window];
    uint32_t q = 0, head = 0;  // next chunk to issue / to retire
    auto retire_push = [&]() {
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, (head % x_read_window) + 1);
        cb_push_back(cb_x_resident, ring[head % x_read_window]);
        ++head;
    };
    // x_eager_push: push every landed chunk (in order) before issuing the next one, not only when its
    // trid is needed again (the op's schedule pushes round r only when issuing round r + window).

    {
        MaybeDeviceZoneScope("x_read_issue");
        for (uint32_t i = 0; i < my_rounds; ++i) {
            const uint32_t r = first_round + i * round_stride;
            const uint32_t nch = (r == 0 || !x_chunk_r0_only) ? x_chunks_per_round : 1;
            const uint32_t chunk_tiles = round_tiles / nch;
            for (uint32_t c = 0; c < nch; ++c, ++q) {
                if constexpr (x_eager_push) {
                    while (head < q &&
                           ncrisc_noc_read_with_transaction_id_flushed(noc_index, (head % x_read_window) + 1)) {
                        retire_push();
                    }
                }
                if (q - head >= x_read_window) {
                    retire_push();
                }
                ring[q % x_read_window] = chunk_tiles;
                dataflow_kernel_lib::set_read_trid(noc, (q % x_read_window) + 1);
#ifndef ABLATE_X_READ
                if (!(r == 0 && c >= nch - r0_brisc_chunks)) {
                    uint32_t dst = base + r * round_bytes + c * chunk_tiles * x_tile_bytes;
                    for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; ++t) {
                        const uint32_t m = t / block_k_tiles, kk = t % block_k_tiles;
                        noc_async_read_tile(m * tensor_k_tiles + r * block_k_tiles + kk, x_acc, dst);
                        dst += x_tile_bytes;
                    }
                }
#endif
            }
        }
    }
    MaybeDeviceZoneScope("x_read_drain");
    while (head < q) {
        retire_push();
    }
    noc_async_read_barrier();
    dataflow_kernel_lib::set_read_trid(noc, 0);
}
