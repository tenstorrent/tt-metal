// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight server, reader half (NCRISC / NoC0). A spare core outside the work grid streams the weight rounds
// of one channel rectangle (one K-block of bk x bn tiles per round, all the participants' rounds in order) from DRAM
// into cb_weights, whose slots sit at the same L1 address as the receivers'. Reads go out on NoC0, the read-preferred
// NoC, while the server writer multicasts on NoC1 from another RISC, so the reads of round r + 1 run under the
// multicast of round r. Each round is read in num_chunks chunks, each under its own transaction id and pushed as soon
// as it landed while the next chunk's reads are in flight, so the multicast of a round starts before its last read.
// One reader issuing one-tile reads is bound by the NIU's outstanding reads (~34 GB/s at 1088 B tiles), not by
// the issue loop.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"
#include "conv2d_3x3_weight_server.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(4);
    // kw-fold (conv2d_3x3_kw_fold.hpp): round kb is n-group kb / 3 of rw output column tiles, weight column kw = (1, 0,
    // 2)[kb % 3], [3 C_in, rw] in (kh, c, n) order (block_k_tiles = C_in, block_n_tiles = 3 rw: the same round size).
    constexpr bool kw_fold = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(6);
    // height_server_read_split: the server writer reads the last w_tiles tiles of every chunk itself (NoC1).
    constexpr uint32_t w_tiles = get_compile_time_arg_val(7);
    // height_server_read_split helpers: this reader reads tiles [lo, lo + own) of every chunk (runtime args 3, 4); on
    // a server, num_helpers helper cores read the rest (this kernel on their NCRISC,
    // conv2d_3x3_weight_server_helper.cpp on their BRISC writes each part into the server's slot) and count their
    // landed parts on semaphore sem_id here.
    constexpr uint32_t num_helpers = get_compile_time_arg_val(9);
    constexpr uint32_t hsr_sem_id = get_compile_time_arg_val(10);
    constexpr auto weight_args = TensorAccessorArgs<11>();
    constexpr uint32_t round_tiles = block_k_tiles * block_n_tiles;
    static_assert(num_chunks >= 1 && num_chunks <= 8 && round_tiles % num_chunks == 0);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;
    static_assert(w_tiles <= chunk_tiles);

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rounds = get_arg_val<uint32_t>(1);
    // Debug (timing only, wrong data): this server reads nothing.
    const bool dbg_skip = get_arg_val<uint32_t>(2) != 0;
    const uint32_t lo = get_arg_val<uint32_t>(3);
    const uint32_t own_tiles = dbg_skip ? 0 : get_arg_val<uint32_t>(4);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, tile_bytes);
    conv2d_3x3::TaggedReads<false> reads;
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* landed =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(hsr_sem_id));
    // Grant round r to every helper: its slot here is reserved.
    auto grant = [&]() {
        for (uint32_t h = 0; h < num_helpers; ++h) {
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(5 + 2 * h), get_arg_val<uint32_t>(6 + 2 * h), get_semaphore(hsr_sem_id)),
                1);
        }
    };

    for (uint32_t r = 0; r < num_rounds; ++r) {
        cb_reserve_back(cb_weights, round_tiles);
        if constexpr (num_helpers > 0) {
            grant();
        }
        const uint32_t l1 = get_write_ptr(cb_weights);
        MaybeDeviceZoneScope("server_read");
        const uint32_t kb = r % num_k_blocks;
        // Issue chunk c, then hand over chunk c - 1.
        for (uint32_t c = 0; c <= num_chunks; ++c) {
            if (c < num_chunks) {
                reads.begin(1 + c);
                for (uint32_t t = c * chunk_tiles + lo; t < c * chunk_tiles + lo + own_tiles; ++t) {
                    const uint32_t page =
                        conv2d_3x3::server_page_of<kw_fold, block_k_tiles, block_n_tiles, tensor_n_tiles>(kb, t);
                    reads.read(weights.get_noc_addr(page), l1 + t * tile_bytes, tile_bytes);
                }
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
            }
            if (c > 0) {
                reads.wait(c);
                if constexpr (num_helpers > 0) {
                    MaybeDeviceZoneScope("hsr_wait_helpers");
                    noc_semaphore_wait_min(landed, (r * num_chunks + c) * num_helpers);
                }
                cb_push_back(cb_weights, chunk_tiles);
            }
        }
    }
    noc_async_atomic_barrier();
}
