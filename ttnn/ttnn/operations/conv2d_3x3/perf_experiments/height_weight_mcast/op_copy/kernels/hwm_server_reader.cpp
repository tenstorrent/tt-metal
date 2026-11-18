// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Experiment height_weight_mcast: weight server, reader half (NCRISC / NoC0). A spare core outside the work grid reads
// its chunk [t0, t1) of every weight round from DRAM on NoC0 (the read-preferred NoC) into the same cb_weights slot the
// receivers use, and pushes it to the server writer, which multicasts it on NoC1. With two slots the read of round r +
// 1 runs under the multicast of round r, on the other NoC and the other RISC. With num_chunks > 1 every chunk of a
// round is pushed as soon as it landed (own transaction id each), so the multicast of chunk c runs under the reads of c
// + 1.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(5);
    constexpr auto weight_args = TensorAccessorArgs<6>();
    constexpr uint32_t round_tiles = block_k_tiles * block_n_tiles;
    static_assert(num_chunks >= 1 && num_chunks <= 8 && round_tiles % num_chunks == 0);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rounds = get_arg_val<uint32_t>(1);
    const uint32_t t0 = get_arg_val<uint32_t>(2);
    const uint32_t t1 = get_arg_val<uint32_t>(3);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, tile_bytes);
    conv2d_3x3::TaggedReads<false> reads;

    for (uint32_t r = 0; r < num_rounds; ++r) {
        cb_reserve_back(cb_weights, round_tiles);
        const uint32_t l1 = get_write_ptr(cb_weights);
        MaybeDeviceZoneScope("server_read");
        const uint32_t k_tile0 = (r % num_k_blocks) * block_k_tiles;
        // Issue chunk c, then hand over chunk c - 1 (landed by now): the reads are bound by the NIU's outstanding-read
        // limit (~34 GB/s at one 1088 B tile per read), so the multicast of chunk c - 1 runs under the reads of c.
        // HWM_SRV_DUAL: the server writer reads the odd chunks on NoC1; this RISC pushes them unread, in order.
        for (uint32_t c = 0; c <= num_chunks; ++c) {
#ifdef HWM_SRV_DUAL
            const bool mine = (c & 1) == 0;
#else
            constexpr bool mine = true;
#endif
            if (c < num_chunks && mine) {
                reads.begin(1 + c);
                const uint32_t lo = t0 > c * chunk_tiles ? t0 : c * chunk_tiles;
                const uint32_t hi = t1 < (c + 1) * chunk_tiles ? t1 : (c + 1) * chunk_tiles;
#ifdef HWM_FASTREAD
                // The round's pages are consecutive (bn == tensor_n_tiles): page P0 + t sits in bank (P0 + t) % NB at
                // that bank's row (P0 + t) / NB. One address per bank, then one-packet reads stepping by the row
                // stride: no per-page bank arithmetic.
                static_assert(block_n_tiles == tensor_n_tiles);
                if (lo < hi) {
                    constexpr uint32_t NB = NUM_DRAM_BANKS;
                    const uint32_t p0 = k_tile0 * tensor_n_tiles + lo;
                    uint64_t bank_addr[NB];
                    for (uint32_t b = 0; b < NB; ++b) {
                        bank_addr[b] = weights.get_noc_addr(p0 + b);
                    }
                    const uint64_t row_stride = weights.get_noc_addr(p0 + NB) - bank_addr[0];
                    uint32_t dst = l1 + lo * tile_bytes;
                    for (uint32_t i = 0, n = hi - lo; i < n;) {
                        for (uint32_t b = 0; b < NB && i < n; ++b, ++i, dst += tile_bytes) {
                            noc_async_read_one_packet(bank_addr[b], dst, tile_bytes);
                            bank_addr[b] += row_stride;
                        }
                        reads.count_packets(NB);
                    }
                }
#else
                for (uint32_t t = lo; t < hi; ++t) {
                    reads.read(
                        weights.get_noc_addr((k_tile0 + t / block_n_tiles) * tensor_n_tiles + t % block_n_tiles),
                        l1 + t * tile_bytes,
                        tile_bytes);
                }
#endif
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
            }
            if (c > 0) {
#ifdef HWM_SRV_DUAL
                if (((c - 1) & 1) == 0) {
                    reads.wait(c);
                }
#else
                reads.wait(c);
#endif
                cb_push_back(cb_weights, chunk_tiles);
            }
        }
    }
}
