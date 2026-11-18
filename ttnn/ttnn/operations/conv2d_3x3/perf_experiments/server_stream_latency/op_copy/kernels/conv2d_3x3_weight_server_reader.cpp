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

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(4);
    // kw-fold (conv2d_3x3_kw_fold.hpp): round kb is the weight column kw = 1, 0, 2 [3 C_in, C_out] in (kh, c, n)
    // order (block_k_tiles = C_in, block_n_tiles = 3 C_out: the same round size).
    constexpr bool kw_fold = get_compile_time_arg_val(5) != 0;
    constexpr auto weight_args = TensorAccessorArgs<6>();
    constexpr uint32_t round_tiles = block_k_tiles * block_n_tiles;
    static_assert(num_chunks >= 1 && num_chunks <= 8 && round_tiles % num_chunks == 0);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rounds = get_arg_val<uint32_t>(1);
#ifdef SSL_DBG_SKIP
    const bool dbg_skip = get_arg_val<uint32_t>(2) != 0;  // debug: this server reads nothing (output wrong)
#endif
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, tile_bytes);
    conv2d_3x3::TaggedReads<false> reads;

#ifndef SSL_WINDOW
    for (uint32_t r = 0; r < num_rounds; ++r) {
        cb_reserve_back(cb_weights, round_tiles);
        const uint32_t l1 = get_write_ptr(cb_weights);
        MaybeDeviceZoneScope("server_read");
        // The K-block's tiles are consecutive pages (bn is the whole C_out on this path); kw-fold's round of column kw
        // reads the prepared tile ((kh * 3 + kw) C_in + c, n) for slot (kh, c, n).
        const uint32_t kb = r % num_k_blocks;
        const uint32_t page0 = kb * round_tiles;
        auto page_of = [&](uint32_t t) -> uint32_t {
            if constexpr (kw_fold) {
                constexpr uint32_t n_tiles = block_n_tiles / 3;
                const uint32_t n = t % n_tiles;
                const uint32_t kc = t / n_tiles;
                const uint32_t kh = kc / block_k_tiles;
                const uint32_t kw = kb == 0 ? 1 : (kb == 1 ? 0 : 2);  // kw_fold.hpp KW_ROUND_ORDER
                return ((kh * 3 + kw) * block_k_tiles + (kc - kh * block_k_tiles)) * n_tiles + n;
            } else {
                return page0 + t;
            }
        };
        // Issue chunk c, then hand over chunk c - 1.
        for (uint32_t c = 0; c <= num_chunks; ++c) {
            if (c < num_chunks) {
                reads.begin(1 + c);
                for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; ++t) {
                    reads.read(weights.get_noc_addr(page_of(t)), l1 + t * tile_bytes, tile_bytes);
                }
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
            }
            if (c > 0) {
                reads.wait(c);
                cb_push_back(cb_weights, chunk_tiles);
            }
        }
    }
#else
    // server_stream_latency: a window of SSL_WINDOW chunks in flight across rounds, chunk j under trid 1 + j % window
    // and at ring slot j % (depth * num_chunks); chunks are pushed in order as they land.
    constexpr uint32_t window = SSL_WINDOW;
    static_assert(window >= 1 && window <= 12);
    constexpr uint32_t ring_chunks = SSL_DEPTH * num_chunks;
    const uint32_t chunk_bytes = chunk_tiles * tile_bytes;
    const uint32_t base = get_write_ptr(cb_weights);
    const uint32_t total = num_rounds * num_chunks;
    uint32_t issued = 0;
#if SSL_FAST_ADDR
    static_assert(!weight_args.is_sharded);
    constexpr uint32_t nbanks = weight_args.is_dram ? NUM_DRAM_BANKS : NUM_L1_BANKS;
    uint64_t bank_base[nbanks];
    for (uint32_t b = 0; b < nbanks; ++b) {
        bank_base[b] = weights.get_noc_addr(b);
    }
    const uint32_t bank_stride = static_cast<uint32_t>(weights.get_noc_addr(nbanks) - bank_base[0]);
#if SSL_TWO_NOCS
    uint64_t bank_base1[nbanks];
    for (uint32_t b = 0; b < nbanks; ++b) {
        bank_base1[b] = weights.get_noc_addr(b, 0, 1);
    }
#endif
#endif
    // In-order pipeline: issue chunk j (trid 1 + j % window), then hand over chunk j - window + 1, so window chunks
    // are in flight across round boundaries. A depth-1 ring drains a round before the next one's reads.
    MaybeDeviceZoneScope("server_read");
    uint32_t landed = 0;
    auto hand_over = [&]() {
#if SSL_SPLIT
        if (landed % num_chunks < SSL_SPLIT) {
            cb_push_back(cb_weights, chunk_tiles);
            ++landed;
            return;
        }
#endif
        reads.wait(1 + landed % window);
        cb_push_back(cb_weights, chunk_tiles);
        ++landed;
    };
    while (issued < total) {
        if (SSL_DEPTH == 1) {
            while (landed < issued && landed / num_chunks < issued / num_chunks) {
                hand_over();
            }
        }
        cb_reserve_back(cb_weights, (issued - landed + 1) * chunk_tiles);
        do {
            const uint32_t r = issued / num_chunks;
            const uint32_t c = issued - r * num_chunks;
            const uint32_t kb = r % num_k_blocks;
            const uint32_t l1 = base + (issued % ring_chunks) * chunk_bytes - c * chunk_bytes;
            const uint32_t page0 = kb * round_tiles;
            auto page_of = [&](uint32_t t) -> uint32_t {
                if constexpr (kw_fold) {
                    constexpr uint32_t n_tiles = block_n_tiles / 3;
                    const uint32_t n = t % n_tiles;
                    const uint32_t kc = t / n_tiles;
                    const uint32_t kh = kc / block_k_tiles;
                    const uint32_t kw = kb == 0 ? 1 : (kb == 1 ? 0 : 2);  // kw_fold.hpp KW_ROUND_ORDER
                    return ((kh * 3 + kw) * block_k_tiles + (kc - kh * block_k_tiles)) * n_tiles + n;
                } else {
                    return page0 + t;
                }
            };
#if SSL_SPLIT
            if (c < SSL_SPLIT) {
                break;  // the server writer reads this chunk itself: only its CB pages are handed over
            }
#endif
            reads.begin(1 + issued % window);
#ifdef SSL_DBG_SKIP
            if (dbg_skip) {
            } else
#endif
#ifdef SSL_DBG_BIGREAD
                // debug (timing only, wrong data): SSL_DBG_BIGREAD tiles per read, contiguous in one bank.
                for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; t += SSL_DBG_BIGREAD) {
                    const uint32_t p = page_of(t);
                    noc_async_read<NOC_MAX_BURST_SIZE>(
                        bank_base[p % nbanks] + (p / nbanks) * bank_stride,
                        l1 + t * tile_bytes,
                        tile_bytes * SSL_DBG_BIGREAD);
                    reads.count_packets(1);
                }
#elif SSL_TWO_NOCS
            // Odd tiles on NoC1's read command buffer (the server writer never reads), under the same trid.
            noc_async_read_set_trid(1 + issued % window, 1);
            for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; ++t) {
                const uint32_t p = page_of(t);
                noc_async_read<NOC_MAX_BURST_SIZE>(
                    ((t & 1) ? bank_base1 : bank_base)[p % nbanks] + (p / nbanks) * bank_stride,
                    l1 + t * tile_bytes,
                    tile_bytes,
                    t & 1);
            }
            noc_async_read_set_trid(0, 1);
#elif SSL_FAST_ADDR
            // Interleaved pages: page p sits in bank p % nbanks, (p / nbanks) bank pages past that bank's base.
            for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; ++t) {
                const uint32_t p = page_of(t);
                noc_async_read<NOC_MAX_BURST_SIZE>(
                    bank_base[p % nbanks] + (p / nbanks) * bank_stride, l1 + t * tile_bytes, tile_bytes);
                reads.count_packets(1);
            }
#else
            for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; ++t) {
                reads.read(weights.get_noc_addr(page_of(t)), l1 + t * tile_bytes, tile_bytes);
            }
#endif
            dataflow_kernel_lib::set_read_trid(reads.noc, 0);
        } while (false);
        ++issued;
        if (issued - landed >= window) {
            hand_over();
        }
    }
    while (landed < total) {
        hand_over();
    }
#endif
}
