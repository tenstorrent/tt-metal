// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// conv2d_3x3 weight server, reader half (NCRISC / NoC0). A spare core outside the work grid streams the weight rounds
// of one channel rectangle (one K-block of bk x bn tiles per round, all the participants' rounds in order) from DRAM
// into cb_weights, whose slots sit at the same L1 address as the receivers'. Reads go out on NoC0, the read-preferred
// NoC, while the server writer multicasts on NoC1 from another RISC, so the reads of round r + 1 run under the
// multicast of round r. Each round is read in num_chunks chunks, each under its own transaction id and pushed as soon
// as it landed while the next chunk's reads are in flight, so the multicast of a round starts before its last read.
// One reader issuing one-tile reads is bound by the NIU's outstanding reads (~34 GB/s at 1088 B tiles), not by
// the issue loop. With ND-sharded weights (conv2d_3x3.weight_memory_config: shards of shard_rows tile-rows x
// shard_cols columns, round-robin over the DRAM banks) every run of slot tiles that is contiguous both in the slot and
// in one shard is one read (page runs): a shard's tiles sit at consecutive aligned-page addresses in its bank. On
// res3.x a 52 KB round is 12 reads of 4 tiles instead of 48 one-tile reads (2.1 -> 1.3-1.5 us). A round 0 of at most
// one NoC burst (kw-fold whole-K: res2, center_head.1) is latency-bound and gates compute's first matmul: all its
// chunks' reads go out at once, so it lands together and the writer sends it as one chain (res2 unzoned,
// leave-one-out: 10.50 us without it, 10.27 us with). Larger rounds keep the pipelined issue: on sem dec res3 fc0's
// 120-tile rounds the extra issue in front of chunk 0 delayed round 0 by ~0.35 us.

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
    // kw-fold (conv2d_3x3_kw_fold.hpp): round kb is n-group kb / 3 of rw output column tiles, weight column kw = (1, 0,
    // 2)[kb % 3], [3 C_in, rw] in (kh, c, n) order (block_k_tiles = C_in, block_n_tiles = 3 rw: the same round size).
    // Arg 5: 0 off kw-fold, else 1 + the fused side rounds' n-group width (0: not fused; kw_fold.hpp): rounds 1 and 2
    // of an n-group set then hold the kw = 0 | 2 pair interleaved per (kh, c) row of 2 rw tiles, each gw-wide
    // n-group's kw = 0 columns, then its kw = 2 columns.
    constexpr bool kw_fold = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t kwf_fused_gw = kw_fold ? get_compile_time_arg_val(5) - 1 : 0;
#ifdef KWF_INTERLEAVE_NG
    static_assert(kw_fold && kwf_fused_gw == 0, "n-group-interleaved rounds: kw-fold without fused side rounds");
#endif
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(6);
    // ND-sharded weights: the shard's [rows, columns] in tiles; 0 rows = interleaved (one read per tile).
    constexpr uint32_t shard_rows = get_compile_time_arg_val(7);
    constexpr uint32_t shard_cols = get_compile_time_arg_val(8);
    constexpr auto weight_args = TensorAccessorArgs<9>();
    constexpr uint32_t round_tiles = block_k_tiles * block_n_tiles;
    static_assert(num_chunks >= 1 && num_chunks <= 8 && round_tiles % num_chunks == 0);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;
    // Guarded: this branch is compiled on every shape (block_n_tiles may be < 3 off kw-fold).
    constexpr uint32_t rw = block_n_tiles >= 3 ? block_n_tiles / 3 : 1;
    // A slot holds rows of row_tiles tiles: (kh, c) rows of one n-group on kw-fold, K-block rows of all C_out
    // otherwise.
    constexpr uint32_t row_tiles = kw_fold ? rw : block_n_tiles;

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rounds = get_arg_val<uint32_t>(1);
    // kw-fold planes (descriptor KWF_N_PLANES): the first output column tile this server's plane streams (0 otherwise).
    const uint32_t n_off = get_arg_val<uint32_t>(2);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, tile_bytes);
    conv2d_3x3::TaggedReads<false> reads;
    const bool small_round = round_tiles * tile_bytes <= NOC_MAX_BURST_SIZE;

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
                if constexpr (kwf_fused_gw != 0) {
                    const uint32_t col = kb % 3;
                    if (col != 0) {
                        constexpr uint32_t frow = 2 * rw;
                        const uint32_t f = (col - 1) * round_tiles + t;
                        const uint32_t kc = f / frow;  // kh * C_in + c
                        const uint32_t j = f - kc * frow;
                        const uint32_t grp = j / (2 * kwf_fused_gw);
                        const uint32_t jj = j - grp * 2 * kwf_fused_gw;
                        const uint32_t kh = kc / block_k_tiles;
                        const uint32_t kwf = jj < kwf_fused_gw ? 0 : 2;
                        const uint32_t n = grp * kwf_fused_gw + (jj < kwf_fused_gw ? jj : jj - kwf_fused_gw);
                        return ((kh * 3 + kwf) * block_k_tiles + (kc - kh * block_k_tiles)) * tensor_n_tiles + n_off +
                               (kb / 3) * rw + n;
                    }
                }
                const uint32_t n = t % rw;
                const uint32_t kc = t / rw;
                const uint32_t kh = kc / block_k_tiles;
#ifdef KWF_INTERLEAVE_NG
                // kw-fold planes, stream mode: the sub-rounds n-group-interleaved (round kb = n-group kb % NG, column
                // kb / NG), so every plane's first round comes early.
                const uint32_t col = kb / KWF_INTERLEAVE_NG;
                const uint32_t grp = kb % KWF_INTERLEAVE_NG;
#else
                const uint32_t col = kb % 3;
                const uint32_t grp = kb / 3;
#endif
                const uint32_t kw = col == 0 ? 1 : (col == 1 ? 0 : 2);  // kw_fold.hpp KW_ROUND_ORDER
                return ((kh * 3 + kw) * block_k_tiles + (kc - kh * block_k_tiles)) * tensor_n_tiles + n_off + grp * rw +
                       n;
            } else {
                return page0 + t;
            }
        };
        // Slot tiles [t, end) of this chunk, as runs. From slot tile t the run goes to the edge of its page's shard row
        // (within the slot row); while a slot row is exactly one shard row, on through the shard's next rows, which the
        // next slot rows hold (kw-fold: up to the end of the kh block; K-block: of the round).
        auto read_tiles = [&](uint32_t t, uint32_t end) {
            while (t < end) {
                const uint32_t page = page_of(t);
                uint32_t len = 1;
                if constexpr (shard_rows > 0 && kwf_fused_gw != 0) {
                    if (kb % 3 != 0) {
                        // Fused side round: a run is one n-group's kw = 0 (or kw = 2) columns of one (kh, c) row.
                        // kernel_main is not a template: the discarded branch is still checked, so no / 0.
                        constexpr uint32_t gw = kwf_fused_gw != 0 ? kwf_fused_gw : 1;
                        static_assert(shard_cols % gw == 0);
                        len = gw - ((kb % 3 - 1) * round_tiles + t) % gw;
                        if (len > end - t) {
                            len = end - t;
                        }
                        reads.read(weights.get_noc_addr(page), l1 + t * tile_bytes, len * tile_bytes);
                        t += len;
                        continue;
                    }
                }
                if constexpr (shard_rows > 0) {
                    const uint32_t i = t / row_tiles;
                    const uint32_t n = t - i * row_tiles;
                    const uint32_t row = page / tensor_n_tiles;
                    const uint32_t c = page - row * tensor_n_tiles;
                    len = shard_cols - c % shard_cols;
                    if (len > row_tiles - n) {
                        len = row_tiles - n;
                    }
                    if (row_tiles == shard_cols && (c - n) % shard_cols == 0 && n + len == row_tiles) {
                        const uint32_t in_shard = shard_rows - 1 - row % shard_rows;
                        const uint32_t follow = block_k_tiles - 1 - (kw_fold ? i % block_k_tiles : i);
                        len += (in_shard < follow ? in_shard : follow) * row_tiles;
                    }
                    if (len > end - t) {
                        len = end - t;
                    }
                }
                reads.read(weights.get_noc_addr(page), l1 + t * tile_bytes, len * tile_bytes);
                t += len;
            }
        };
        // Issue chunk c, then hand over chunk c - 1; a small round 0 issues every chunk first.
        const uint32_t ahead = r == 0 && small_round ? num_chunks : 1;
        for (uint32_t c = 0; c < num_chunks + ahead; ++c) {
            if (c < num_chunks) {
                reads.begin(1 + c);
                read_tiles(c * chunk_tiles, (c + 1) * chunk_tiles);
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
            }
            if (c >= ahead) {
                reads.wait(1 + c - ahead);
                cb_push_back(cb_weights, chunk_tiles);
            }
        }
    }
}
