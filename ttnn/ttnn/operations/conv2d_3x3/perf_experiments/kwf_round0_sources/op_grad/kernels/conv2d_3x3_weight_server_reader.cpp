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
// res3.x a 52 KB round is 12 reads of 4 tiles instead of 48 one-tile reads (2.1 -> 1.3-1.5 us). A round of at most
// one NoC burst (kw-fold whole-K: res2, center_head.1) is latency-bound: all its chunks' reads go out at once, so it
// lands together and the writer sends it as one chain (round 0: res2 unzoned, leave-one-out: 10.50 us without it, 10.27
// us with). Every such round does, not round 0 alone: issued chunk by chunk, rounds 1 and 2 took ~1.1 us each (three
// DRAM round trips back to back), and with round 0 no longer held for start acks (the server writer) res2's fused side
// rounds and center_head.1's Q_0 / Q_2 rounds waited for them. Larger rounds keep the pipelined issue: on sem dec res3
// fc0's 120-tile rounds the extra issue in front of chunk 0 delayed round 0 by ~0.35 us (round 0 all at once there and
// on res3.x's 52 KB round: sem dec res3 fc0 / inst dec res3 fc0 first matmul ~1.1 us later, zoned).
//
// Dual-source rectangles (kw-fold; conv2d_3x3_weight_server_writer.cpp): two servers share every chunk, this one
// reading slot tiles [part_lo, part_hi) of each (whole slot rows, so the page runs stay whole), the partner the rest;
// a single server reads [0, chunk). With half a chunk per server, one chunk ahead left ~4 page-run reads in flight and
// the reads latency-bound (res3.x: chunk halves landed ~0.3 us apart, as whole chunks had): a dual server issues every
// chunk of every round first (arg ahead = num_chunks), each still under its own transaction id and handed over in
// order. res3.x (unzoned, medians of 3): dual with one chunk ahead 9.23 us, round 0 all ahead 8.98, every round 8.94.
// Round 0 has its own split (arg r0_split, byte c: the first server's slot tiles [0, split_c) of chunk c, the partner
// the rest; see the server writer), and a partner's writer may read the partner's part of round 0 itself
// (SERVER_R0_SELF_READ, arg 6): this reader hands round 0's slot over unread there.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "conv2d_3x3_gather_dataflow.hpp"
#include "conv2d_3x3_server_pages.hpp"

void kernel_main() {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_n_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(4);
    // Arg 5: 0 off kw-fold, else 1 + the fused side rounds' n-group width (conv2d_3x3_server_pages.hpp).
    constexpr uint32_t kwf_arg = get_compile_time_arg_val(5);
    constexpr uint32_t tensor_n_tiles = get_compile_time_arg_val(6);
    // ND-sharded weights: the shard's [rows, columns] in tiles; 0 rows = interleaved (one read per tile).
    constexpr uint32_t shard_rows = get_compile_time_arg_val(7);
    constexpr uint32_t shard_cols = get_compile_time_arg_val(8);
    constexpr auto weight_args = TensorAccessorArgs<9>();
    using Pages = conv2d_3x3::
        ServerPages<block_k_tiles, block_n_tiles, num_k_blocks, kwf_arg, tensor_n_tiles, shard_rows, shard_cols>;
    constexpr uint32_t round_tiles = block_k_tiles * block_n_tiles;
    static_assert(num_chunks >= 1 && num_chunks <= 4 && round_tiles % num_chunks == 0);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rounds = get_arg_val<uint32_t>(1);
    // This server's part of every chunk and the chunks issued ahead of the first hand-over (see the file comment).
    const uint32_t part_lo = get_arg_val<uint32_t>(2);
    const uint32_t part_hi = get_arg_val<uint32_t>(3);
    const uint32_t chunks_ahead = get_arg_val<uint32_t>(4);
    // Round 0's split (see the file comment; 0: the args' part); part_lo 0 is the first (or only) server.
    const uint32_t r0_split = get_arg_val<uint32_t>(5);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, tile_bytes);
    conv2d_3x3::TaggedReads<false> reads;
    const bool small_round = round_tiles * tile_bytes <= NOC_MAX_BURST_SIZE;

    for (uint32_t r = 0; r < num_rounds; ++r) {
        cb_reserve_back(cb_weights, round_tiles);
        const uint32_t l1 = get_write_ptr(cb_weights);
        MaybeDeviceZoneScope("server_read");
        // Issue chunk c, then hand over chunk c - ahead; a one-burst round issues every chunk first.
        const uint32_t ahead = small_round ? num_chunks : chunks_ahead;
        for (uint32_t c = 0; c < num_chunks + ahead; ++c) {
            if (c < num_chunks) {
                uint32_t lo = part_lo;
                uint32_t hi = part_hi;
                if (r == 0 && r0_split != 0) {
                    const uint32_t split = (r0_split >> (8 * c)) & 0xFF;
                    lo = part_lo == 0 ? 0 : split;
                    hi = part_lo == 0 ? split : chunk_tiles;
                }
                reads.begin(1 + c);
#ifdef SERVER_R0_SELF_READ
                // A partner whose writer reads its part of round 0 (arg 6).
                if (r != 0 || get_arg_val<uint32_t>(6) == 0)
#endif
                {
                    Pages::read(reads, weights, tile_bytes, r, l1, c * chunk_tiles + lo, c * chunk_tiles + hi);
                }
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
            }
            if (c >= ahead) {
                reads.wait(1 + c - ahead);
                cb_push_back(cb_weights, chunk_tiles);
            }
        }
    }
}
