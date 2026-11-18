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
    constexpr auto weight_args = TensorAccessorArgs<5>();
    constexpr uint32_t round_tiles = block_k_tiles * block_n_tiles;
    static_assert(num_chunks >= 1 && num_chunks <= 8 && round_tiles % num_chunks == 0);
    constexpr uint32_t chunk_tiles = round_tiles / num_chunks;

    const uint32_t weight_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rounds = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_weights);
    const auto weights = TensorAccessor(weight_args, weight_addr, tile_bytes);
    conv2d_3x3::TaggedReads<false> reads;

    for (uint32_t r = 0; r < num_rounds; ++r) {
        cb_reserve_back(cb_weights, round_tiles);
        const uint32_t l1 = get_write_ptr(cb_weights);
        MaybeDeviceZoneScope("server_read");
        // The K-block's tiles are consecutive pages (bn is the whole C_out on this path).
        const uint32_t page0 = (r % num_k_blocks) * round_tiles;
        // Issue chunk c, then hand over chunk c - 1.
        for (uint32_t c = 0; c <= num_chunks; ++c) {
            if (c < num_chunks) {
                reads.begin(1 + c);
                for (uint32_t t = c * chunk_tiles; t < (c + 1) * chunk_tiles; ++t) {
                    reads.read(weights.get_noc_addr(page0 + t), l1 + t * tile_bytes, tile_bytes);
                }
                dataflow_kernel_lib::set_read_trid(reads.noc, 0);
            }
            if (c > 0) {
                reads.wait(c);
                cb_push_back(cb_weights, chunk_tiles);
            }
        }
    }
}
