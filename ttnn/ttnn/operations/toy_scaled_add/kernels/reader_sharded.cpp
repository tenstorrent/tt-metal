// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// toy_scaled_add reader for height-sharded a / b (NCRISC, NoC0).
//
// The a and b circular buffers are backed by this core's shards, so their tiles are already in L1:
// publishing them to compute is one push each, no NoC traffic. Only gamma, which is interleaved, is
// read — all Wt tiles of its row, once.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/toy_scaled_add/device/kernels/toy_scaled_add_args.hpp"

using namespace toy_scaled_add;

void kernel_main() {
    constexpr uint32_t Wt = get_named_compile_time_arg_val("Wt");

    const uint32_t num_rows = get_arg_val<uint32_t>(core_arg::NUM_ROWS);
    const uint32_t shard_tiles = num_rows * Wt;

    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<0>();

#ifdef TOY_SCALED_ADD_HAS_GAMMA
    Noc noc;
    CircularBuffer cb_gamma(cb::GAMMA);
    const uint32_t gamma_tile_bytes = get_tile_size(cb::GAMMA);
    const auto gamma =
        TensorAccessor(gamma_args, get_common_arg_val<uint32_t>(sharded_reader_arg::GAMMA_ADDR), gamma_tile_bytes);
    cb_gamma.reserve_back(Wt);
    for (uint32_t c = 0; c < Wt; ++c) {
        noc.async_read(gamma, cb_gamma, gamma_tile_bytes, {.page_id = c}, {.offset_bytes = c * gamma_tile_bytes});
    }
    noc.async_read_barrier();
    cb_gamma.push_back(Wt);
#endif

    CircularBuffer cb_a(cb::A);
    CircularBuffer cb_b(cb::B);
    cb_a.reserve_back(shard_tiles);
    cb_a.push_back(shard_tiles);
    cb_b.reserve_back(shard_tiles);
    cb_b.push_back(shard_tiles);
}
