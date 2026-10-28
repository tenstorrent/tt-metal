// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// clamp_square_practice reader (NCRISC / NoC0) — CANDIDATE: transaction-id pipelined load.
//
// Baseline (op's reader): reserve, `valid` reads, noc_async_read_barrier(), push — the barrier
// drains EVERY outstanding read, so the NoC goes idle once per block.
// Candidate: block k is tagged with trid (k % T) + 1; once T blocks are outstanding the OLDEST id
// is retired with a per-id barrier and pushed, while the T-1 newer blocks stay on the wire.
// Block k and block k-T share an id, so one barrier both retires the old block and frees the id.
//
// LANDING SLOT: the CB write pointer only advances on cb_push_back, so it lags the in-flight
// blocks. The new block lands at get_write_ptr() + in_flight slots (wrapped at the CB end), and
// we reserve (in_flight + 1) blocks so that slot is genuinely free.
//
// CB quantum stays the nominal block_tiles (FullBlock contract with compute), ragged last block
// included: it issues only `valid` reads but pushes block_tiles.
//
// Helper usage: trid tag/wait via dataflow_kernel_lib::set_read_trid / async_read_barrier_with_trid.
// Defines: TRID_DEPTH (T), CB_BLOCKS (CB capacity in blocks), ABLATE_READ (stub the NoC reads).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifndef ABLATE_READ
#define ABLATE_READ 0
#endif

void kernel_main() {
    constexpr uint32_t cb_input_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr auto input_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_blocks = CB_BLOCKS;
    constexpr uint32_t depth = TRID_DEPTH < cb_blocks ? TRID_DEPTH : cb_blocks;
    static_assert(depth >= 1 && depth <= NOC_MAX_TRANSACTION_ID, "trid depth out of range");
    static_assert(block_tiles <= NOC_MAX_TRANSACTION_ID_COUNT, "block exceeds per-id counter");

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);
    const uint32_t core_tiles = get_arg_val<uint32_t>(2);

    const auto input_accessor = TensorAccessor(input_args, src_addr, page_bytes);
    const Noc noc;

    constexpr uint32_t block_bytes = block_tiles * page_bytes;
    constexpr uint32_t cb_bytes = cb_blocks * block_bytes;
    const uint32_t cb_base = get_write_ptr(cb_input_tiles);
    const uint32_t cb_end = cb_base + cb_bytes;
    auto slot = [=](uint32_t from, uint32_t j) {
        const uint32_t a = from + j * block_bytes;
        return a >= cb_end ? a - cb_bytes : a;
    };

    const uint32_t num_blocks = (core_tiles + block_tiles - 1) / block_tiles;

    MaybeDeviceZoneScope("reader_read");
    for (uint32_t k = 0; k < num_blocks; ++k) {
        const uint32_t trid = (k % depth) + 1;
        if (k >= depth) {
            dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid);
            cb_push_back(cb_input_tiles, block_tiles);
        }
        const uint32_t in_flight = (k < depth) ? k : (depth - 1);
        cb_reserve_back(cb_input_tiles, (in_flight + 1) * block_tiles);
        uint32_t l1_write_addr = slot(get_write_ptr(cb_input_tiles), in_flight);
        const uint32_t done = k * block_tiles;
        const uint32_t remaining = core_tiles - done;
        const uint32_t valid = remaining < block_tiles ? remaining : block_tiles;
        const uint32_t page0 = start_tile + done;
        dataflow_kernel_lib::set_read_trid(noc, trid);
        for (uint32_t i = 0; i < valid; ++i) {
            if constexpr (!ABLATE_READ) {
                noc_async_read_page(page0 + i, input_accessor, l1_write_addr);
            }
            l1_write_addr += page_bytes;
        }
    }
    const uint32_t to_drain = num_blocks < depth ? num_blocks : depth;
    for (uint32_t d = 0; d < to_drain; ++d) {
        dataflow_kernel_lib::async_read_barrier_with_trid(noc, ((num_blocks - to_drain + d) % depth) + 1);
        cb_push_back(cb_input_tiles, block_tiles);
    }
    dataflow_kernel_lib::set_read_trid(noc, 0);  // restore untagged
}
