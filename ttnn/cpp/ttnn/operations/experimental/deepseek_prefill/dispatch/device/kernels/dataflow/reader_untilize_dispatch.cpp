// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Reader kernel for idle (worker) cores in the dispatch op.
// Pairs with writer_untilize_dispatch.cpp running on the other data-movement
// RISC — this kernel focuses on streaming tiled input from DRAM so the writer
// can overlap NOC-writes to the owning sender with the next batch's reads.
//
// Token batches are distributed round-robin across total_workers (k_s idle
// cores + the sender itself). Core i processes batches i, i+total_workers, …
//
// For each assigned batch:
//   1. Signal compute to start untilizing this batch (cb_signal_id).
//   2. Read tiled input stripe from DRAM → cb_input_id, block_ct_dim tiles at a time.
//

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

#define ENABLE_DISPATCH_DEBUG 0
#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH DPRINT
#else
#define DPRINT_DISPATCH \
    if (0)              \
    DebugPrinter()
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    // ===== Compile-time args =====
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_signal_id = get_compile_time_arg_val(1);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(2);
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t total_batches = get_compile_time_arg_val(4);
    constexpr uint32_t core_id = get_compile_time_arg_val(5);
    constexpr uint32_t total_workers = get_compile_time_arg_val(6);
    constexpr auto input_args = TensorAccessorArgs<7>();

    constexpr uint32_t tiles_per_row = hidden_size / 32;
    constexpr uint32_t block_ct_dim = 8;
    constexpr uint32_t num_tile_blocks = tiles_per_row / block_ct_dim;

    // ===== Runtime args =====
    uint32_t input_tensor_address = get_arg_val<uint32_t>(0);

    const auto input_addr_gen = TensorAccessor(input_args, input_tensor_address, aligned_input_page_size);

    DPRINT_DISPATCH << "Idle reader " << core_id << "/" << total_workers << " total_batches=" << total_batches
                    << ENDL();

    for (uint32_t batch_idx = core_id; batch_idx < total_batches; batch_idx += total_workers) {
        uint32_t tile_base_page = batch_idx * tiles_per_row;

        // 1. Signal compute to untilize this batch (before streaming tiles so compute is ready)
        cb_reserve_back(cb_signal_id, 1);
        volatile tt_l1_ptr uint32_t* signal_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
        signal_ptr[0] = 0x00000000;
        cb_push_back(cb_signal_id, 1);

        // 2. Stream tiled input stripe from DRAM in blocks of 8 tiles
        for (uint32_t blk = 0; blk < num_tile_blocks; blk++) {
            cb_reserve_back(cb_input_id, block_ct_dim);
            uint32_t blk_write_ptr = get_write_ptr(cb_input_id);
            uint32_t blk_start = tile_base_page + blk * block_ct_dim;
            for (uint32_t col = 0; col < block_ct_dim; col++) {
                noc_async_read_page(blk_start + col, input_addr_gen, blk_write_ptr + col * aligned_input_page_size);
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_id, block_ct_dim);
        }

        DPRINT_DISPATCH << "Idle reader " << core_id << " queued batch " << batch_idx << ENDL();
    }

    // Send sentinel to compute to break out of its loop
    cb_reserve_back(cb_signal_id, 1);
    volatile tt_l1_ptr uint32_t* signal_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_signal_id));
    signal_ptr[0] = ROUTE_INFO_SENTINEL;
    cb_push_back(cb_signal_id, 1);

    DPRINT_DISPATCH << "Idle reader " << core_id << " done" << ENDL();
}
