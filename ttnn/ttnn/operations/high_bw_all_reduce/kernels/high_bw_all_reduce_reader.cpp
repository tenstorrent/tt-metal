// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — reducer reader (RISCV_1).
//
// Per block (chunk c = block_idx * W + r of this core's lane):
//   grant_landing_slot    : credit the port's partial_granted[r] word once a landing slot is free
//   load_local_block      : DRAM -> cb_local_input (valid pages only, bank-run layout of
//                           high_bw_all_reduce_chunk_io.hpp; nominal push of chunk_tiles)
//   receive_partial_block : the upstream port_fwd writes the partial over Fabric straight into
//                           cb_remote_partial's backing shard and bumps gsem_partial_arrival once
//                           per chunk (fused inc on its last packet); we then push chunk_tiles.
// Head chunks (per-chunk role, see high_bw_all_reduce_roles.hpp) have no upstream partial: only
// load_local_block runs. Landing slots are consumed in receive order (the q-th received chunk of
// this reducer lands in slot q mod RECV_DEPTH — the upstream port_fwd counts the same sequence).
//
// Raw dataflow (no kernel_lib helper covers TensorAccessor page I/O into a CB, nor a
// Fabric-written landing CB) — see op_design.md "Helpers considered and rejected".

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "high_bw_all_reduce_roles.hpp"
#include "high_bw_all_reduce_chunk_io.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_remote_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_local_input = get_compile_time_arg_val(1);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t recv_depth_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t pos = get_compile_time_arg_val(6);
    constexpr uint32_t group_size = get_compile_time_arg_val(7);
    constexpr uint32_t num_slices = get_compile_time_arg_val(8);
    constexpr bool bank_run_layout = get_compile_time_arg_val(9) != 0;  // host BANK_RUN_LAYOUT
    constexpr auto input_args = TensorAccessorArgs<10>();
    using Io = ChunkIo<chunk_tiles, tile_bytes, bank_run_layout>;

    uint32_t a = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lane_start = get_arg_val<uint32_t>(a++);
    const uint32_t lane_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t reducer_idx = get_arg_val<uint32_t>(a++);
    const uint32_t num_reducers = get_arg_val<uint32_t>(a++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(a++);
    const uint32_t port_x = get_arg_val<uint32_t>(a++);
    const uint32_t port_y = get_arg_val<uint32_t>(a++);
    const uint32_t granted_word_addr = get_arg_val<uint32_t>(a++);
    const uint32_t arrival_addr = get_arg_val<uint32_t>(a++);

    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);
    const ChainRoles<pos, group_size, num_slices> roles(num_reducers);

    const uint64_t granted_noc = get_noc_addr(port_x, port_y, granted_word_addr);
    volatile tt_l1_ptr uint32_t* arrival = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrival_addr);
    uint32_t granted = 0;

    // Blocks with an upstream partial (every non-head chunk of this reducer).
    uint32_t num_recv = 0;
    for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        num_recv += roles.is_head(reducer_idx, block_idx, num_blocks) ? 0 : 1;
    }

    // Grant landing slots while the ring has room: receives [pushed, granted) hold a slot each.
    auto try_grant = [&](uint32_t pushed_recv) {
        while (granted < num_recv && granted - pushed_recv < recv_depth_chunks &&
               cb_pages_reservable_at_back(cb_remote_partial, (granted - pushed_recv + 1) * chunk_tiles)) {
            noc_semaphore_inc(granted_noc, 1);
            ++granted;
        }
    };

    {
        MaybeDeviceZoneScope("reader_wait_go");
        // The port zeroes its control array before raising go.
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id)), 1);
    }

    // Per-block stall totals (one marker each at kernel end; per-block zones would blow the budget).
    MaybePerfAccum(acc_input_reserve);  // cb_local_input full: compute is behind
    MaybePerfAccum(acc_dram_read);      // DRAM read issue + barrier
    MaybePerfAccum(acc_partial_wait);   // landing slot reserve + upstream partial arrival
    uint32_t recv_idx = 0;
    {
        MaybeDeviceZoneScope("reader_main");
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const uint32_t chunk = block_idx * num_reducers + reducer_idx;
            const bool is_head = roles.is_head(reducer_idx, block_idx, num_blocks);
            try_grant(recv_idx);  // grant_landing_slot

            // load_local_block
            const uint32_t chunk_first = chunk * chunk_tiles;
            const uint32_t remaining = lane_tiles - chunk_first;
            const uint32_t valid_tiles = remaining < chunk_tiles ? remaining : chunk_tiles;
            MaybePerfBegin(acc_input_reserve);
            cb_reserve_back(cb_local_input, chunk_tiles);
            MaybePerfEnd(acc_input_reserve);
            MaybePerfBegin(acc_dram_read);
#ifndef HBAR_ABLATE_DRAM_READ  // perf ablation only (probes/perf1_bench.py): payload stubbed, sync kept
            Io::read(input, lane_start + chunk_first, valid_tiles, get_write_ptr(cb_local_input));
#else
            (void)sizeof(Io);
#endif
            noc_async_read_barrier();
            MaybePerfEnd(acc_dram_read);
            cb_push_back(cb_local_input, chunk_tiles);

            // receive_partial_block
            if (!is_head) {
                MaybePerfBegin(acc_partial_wait);
                cb_reserve_back(cb_remote_partial, chunk_tiles);
                while (*arrival <= recv_idx) {  // one arrival increment per landed chunk
                    try_grant(recv_idx);
                }
                MaybePerfEnd(acc_partial_wait);
                cb_push_back(cb_remote_partial, chunk_tiles);
                ++recv_idx;
            }
        }
    }
    MaybePerfReport("reader_input_reserve", acc_input_reserve);
    MaybePerfReport("reader_dram_read", acc_dram_read);
    MaybePerfReport("reader_partial_wait", acc_partial_wait);

    // End of invocation: remove exactly this invocation's arrivals (never reset to 0).
    if (num_recv > 0) {
        noc_semaphore_inc(get_noc_addr(arrival_addr), 0u - num_recv);
    }
    noc_async_atomic_barrier();
    noc_async_write_barrier();
}
