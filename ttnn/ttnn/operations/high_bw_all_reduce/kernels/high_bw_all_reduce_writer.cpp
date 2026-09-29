// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — reducer writer (RISCV_0).
// stage_block, per chunk role (high_bw_all_reduce_roles.hpp):
//   * non-tail: cb_reduced chunk -> the lane port's staging slot ((s mod SD) * W + r), s = this
//     reducer's staging ordinal, then bump staged[r]. Slot reuse is gated by the egress_credit
//     program semaphore, raised by port_fwd when it has forwarded (freed) one of our slots.
//   * tail: the chunk is already the final sum. Write it to the port's final-landing slot
//     ((k mod FD) * W + r), k = block index, and its valid pages to the output DRAM straight from
//     cb_reduced, then bump final_ready[r]. Reuse is gated by final_egress, raised by port_bwd for
//     every chunk of reducer r it frees (the slot's previous occupant may have landed over Fabric).
// Raw dataflow: an on-chip write into another core's raw L1 slot + semaphore inc has no helper.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "high_bw_all_reduce_roles.hpp"
#include "high_bw_all_reduce_chunk_io.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_reduced = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t staging_depth = get_compile_time_arg_val(3);
    constexpr uint32_t final_depth = get_compile_time_arg_val(4);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t egress_sem_id = get_compile_time_arg_val(6);
    constexpr uint32_t final_egress_sem_id = get_compile_time_arg_val(7);
    constexpr uint32_t pos = get_compile_time_arg_val(8);
    constexpr uint32_t group_size = get_compile_time_arg_val(9);
    constexpr uint32_t num_slices = get_compile_time_arg_val(10);
    constexpr bool bank_run_layout = get_compile_time_arg_val(11) != 0;  // host BANK_RUN_LAYOUT
    constexpr auto output_args = TensorAccessorArgs<12>();
    using Io = ChunkIo<chunk_tiles, tile_bytes, bank_run_layout>;
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;

    uint32_t a = 0;
    const uint32_t reducer_idx = get_arg_val<uint32_t>(a++);
    const uint32_t num_reducers = get_arg_val<uint32_t>(a++);
    const uint32_t num_blocks = get_arg_val<uint32_t>(a++);
    const uint32_t port_x = get_arg_val<uint32_t>(a++);  // port_bwd core: final landing + final_ready
    const uint32_t port_y = get_arg_val<uint32_t>(a++);
    const uint32_t fwd_x = get_arg_val<uint32_t>(a++);  // port_fwd core: staging + staged
    const uint32_t fwd_y = get_arg_val<uint32_t>(a++);
    const uint32_t staging_base = get_arg_val<uint32_t>(a++);
    const uint32_t staged_word_addr = get_arg_val<uint32_t>(a++);
    const uint32_t final_base = get_arg_val<uint32_t>(a++);
    const uint32_t final_ready_word_addr = get_arg_val<uint32_t>(a++);
    const uint32_t output_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lane_start = get_arg_val<uint32_t>(a++);
    const uint32_t lane_tiles = get_arg_val<uint32_t>(a++);
    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);
    const ChainRoles<pos, group_size, num_slices> roles(num_reducers);

    {
        MaybeDeviceZoneScope("writer_wait_go");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id)), 1);
    }
    volatile tt_l1_ptr uint32_t* egress_credit =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(egress_sem_id));
    volatile tt_l1_ptr uint32_t* final_egress =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(final_egress_sem_id));
    const uint64_t staged_noc = get_noc_addr(fwd_x, fwd_y, staged_word_addr);
    const uint64_t final_ready_noc = get_noc_addr(port_x, port_y, final_ready_word_addr);

    MaybePerfAccum(acc_reduced_wait);  // cb_reduced empty: compute (or its inputs) is behind
    MaybePerfAccum(acc_slot_wait);     // staging / final-landing slot not yet freed by the port
    MaybePerfAccum(acc_write);         // L1->port (+ tail DRAM) write issue + barrier
    uint32_t staged_count = 0;         // staging ordinal s
    {
        MaybeDeviceZoneScope("writer_main");
        for (uint32_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
            const uint32_t chunk = block_idx * num_reducers + reducer_idx;
            MaybePerfBegin(acc_reduced_wait);
            cb_wait_front(cb_reduced, chunk_tiles);
            MaybePerfEnd(acc_reduced_wait);
            const uint32_t src = get_read_ptr(cb_reduced);
            if (roles.is_tail(reducer_idx, block_idx, num_blocks)) {
                MaybePerfBegin(acc_slot_wait);
                if (block_idx >= final_depth) {
                    noc_semaphore_wait_min(final_egress, block_idx - final_depth + 1);
                }
                MaybePerfEnd(acc_slot_wait);
                MaybePerfBegin(acc_write);
                const uint32_t slot = (block_idx % final_depth) * num_reducers + reducer_idx;
                noc_async_write(src, get_noc_addr(port_x, port_y, final_base + slot * chunk_bytes), chunk_bytes);
                const uint32_t chunk_first = chunk * chunk_tiles;
                const uint32_t remaining = lane_tiles - chunk_first;
                const uint32_t valid_tiles = remaining < chunk_tiles ? remaining : chunk_tiles;
#ifndef HBAR_ABLATE_DRAM_WRITE  // perf ablation only (probes/perf1_bench.py)
                Io::write(output, lane_start + chunk_first, valid_tiles, src);
#else
                (void)sizeof(Io);
#endif
                noc_async_write_barrier();
                MaybePerfEnd(acc_write);
                noc_semaphore_inc(final_ready_noc, 1);
            } else {
                MaybePerfBegin(acc_slot_wait);
                if (staged_count >= staging_depth) {
                    noc_semaphore_wait_min(egress_credit, staged_count - staging_depth + 1);
                }
                MaybePerfEnd(acc_slot_wait);
                MaybePerfBegin(acc_write);
                const uint32_t slot = (staged_count % staging_depth) * num_reducers + reducer_idx;
                noc_async_write(src, get_noc_addr(fwd_x, fwd_y, staging_base + slot * chunk_bytes), chunk_bytes);
                noc_async_write_barrier();
                MaybePerfEnd(acc_write);
                noc_semaphore_inc(staged_noc, 1);
                ++staged_count;
            }
            cb_pop_front(cb_reduced, chunk_tiles);
        }
    }
    MaybePerfReport("writer_reduced_wait", acc_reduced_wait);
    MaybePerfReport("writer_slot_wait", acc_slot_wait);
    MaybePerfReport("writer_write", acc_write);
    noc_async_atomic_barrier();
}
