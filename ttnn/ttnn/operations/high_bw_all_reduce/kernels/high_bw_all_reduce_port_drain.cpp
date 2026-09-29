// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// high_bw_all_reduce — port_drain (RISCV_0 of the lane's backward-port core; split ports only,
// host SPLIT_PORTS, Refinement 4).
//
// drain_final_block, served per reducer in r's block order (round-robin by readiness, like
// port_bwd): once block k's final is in its final-landing slot ((k mod FD) * W + r) — landed over
// Fabric from p+1, counted by gsem_final_arrival[r] — write its valid pages to output DRAM from
// this core's L1 (bank-run transfers, high_bw_all_reduce_chunk_io.hpp), then publish drained[r] =
// k + 1. Tail blocks need no write (their reducer wrote DRAM itself) and are passed through.
// port_bwd frees a slot only once it has relayed it AND drained[r] covers it.
//
// Why a separate RISC: on a snake / ring middle the relay RISC both forwards every final over
// Fabric and writes it to DRAM; doing both on one RISC and one NoC serializes them per chunk and
// bound the whole pipeline (measured: removing the DRAM write alone gave -10%). Here the DRAM
// write runs concurrently, on this core's other NoC.
// Raw dataflow: DRAM page writes from a raw L1 slot have no kernel_lib helper.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "high_bw_all_reduce_roles.hpp"
#include "high_bw_all_reduce_chunk_io.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t final_depth = get_compile_time_arg_val(2);
    constexpr uint32_t go_sem_id = get_compile_time_arg_val(3);
    constexpr uint32_t CONTROL_WORD_STRIDE = get_compile_time_arg_val(4);
    constexpr uint32_t pos = get_compile_time_arg_val(5);
    constexpr uint32_t group_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_slices = get_compile_time_arg_val(7);
    constexpr uint32_t MAX_REDUCERS = get_compile_time_arg_val(8);
    constexpr bool bank_run_layout = get_compile_time_arg_val(9) != 0;
    constexpr auto output_args = TensorAccessorArgs<10>();
    using Io = ChunkIo<chunk_tiles, tile_bytes, bank_run_layout>;
    constexpr uint32_t chunk_bytes = chunk_tiles * tile_bytes;

    size_t a = 0;
    const uint32_t num_chunks = get_arg_val<uint32_t>(a++);
    const uint32_t num_reducers = get_arg_val<uint32_t>(a++);
    const uint32_t final_addr = get_arg_val<uint32_t>(a++);
    const uint32_t drained_arr = get_arg_val<uint32_t>(a++);
    const uint32_t output_addr = get_arg_val<uint32_t>(a++);
    const uint32_t lane_start = get_arg_val<uint32_t>(a++);
    const uint32_t lane_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t final_arrival_idx = a;  // per-reducer gsem_final_arrival[r] (local, read-only here)

    const auto output = TensorAccessor(output_args, output_addr, tile_bytes);
    auto drained = [&](uint32_t r) {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(drained_arr + r * CONTROL_WORD_STRIDE);
    };

    // port_bwd zeroes the control array (drained[] included) before raising go.
    {
        MaybeDeviceZoneScope("drain_wait_go");
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(go_sem_id)), 1);
    }

    const ChainRoles<pos, group_size, num_slices> roles(num_reducers);
    uint32_t nb[MAX_REDUCERS];
    uint32_t next_k[MAX_REDUCERS];
    uint32_t arrivals[MAX_REDUCERS];  // finals of reducer r seen landed so far
    uint32_t pending = 0;
    for (uint32_t r = 0; r < num_reducers; ++r) {
        nb[r] = roles.blocks_of(r, num_chunks);
        next_k[r] = 0;
        arrivals[r] = 0;
        pending += nb[r];
    }

    MaybePerfAccum(acc_write);  // DRAM write issue + source flush
    uint32_t r = 0;
    {
        MaybeDeviceZoneScope("drain_main");
        while (pending > 0) {
            for (uint32_t scan = 0; scan < num_reducers; ++scan, r = (r + 1 == num_reducers) ? 0 : r + 1) {
                const uint32_t k = next_k[r];
                if (k >= nb[r]) {
                    continue;
                }
                if (!roles.is_tail(r, k, nb[r])) {
                    const auto* arrival =
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(final_arrival_idx + r));
                    if (*arrival <= arrivals[r]) {
                        continue;  // final not here yet
                    }
                    const uint32_t slot_addr = final_addr + ((k % final_depth) * num_reducers + r) * chunk_bytes;
                    const uint32_t chunk_first = (k * num_reducers + r) * chunk_tiles;
                    const uint32_t remaining = lane_tiles - chunk_first;
                    const uint32_t valid_tiles = remaining < chunk_tiles ? remaining : chunk_tiles;
                    MaybePerfBegin(acc_write);
#ifndef HBAR_ABLATE_DRAM_WRITE  // perf ablation only (probes/perf1_bench.py)
                    Io::write(output, lane_start + chunk_first, valid_tiles, slot_addr);
#else
                    (void)sizeof(Io);
#endif
                    noc_async_writes_flushed();  // the slot's bytes have left L1: port_bwd may free it
                    MaybePerfEnd(acc_write);
                    ++arrivals[r];
                }
                next_k[r] = k + 1;
                drained(r)[0] = k + 1;
                --pending;
                r = (r + 1 == num_reducers) ? 0 : r + 1;
                break;
            }
        }
    }
    {
        MaybeDeviceZoneScope("drain_teardown");
        noc_async_write_barrier();
    }
    MaybePerfReport("drain_write", acc_write);
}
