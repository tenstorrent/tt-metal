// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "hc_collapse_replica.hpp"

// The fused collapsed RMSNorm's writer on the collapse cores (see FusedSingleUserInputs::collapsed_norm_weight).
// Each core writes its sum(x^2) tile into slot core_idx of every collapse core's cb_gather (its own included) and
// raises their semaphore; once all num_cores have arrived the compute sums them. It then writes the core's normed
// D/8 columns to their place in the DRAM row.
//
// Compile-time args:
//   0: cb_stat   1: cb_gather   2: cb_out   3: row_chunks (cb_out's pages)   4: stat_tile_bytes
//   5: row_bytes (the core's columns)   6: num_cores   7: arrive semaphore id
//   8: out_sharded (collapsed is L1 WIDTH_SHARDED [1, D/8]: the row goes to this core's shard)
//   9: has_replica (also multicast the row to collapsed's replica)   10..13: the replica rectangle's NoC-0 start x,
//   y and end x, y   14: its cores   15: replica semaphore id
//   16+: TensorAccessorArgs(out) (DRAM interleaved; unused when sharded)
//
// Runtime args:
//   0: out address   1: core_idx   2..2+2*num_cores: the collapse cores' NoC (x, y), in core_idx order, then the
//   replica address and the hub's NoC x, y (hc_collapse_replica.hpp)
void kernel_main() {
    constexpr uint32_t cb_stat = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gather = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t row_chunks = get_compile_time_arg_val(3);
    constexpr uint32_t stat_tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t arrive_sem_id = get_compile_time_arg_val(7);
    constexpr bool out_sharded = get_compile_time_arg_val(8) != 0;
    constexpr bool has_replica = get_compile_time_arg_val(9) != 0;
    constexpr uint32_t rep_start_x = get_compile_time_arg_val(10);
    constexpr uint32_t rep_start_y = get_compile_time_arg_val(11);
    constexpr uint32_t rep_end_x = get_compile_time_arg_val(12);
    constexpr uint32_t rep_end_y = get_compile_time_arg_val(13);
    constexpr uint32_t rep_cores = get_compile_time_arg_val(14);
    constexpr uint32_t rep_sem_id = get_compile_time_arg_val(15);
    constexpr auto out_args = TensorAccessorArgs<16>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t core_idx = get_arg_val<uint32_t>(1);

    const uint32_t arrive_sem_addr = get_semaphore(arrive_sem_id);
    auto* arrive_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrive_sem_addr);

    CircularBuffer stat(cb_stat);
    CircularBuffer gather(cb_gather);
    gather.reserve_back(num_cores);
    // cb_gather sits at the same address on every collapse core.
    const uint32_t slot_addr = gather.get_write_ptr() + core_idx * stat_tile_bytes;

    stat.wait_front(1);
    const uint32_t stat_addr = stat.get_read_ptr();
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t x = get_arg_val<uint32_t>(2 + 2 * i);
        const uint32_t y = get_arg_val<uint32_t>(3 + 2 * i);
        noc_async_write(stat_addr, get_noc_addr(x, y, slot_addr), stat_tile_bytes);
    }
    noc_async_write_barrier();
    stat.pop_front(1);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const uint32_t x = get_arg_val<uint32_t>(2 + 2 * i);
        const uint32_t y = get_arg_val<uint32_t>(3 + 2 * i);
        noc_semaphore_inc(get_noc_addr(x, y, arrive_sem_addr), 1);
    }
    noc_semaphore_wait(arrive_sem_ptr, num_cores);
    noc_semaphore_set(arrive_sem_ptr, 0);
    gather.push_back(num_cores);

    CircularBuffer out(cb_out);
    out.wait_front(row_chunks);
    if constexpr (out_sharded) {
        noc_async_write(out.get_read_ptr(), get_noc_addr(out_addr), row_bytes);
    } else {
        const auto out_acc = TensorAccessor(out_args, out_addr);
        noc_async_write(out.get_read_ptr(), out_acc.get_noc_addr(0, core_idx * row_bytes), row_bytes);
    }
    if constexpr (has_replica) {
        hc_send_replica_slice(
            out.get_read_ptr(),
            row_bytes,
            core_idx,
            get_arg_val<uint32_t>(2 + 2 * num_cores),
            get_arg_val<uint32_t>(3 + 2 * num_cores),
            get_arg_val<uint32_t>(4 + 2 * num_cores),
            get_semaphore(rep_sem_id),
            num_cores,
            num_cores * row_bytes,
            rep_start_x,
            rep_start_y,
            rep_end_x,
            rep_end_y,
            rep_cores);
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    out.pop_front(row_chunks);
}
