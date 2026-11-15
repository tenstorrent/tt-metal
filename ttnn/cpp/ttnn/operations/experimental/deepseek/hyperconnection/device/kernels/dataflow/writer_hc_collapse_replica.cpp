// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "hc_collapse_replica.hpp"

// The collapse cores' writer for collapsed's replica without the fused norm: once the compute has packed this
// core's slice of the ROW_MAJOR collapsed row, send it to the replica (hc_collapse_replica.hpp).
//
// Compile-time args: 0: cb_row   1: row_chunks (its pages)   2: row_bytes (this core's slice)
//   3..6: the replica rectangle's NoC-0 start x, y and end x, y   7: its cores   8: collapse cores
//   9: replica semaphore id
// Runtime args: 0: collapsed address (unused)   1: core_idx   2: replica address   3, 4: the hub's NoC x, y
void kernel_main() {
    constexpr uint32_t cb_row = get_compile_time_arg_val(0);
    constexpr uint32_t row_chunks = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t start_x = get_compile_time_arg_val(3);
    constexpr uint32_t start_y = get_compile_time_arg_val(4);
    constexpr uint32_t end_x = get_compile_time_arg_val(5);
    constexpr uint32_t end_y = get_compile_time_arg_val(6);
    constexpr uint32_t mcast_cores = get_compile_time_arg_val(7);
    constexpr uint32_t num_senders = get_compile_time_arg_val(8);
    constexpr uint32_t sem_id = get_compile_time_arg_val(9);

    const uint32_t core_idx = get_arg_val<uint32_t>(1);
    const uint32_t replica_addr = get_arg_val<uint32_t>(2);
    const uint32_t hub_x = get_arg_val<uint32_t>(3);
    const uint32_t hub_y = get_arg_val<uint32_t>(4);

    CircularBuffer row(cb_row);
    row.wait_front(row_chunks);
    hc_send_replica_slice(
        row.get_read_ptr(),
        row_bytes,
        core_idx,
        replica_addr,
        hub_x,
        hub_y,
        get_semaphore(sem_id),
        num_senders,
        num_senders * row_bytes,
        start_x,
        start_y,
        end_x,
        end_y,
        mcast_cores);
    row.pop_front(row_chunks);
}
