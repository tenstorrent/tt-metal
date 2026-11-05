// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/socket_api.h"

// The persistent op's output to the host (RISCV_1 of the root chip's control core, NoC 1). Iteration i's
// num_out_cores reduced output shards land in half i & 1 of the staging, each bumping sem_d2h_count;
// once all of them have, the row goes out as one D2H socket page and sem_d2h_done counts it, which is
// what lets the control core reuse that half two iterations later. The control core sets sem_d2h_stop
// to kStopBit | (pages to push) at the stop.
//
// Compile-time args:
//   0: d2h socket config address   1: row_bytes (one socket page)   2: num_out_cores
//   3: sem_d2h_count   4: sem_d2h_done   5: sem_d2h_stop
//
// Runtime args:
//   0: staging address (two rows)

constexpr uint32_t kStopBit = 0x80000000u;

void kernel_main() {
    constexpr uint32_t d2h_config_addr = get_compile_time_arg_val(0);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_out_cores = get_compile_time_arg_val(2);
    constexpr uint32_t sem_d2h_count_id = get_compile_time_arg_val(3);
    constexpr uint32_t sem_d2h_done_id = get_compile_time_arg_val(4);
    constexpr uint32_t sem_d2h_stop_id = get_compile_time_arg_val(5);
    const uint32_t staging_addr = get_arg_val<uint32_t>(0);

    noc_write_init_state<write_cmd_buf>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    SocketSenderInterface d2h = create_sender_socket_interface(d2h_config_addr);
    set_sender_socket_page_size(d2h, row_bytes);
    const uint64_t pcie_base =
        (static_cast<uint64_t>(d2h.d2h.data_addr_hi) << 32) | static_cast<uint64_t>(d2h.downstream_fifo_addr);

    auto* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_d2h_count_id));
    auto* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_d2h_done_id));
    auto* stop = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_d2h_stop_id));

    for (uint32_t page = 0;; ++page) {
        // The stop comes only once every page's shards are in, so a count that is still short at the
        // stop means there is no such page.
        bool stopped = false;
        {
            DeviceZoneScopedN("FE_D2H_SHARDS_WAIT");
            while (true) {
                invalidate_l1_cache();
                if (*count >= num_out_cores * (page + 1u)) {
                    break;
                }
                const uint32_t s = *stop;
                if ((s & kStopBit) != 0u && (s & ~kStopBit) == page) {
                    stopped = true;
                    break;
                }
            }
        }
        if (stopped) {
            break;
        }

        DeviceZoneScopedN("FE_D2H_PUSH");
        socket_reserve_pages(d2h, 1);
        uint32_t src = staging_addr + (page & 1u) * row_bytes;
        uint64_t dst = pcie_base + d2h.write_ptr;
        for (uint32_t left = row_bytes; left > 0;) {
            const uint32_t chunk = left > NOC_MAX_BURST_SIZE ? NOC_MAX_BURST_SIZE : left;
            noc_wwrite_with_state<noc_mode, write_cmd_buf, CQ_NOC_SNDL, CQ_NOC_SEND, CQ_NOC_WAIT, true, false>(
                NOC_INDEX, src, d2h.d2h.pcie_xy_enc, dst, chunk, 1);
            src += chunk;
            dst += chunk;
            left -= chunk;
        }
        noc_async_writes_flushed();
        socket_push_pages(d2h, 1);
        socket_notify_receiver(d2h);
        *done = page + 1u;
    }

    update_socket_config(d2h);
    noc_async_write_barrier();
}
