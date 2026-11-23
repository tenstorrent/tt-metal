// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x helper reader (perf experiment x_read_fast, X_HELPER): a second core (NCRISC) that reads the
// odd x rounds from DRAM on its own NoC (read_noc) into its copy of cb_x_resident (same address as on the sender),
// then writes each round to the same address on the x sender over the OTHER NoC (one 20 KB write per round,
// so the sender's read-response ingress NoC carries only the sender's own reads) and publishes it by writing
// the round count into the sender's cell right behind the payload (same cmd buf, path and VC -> lands after it).
//
// Why a second core: one core's DRAM reads of 2 KB pages are capped at ~70-80 GB/s no matter how many NoCs or
// how cheap the issue (measured in this dir: untagged all-in-flight 6.95 us, both NoCs 6.96 us, one-packet
// stateful issue 6.72 us; 8 KB requests on both NoCs 3.5 us). The x layout forbids larger requests (bank-adjacent
// pages land 4 rounds apart in L1), so the request rate is raised by adding a requester.
//
// Raw NoC calls (no kernel_lib helper): the publish-behind-payload ordering on a unicast write has no helper
// form (Mcast SenderPipe is multicast-only and its Counter publish waits on a write barrier).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t tensor_k_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t x_read_trids = get_compile_time_arg_val(5);
    constexpr uint32_t first_round = get_compile_time_arg_val(6);   // 1
    constexpr uint32_t round_stride = get_compile_time_arg_val(7);  // 2
    constexpr uint32_t cell_sem_id = get_compile_time_arg_val(8);
    constexpr auto x_args = TensorAccessorArgs<9>();
    constexpr uint32_t my_rounds = (num_k_blocks - first_round + round_stride - 1) / round_stride;
    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;

    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t sender_x = get_arg_val<uint32_t>(1);
    const uint32_t sender_y = get_arg_val<uint32_t>(2);

    const uint32_t x_tile_bytes = get_tile_size(cb_x_resident);
    const uint32_t round_bytes = round_tiles * x_tile_bytes;
    const auto x_acc = TensorAccessor(x_args, x_addr, x_tile_bytes);
    constexpr uint8_t rd_noc = noc_index;
    constexpr uint8_t wr_noc = 1 - noc_index;
    // No noc_local_state_init ran for wr_noc on this RISC; nothing else writes on it from this core.
    noc_nonposted_writes_num_issued[wr_noc] = NOC_STATUS_READ_REG(wr_noc, NIU_MST_NONPOSTED_WR_REQ_SENT);
    noc_nonposted_writes_acked[wr_noc] = NOC_STATUS_READ_REG(wr_noc, NIU_MST_WR_ACK_RECEIVED);

    const uint32_t base = get_write_ptr(cb_x_resident);
    const uint32_t cell_addr = get_semaphore(cell_sem_id);
    volatile tt_l1_ptr uint32_t* cell = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cell_addr);
    const uint64_t sender_cell = get_noc_addr(sender_x, sender_y, cell_addr, wr_noc);
    const uint64_t sender_base = get_noc_addr(sender_x, sender_y, 0, wr_noc);

    auto issue_round = [&](uint32_t r) {
        uint32_t dst = base + r * round_bytes;
        for (uint32_t m = 0; m < block_m_tiles; ++m) {
            uint32_t page = m * tensor_k_tiles + r * block_k_tiles;
            for (uint32_t kk = 0; kk < block_k_tiles; ++kk) {
#ifndef ABLATE_X_READ
                noc_async_read(x_acc.get_noc_addr(page, 0, rd_noc), dst, x_tile_bytes, rd_noc);
#endif
                ++page;
                dst += x_tile_bytes;
            }
        }
    };
    auto trid_of = [](uint32_t i) { return (i % x_read_trids) + 1; };

    uint32_t issued = 0;
    {
        MaybeDeviceZoneScope("x_help");
        for (uint32_t i = 0; i < my_rounds; ++i) {
            while (issued < my_rounds && issued < i + x_read_trids) {
                noc_async_read_set_trid(trid_of(issued), rd_noc);
                issue_round(first_round + issued * round_stride);
                ++issued;
            }
            noc_async_read_barrier_with_trid(trid_of(i), rd_noc);
            const uint32_t slot = base + (first_round + i * round_stride) * round_bytes;
            noc_async_write(slot, sender_base | slot, round_bytes, wr_noc);
            noc_async_writes_flushed(wr_noc);  // previous count's source read is done before rewriting the cell
            *cell = i + 1;
            noc_async_write(cell_addr, sender_cell, sizeof(uint32_t), wr_noc);
        }
    }
    noc_async_read_set_trid(0, rd_noc);
    noc_async_read_barrier(rd_noc);
    noc_async_write_barrier(wr_noc);
}
