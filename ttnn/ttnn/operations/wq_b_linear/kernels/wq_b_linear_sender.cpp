// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-multicaster (sender core, BRISC / NoC1; the core sits outside the worker rectangle).
// For each round r: wait for the x-reader (NCRISC / NoC0) to land round r in cb_x_resident, then
// multicast that slot to the same address on every worker and publish it with the Counter
// data-ready signal (no consumer handshake: every slot is written once and never reused).
// cb_x_resident's address is identical on the sender and the workers (one CB descriptor).
//
// HELPER DEVIATION (payload only). The Counter-mode SenderPipe::send() issues the payload
// UNLINKED (one path reservation per 16 KB burst) and then blocks on a full write barrier before the
// counter atomic, so no two rounds ever overlap on the NoC (Phase 0 measured ~1 us fixed cost per
// round; the multicast, not the DRAM read, bound the op). send() has no form that defers the
// signal, so this kernel issues the payload itself:
//   * x_mcast_linked:  the bursts of a round are one linked sequence (one path reservation); the
//                      round's last burst is unlinked so no path stays reserved for the atomic.
//   * x_signal_lag 1:  round r's payload is issued BEFORE round r-1 is published; round r-1's
//                      counter goes out as soon as its own acks are in (a partial ack wait on the
//                      NIU's write-ack counter), so payload keeps flowing across the publish.
// The counter protocol itself stays the helper's: SenderPipe::send_signal() (Counter inc_multicast
// + atomic barrier), paired with ReceiverPipe::receive(r) on the workers. x_mcast_linked 0 +
// x_signal_lag 0 is the Phase 0 schedule.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_x_resident = get_compile_time_arg_val(0);
    constexpr uint32_t block_m_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t block_k_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_k_blocks = get_compile_time_arg_val(3);
    constexpr bool x_mcast_linked = get_compile_time_arg_val(4) != 0;
    constexpr bool x_signal_lag = get_compile_time_arg_val(5) != 0;
    constexpr auto mc = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>();

    // Receiver rectangle in NoC routing order for this RISC's NoC (host-prepared), and its size.
    const uint32_t dst_x_start = get_arg_val<uint32_t>(0);
    const uint32_t dst_y_start = get_arg_val<uint32_t>(1);
    const uint32_t dst_x_end = get_arg_val<uint32_t>(2);
    const uint32_t dst_y_end = get_arg_val<uint32_t>(3);
    const uint32_t num_dests = get_arg_val<uint32_t>(4);

    constexpr uint32_t round_tiles = block_m_tiles * block_k_tiles;
    const uint32_t round_bytes = round_tiles * get_tile_size(cb_x_resident);

    Noc noc;
    auto pipe = mc.sender(noc);

    auto issue_payload = [&](uint32_t slot) {
        if constexpr (x_mcast_linked) {
            uint32_t off = 0;
            for (; off + NOC_MAX_BURST_SIZE < round_bytes; off += NOC_MAX_BURST_SIZE) {
                noc_async_write_multicast(
                    slot + off,
                    get_noc_multicast_addr(dst_x_start, dst_y_start, dst_x_end, dst_y_end, slot + off),
                    NOC_MAX_BURST_SIZE,
                    num_dests,
                    /*linked=*/true);
            }
            noc_async_write_multicast(
                slot + off,
                get_noc_multicast_addr(dst_x_start, dst_y_start, dst_x_end, dst_y_end, slot + off),
                round_bytes - off,
                num_dests,
                /*linked=*/false);
        } else {
            noc_async_write_multicast(
                slot,
                get_noc_multicast_addr(dst_x_start, dst_y_start, dst_x_end, dst_y_end, slot),
                round_bytes,
                num_dests,
                /*linked=*/false);
        }
    };
    // Partial write-ack wait: every ack up to `target` (a snapshot of the issued-ack count) is in.
    auto acks_in = [](uint32_t target) {
        return static_cast<int32_t>(NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) - target) >= 0;
    };

    bool pending = false;  // a round whose payload is issued but whose counter is not yet published
    uint32_t pending_acks = 0;
    for (uint32_t r = 0; r < num_k_blocks; ++r) {
        if constexpr (x_signal_lag) {
            // Publish the previous round as soon as its acks land, even while round r is still being read.
            while (!cb_pages_available_at_front(cb_x_resident, round_tiles)) {
                if (pending && acks_in(pending_acks)) {
                    pipe.template send_signal<SourceL1Guard::CallerManaged>();
                    pending = false;
                }
            }
        }
        cb_wait_front(cb_x_resident, round_tiles);
        issue_payload(get_read_ptr(cb_x_resident));
        if constexpr (x_signal_lag) {
            if (pending) {
                while (!acks_in(pending_acks)) {
                }
                pipe.template send_signal<SourceL1Guard::CallerManaged>();
            }
            pending_acks = noc_nonposted_writes_acked[noc_index];
            pending = true;
        } else {
            noc_async_write_barrier();
            pipe.template send_signal<SourceL1Guard::CallerManaged>();
        }
        cb_pop_front(cb_x_resident, round_tiles);  // slot is never rewritten (x resident)
    }
    if (pending) {
        noc_async_write_barrier();
        pipe.template send_signal<SourceL1Guard::CallerManaged>();
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
