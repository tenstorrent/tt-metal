// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// wq_b_linear x-multicaster (sender core BRISC, on its channel's NoC; the core sits outside the
// worker rectangle). It owns rounds r = first_round, first_round + round_stride, ... (one Mcast
// channel per x sender; the workers receive round r on channel r % round_stride). For each round r:
// wait for the x-reader (NCRISC, the other NoC) to land round r in cb_x_resident, then multicast that slot to the same
// address on every worker and publish it with the Counter data-ready signal (no consumer handshake: every slot is
// written once and never reused). cb_x_resident's address is identical on the sender and the workers (one CB
// descriptor).
//
// HELPER DEVIATION (payload and, with x_ordered_signal, the publish). The Counter-mode SenderPipe::send() issues the
// payload UNLINKED (one path reservation per 16 KB burst) and then blocks on a full write barrier before the counter
// atomic, so no two rounds ever overlap on the NoC (Phase 0 measured ~1 us fixed cost per round; the multicast, not the
// DRAM read, bound the op). send() has no form that defers the signal or orders it behind the payload, so this kernel
// issues the payload itself:
//   * x_mcast_linked:  the bursts of a round are one linked sequence (one path reservation); the
//                      round's last burst is unlinked so no path stays reserved for the atomic.
//   * x_signal_lag 1:  round r's payload is issued BEFORE round r-1 is published; round r-1's
//                      counter goes out as soon as its own acks are in (a partial ack wait on the
//                      NIU's write-ack counter), so payload keeps flowing across the publish.
//   * x_ordered_signal: publish a round by multicasting its count into the receivers' Counter cell
//                      right behind its payload (same path + VC => lands after the data), instead of
//                      a write barrier + atomic increment + atomic barrier per round (~1.1 us each,
//                      measured). Rounds then stream back to back.
//                      Perf 2: the whole stream is ONE linked sequence -- every payload burst and every count
//                      write is linked except the last count write -- so the multicast path is reserved once
//                      for all of x instead of once per round (x multicast alone 14.9 -> 10.3 us, whole op
//                      18.7 -> 16.7 us). Each round's count goes out of its own L1 word (cb_count_src, sender
//                      only), so no flush is needed before writing the next count. Holding the path while
//                      waiting on the x-reader is safe: nothing else multicasts or writes on this NoC across
//                      the receiver rect until x is done (verified under the watcher's NoC sanitizer).
// Without x_ordered_signal the counter protocol is the helper's SenderPipe::send_signal() (Counter
// inc_multicast + atomic barrier). Receivers always use ReceiverPipe::receive(r) (counter >= r + 1).
// x_ordered_signal 0 + x_mcast_linked 0 + x_signal_lag 0 is the Phase 0 schedule.

#include <stdint.h>
#include <type_traits>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
// Zones (permanent, ordered-signal path): x_mcast_wait (waiting on the x-reader per round), x_mcast_issue
// (payload + count multicast issue), x_mcast_barrier (final drain).
// Ablation: ABLATE_X_MCAST skips the payload multicast; the count signal still goes out.
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
    constexpr uint32_t first_round = get_compile_time_arg_val(6);
    constexpr uint32_t round_stride = get_compile_time_arg_val(7);
    constexpr bool x_ordered_signal = get_compile_time_arg_val(8) != 0;
    constexpr uint32_t data_ready_sem_id = get_compile_time_arg_val(9);  // this channel's Counter cell
    constexpr uint32_t cb_count_src = get_compile_time_arg_val(10);      // sender-only, one 16 B word per round
    constexpr uint32_t my_rounds = (num_k_blocks - first_round + round_stride - 1) / round_stride;
    // This sender's channel: exactly one of the two slots is attached (the other is absent).
    using McA = McastArgs<
        get_named_compile_time_arg_val("x_mcast_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_rt_offset")>;
    using McB = McastArgs<
        get_named_compile_time_arg_val("x_mcast_b_ct_offset"),
        get_named_compile_time_arg_val("x_mcast_b_rt_offset")>;
    static_assert(McA::active != McB::active, "a sender kernel carries exactly one x channel");
    constexpr std::conditional_t<McA::active, McA, McB> mc{};

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
    // Round r sits at base + r * round_bytes here and on every worker (x resident, one CB descriptor).
    const uint32_t base = get_read_ptr(cb_x_resident);

    auto issue_payload = [&](uint32_t slot) {
#ifdef ABLATE_X_MCAST
        return;
#endif
        if constexpr (x_ordered_signal) {
            // Every burst linked: only the stream's final count write is unlinked (Perf 2).
            for (uint32_t off = 0; off < round_bytes; off += NOC_MAX_BURST_SIZE) {
                const uint32_t len = round_bytes - off < NOC_MAX_BURST_SIZE ? round_bytes - off : NOC_MAX_BURST_SIZE;
                noc_async_write_multicast(
                    slot + off,
                    get_noc_multicast_addr(dst_x_start, dst_y_start, dst_x_end, dst_y_end, slot + off),
                    len,
                    num_dests,
                    /*linked=*/true);
            }
        } else if constexpr (x_mcast_linked) {
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

    if constexpr (x_ordered_signal) {
        // Publish local round i by multicasting the count i + 1 into every receiver's Counter cell, right
        // behind the round's payload on the same NoC path and VC, so it lands after the data. The value is
        // monotone, so ReceiverPipe::receive(i) (wait counter >= i + 1) is unchanged. The whole stream is one
        // linked sequence (only the final count write is unlinked); round i's count source is its own word.
        const uint32_t cell_addr = get_semaphore(data_ready_sem_id);
        const uint64_t cell_mcast = get_noc_multicast_addr(dst_x_start, dst_y_start, dst_x_end, dst_y_end, cell_addr);
        const uint32_t count_src = get_write_ptr(cb_count_src);
        for (uint32_t i = 0; i < my_rounds; ++i) {
            const uint32_t r = first_round + i * round_stride;
            {
                MaybeDeviceZoneScope("x_mcast_wait");
                cb_wait_front(cb_x_resident, round_tiles);
            }
            {
                MaybeDeviceZoneScope("x_mcast_issue");
                issue_payload(base + r * round_bytes);
                const uint32_t src = count_src + i * 16;
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src) = i + 1;
                noc_async_write_multicast(src, cell_mcast, sizeof(uint32_t), num_dests, /*linked=*/i + 1 < my_rounds);
            }
            cb_pop_front(cb_x_resident, round_tiles);  // slot is never rewritten (x resident)
        }
        {
            MaybeDeviceZoneScope("x_mcast_barrier");
            noc_async_write_barrier();
        }
        return;
    }

    bool pending = false;  // a round whose payload is issued but whose counter is not yet published
    uint32_t pending_acks = 0;
    for (uint32_t i = 0; i < my_rounds; ++i) {
        const uint32_t r = first_round + i * round_stride;
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
        issue_payload(base + r * round_bytes);
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
