// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// nomic_residual_layer_norm — writer (exchange_block + store_block).
//
// exchange_block: all-gather of the packed fp32 (S, Q) statistics inside a group of hidden_split consecutive
// row-major cores. In round c, group member c sends its cb_stats_local pages into slot c of every member's
// cb_stats_gather (its own slot by local copy) and the other members receive. Slot order equals member
// order on every core, so the compute kernel combines the contributors in one fixed order everywhere. The
// gather credit is pushed only after every round of the block has landed.
//
// Transport: one Mcast channel per member index c (fixed sender = member c of each group, prefix "stats<c>").
// EXCHANGE_MODE (host) selects the transport. With one exchange round per call (num_row_blocks == 1) the channels run
// without a handshake: cb_stats_gather is a dedicated, never-aliased buffer that nothing reads before the round, so
// every member multicasts its slot at once and then waits for the others (perf lamp L5). With several rounds per call
// the gather slots are reused, so the channels are handshaked (ChainUnicast for row-wrapping groups) and the rounds run
// in member order. Deviation from op_design (which names ONE rotating-sender channel with ChainUnicast): the Mcast
// helper refuses both a rotating sender on a chain (mcast_impl.cpp "chain forwarding requires one fixed sender") and a
// rotating sender on a non-rectangular group (mcast.cpp "rotating sender requires receiver group to form a rectangle");
// row-major triples on an 11-wide grid wrap rows. G fixed-sender channels driven in the same round order reproduce the
// rotating schedule exactly (one rendezvous, G rounds, member order).
//
// store_block: the output CB is the core's own output shard; the writer only retires its credits after the last
// exchange (the compute kernel may run phase A of block b+1 before block b's normalize: lookahead).

#include <cstdint>
#include <string_view>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp"

using namespace dataflow_kernel_lib;

namespace {

constexpr uint32_t kMaxContributors = 8;
constexpr uint32_t kConcurrentMcast = 2;    // EXCHANGE_MODES index, host side
constexpr uint32_t kConcurrentUnicast = 3;  // EXCHANGE_MODES index, host side
constexpr std::string_view kChannelCt[kMaxContributors] = {
    "stats0_ct_offset",
    "stats1_ct_offset",
    "stats2_ct_offset",
    "stats3_ct_offset",
    "stats4_ct_offset",
    "stats5_ct_offset",
    "stats6_ct_offset",
    "stats7_ct_offset"};
constexpr std::string_view kChannelRt[kMaxContributors] = {
    "stats0_rt_offset",
    "stats1_rt_offset",
    "stats2_rt_offset",
    "stats3_rt_offset",
    "stats4_rt_offset",
    "stats5_rt_offset",
    "stats6_rt_offset",
    "stats7_rt_offset"};

// The Mcast channels (named "stats<c>" arguments) exist in every EXCHANGE_MODE except concurrent_unicast.
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
// Serialized rounds (handshaked channels): in round C member C sends, everybody else receives (and forwards).
template <uint32_t C, uint32_t G>
FORCE_INLINE void exchange_rounds(
    const Noc& noc, uint32_t local_addr, uint32_t gather_addr, uint32_t contributor_bytes, uint32_t block) {
    if constexpr (C < G) {
        constexpr auto channel =
            McastArgs<get_named_compile_time_arg_val(kChannelCt[C]), get_named_compile_time_arg_val(kChannelRt[C])>();
        const uint32_t slot_addr = gather_addr + C * contributor_bytes;
        if (channel.can_send()) {
            channel.sender(noc).send(local_addr, slot_addr, contributor_bytes);  // includes the own-slot copy
        } else {
            channel.receiver(noc).receive_and_forward(slot_addr, contributor_bytes, block);
        }
        exchange_rounds<C + 1, G>(noc, local_addr, gather_addr, contributor_bytes, block);
    }
}

// Concurrent exchange (no handshake; one round per call): every member pushes its slot at once ...
template <uint32_t C, uint32_t G>
FORCE_INLINE void send_own_slot(const Noc& noc, uint32_t local_addr, uint32_t gather_addr, uint32_t contributor_bytes) {
    if constexpr (C < G) {
        constexpr auto channel =
            McastArgs<get_named_compile_time_arg_val(kChannelCt[C]), get_named_compile_time_arg_val(kChannelRt[C])>();
        if (channel.can_send()) {
            channel.sender(noc).send(local_addr, gather_addr + C * contributor_bytes, contributor_bytes);
        }
        send_own_slot<C + 1, G>(noc, local_addr, gather_addr, contributor_bytes);
    }
}

// ... then waits for the other members' slots.
template <uint32_t C, uint32_t G>
FORCE_INLINE void receive_peer_slots(const Noc& noc, uint32_t gather_addr, uint32_t contributor_bytes, uint32_t block) {
    if constexpr (C < G) {
        constexpr auto channel =
            McastArgs<get_named_compile_time_arg_val(kChannelCt[C]), get_named_compile_time_arg_val(kChannelRt[C])>();
        if (!channel.can_send()) {
            channel.receiver(noc).receive_and_forward(gather_addr + C * contributor_bytes, contributor_bytes, block);
        }
        receive_peer_slots<C + 1, G>(noc, gather_addr, contributor_bytes, block);
    }
}

// Own slot by explicit local copy: a fixed-sender multicast whose rectangles exclude the sender skips it.
template <uint32_t C, uint32_t G>
FORCE_INLINE void copy_own_slot(uint32_t local_addr, uint32_t gather_addr, uint32_t contributor_bytes) {
    if constexpr (C < G) {
        constexpr auto channel =
            McastArgs<get_named_compile_time_arg_val(kChannelCt[C]), get_named_compile_time_arg_val(kChannelRt[C])>();
        if (channel.can_send()) {
            noc_async_write(local_addr, get_noc_addr(gather_addr + C * contributor_bytes), contributor_bytes);
        }
        copy_own_slot<C + 1, G>(local_addr, gather_addr, contributor_bytes);
    }
}

#endif  // KERNEL_COMPILE_TIME_ARG_MAP

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_stats_local = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stats_gather = get_compile_time_arg_val(1);
    constexpr uint32_t cb_output = get_compile_time_arg_val(2);
    constexpr uint32_t stat_pages = get_compile_time_arg_val(3);  // stat_tiles pair tiles (S and Q of 2 rows each)
    constexpr uint32_t stat_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t hidden_split = get_compile_time_arg_val(5);  // contributors (group size)
    constexpr uint32_t block_tiles = get_compile_time_arg_val(6);
    constexpr uint32_t num_row_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t exchange_mode = get_compile_time_arg_val(8);  // index into EXCHANGE_MODES (host)
    constexpr uint32_t stats_arrived_semaphore_id = get_compile_time_arg_val(9);
    constexpr bool concurrent_mcast = exchange_mode == kConcurrentMcast;
    constexpr bool concurrent_unicast = exchange_mode == kConcurrentUnicast;
    // RT: [my member index, (noc x, noc y) of every member of my group]
    const uint32_t my_member = get_arg_val<uint32_t>(0);
    static_assert(hidden_split <= kMaxContributors, "add channel names for a larger HIDDEN_SPLIT");

    constexpr uint32_t contributor_bytes = stat_pages * stat_page_bytes;

    CircularBuffer stats_local(cb_stats_local);
    CircularBuffer stats_gather(cb_stats_gather);
    CircularBuffer output(cb_output);
    Noc noc;

    for (uint32_t block = 0; block < num_row_blocks; ++block) {
        // ---- exchange_block ----
        stats_local.wait_front(stat_pages);
        stats_gather.reserve_back(stat_pages * hidden_split);
        {
            const uint32_t local_addr = stats_local.get_read_ptr();
            const uint32_t gather_addr = stats_gather.get_write_ptr();
            if constexpr (concurrent_unicast) {
                // Lamp L5: write my slot into every member's gather CB (mine included), then bump each peer's
                // arrival semaphore once the writes have landed, and wait for all peers' bumps.
                const uint32_t my_slot = gather_addr + my_member * contributor_bytes;
                for (uint32_t member = 0; member < hidden_split; ++member) {
                    const uint32_t x = get_arg_val<uint32_t>(1 + 2 * member);
                    const uint32_t y = get_arg_val<uint32_t>(2 + 2 * member);
                    noc_async_write(local_addr, get_noc_addr(x, y, my_slot), contributor_bytes);
                }
                noc_async_write_barrier();
                const uint32_t arrived_addr = get_semaphore(stats_arrived_semaphore_id);
                for (uint32_t member = 0; member < hidden_split; ++member) {
                    if (member != my_member) {
                        const uint32_t x = get_arg_val<uint32_t>(1 + 2 * member);
                        const uint32_t y = get_arg_val<uint32_t>(2 + 2 * member);
                        noc_semaphore_inc(get_noc_addr(x, y, arrived_addr), 1);
                    }
                }
                noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrived_addr), hidden_split - 1);
                noc_async_atomic_barrier();
            } else {
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
                if constexpr (concurrent_mcast) {
                    copy_own_slot<0, hidden_split>(local_addr, gather_addr, contributor_bytes);
                    send_own_slot<0, hidden_split>(noc, local_addr, gather_addr, contributor_bytes);
                    receive_peer_slots<0, hidden_split>(noc, gather_addr, contributor_bytes, block);
                    noc_async_write_barrier();
                } else {
                    exchange_rounds<0, hidden_split>(noc, local_addr, gather_addr, contributor_bytes, block);
                }
#endif
            }
        }
        stats_gather.push_back(stat_pages * hidden_split);
        stats_local.pop_front(stat_pages);
    }

    // ---- store_block ----
    // The output CB is the core's own (resident) output shard, so nothing moves: the writer only retires the credits,
    // once for the whole shard, after every exchange. Draining per block would hold block b+1's exchange behind
    // block b's normalize and defeat the compute kernel's row-block lookahead.
    output.wait_front(block_tiles * num_row_blocks);
    output.pop_front(block_tiles * num_row_blocks);
}
