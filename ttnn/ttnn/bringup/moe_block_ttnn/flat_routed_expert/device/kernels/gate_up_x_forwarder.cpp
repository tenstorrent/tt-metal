// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// gate_up_x_forwarder: forwards x along the gate/up snake and reads this core's gate/up weights
// Runs on: gate/up cores (p.gu), RISCV_1 (NCRISC), NoC 0
// Responsibilities:
//  - reads its gate/up weight chunks from DRAM over NoC 0 into the landing slots gate_up_core_reader grants
//    (gate_up_weight_read.hpp), bumping DATA as chunks land
//  - unicasts each arrived x piece (XARR) into the successor's x ring (transaction ids, DEPTH in flight) once its
//    slot is free (SFREE), then bumps the successor's XARR
//  - reports min(consumed, forwarded) blocks upstream: predecessor's SFREE or, on a snake head, the primary relay's
//    freed word (x_injector_writer); ends on gate_up_core_reader's end count (XEND)

// Gate/up x forwarder (NCRISC) on a gate/up core, also its weight reads (gate_up_weight_read.hpp): x travels the
// gate/up cores by unicast (no multicast: the NOC0 wedge of a multicast into the gate/up rectangle beside the weight
// and DRAM traffic). The receiver (gate_up_core_reader.cpp, BRISC) only consumes x; this kernel forwards
// each piece as soon as it has arrived (XARR here, in pieces) into the successor's x ring, once the successor has freed
// the block's slot (SFREE here), then bumps the successor's XARR after the write is acknowledged; and reports
// min(consumed (XCONS, published by gate_up_core_reader), forwarded) blocks upstream - the predecessor's SFREE, or on a
// chain head the relay's word. It ends once gate_up_core_reader has published its last count (XEND = blocks + 1) and
// every block is forwarded and reported. Named CT args: x_forward_depth (DEPTH: pieces in flight); flat_options:
// X_SLOTS = x_ring_blocks, X_BYTES = MT x x_block_k_tiles x x_tile_bytes, X_PIECES = x_chain_pieces, XARR_SEM =
// x_arrived_sem, SFREE_SEM = x_slot_free_sem, XCONS_SEM = x_consumed_sem, XEND_SEM = x_end_sem, NB (x blocks a
// sub-block) = super_blocks_per_sub_block x super_block_tiles / x_block_k_tiles, MT = row_tiles_per_sub_block
// RT: 0 x ring address, 1 predecessor xy (0: chain head), 2 sem id of this core's free word on the predecessor,
//     3 successor xy (0: chain tail), 4 relay xy, 5 the relay's word for this core (a chain head), 6.. the
//     gate_up_weight_read.hpp args
#include "../../../common/kernels/moe_block_args.hpp"
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "expert_schedule.hpp"
#include "flat_options.hpp"
#include "gate_up_weight_read.hpp"

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table, or the runtime args: moe_block_args.hpp)
    // (MT: row tiles a block; NB: x blocks a sub-block)
    constexpr uint32_t row_tiles = flat_options::row_tiles_per_sub_block;
    constexpr uint32_t x_slots = flat_options::x_ring_blocks;
    constexpr uint32_t x_bytes = row_tiles * flat_options::x_block_k_tiles * flat_options::x_tile_bytes;
    constexpr uint32_t x_pieces = flat_options::x_chain_pieces, xarr_id = flat_options::x_arrived_sem;
    constexpr uint32_t sfree_id = flat_options::x_slot_free_sem, xcons_id = flat_options::x_consumed_sem;
    constexpr uint32_t xend_id = flat_options::x_end_sem;
    constexpr uint32_t depth = get_named_compile_time_arg_val("x_forward_depth");
    constexpr uint32_t blocks_per_sub_block =
        flat_options::super_blocks_per_sub_block * flat_options::super_block_tiles / flat_options::x_block_k_tiles;
    static_assert(depth >= 1 && depth <= 14);
    constexpr uint32_t piece = x_bytes / x_pieces;
    const uint32_t x_ring = moe_block_args::arg(0);
    const uint32_t pred_xy = moe_block_args::arg(1), succ_xy = moe_block_args::arg(3);
    const uint64_t pred_free = get_noc_addr(pred_xy >> 16, pred_xy & 0xFFFF, get_semaphore(moe_block_args::arg(2)));
    const uint32_t relay_xy = moe_block_args::arg(4);
    const uint64_t relay_word = get_noc_addr(relay_xy >> 16, relay_xy & 0xFFFF, moe_block_args::arg(5));
    const uint64_t s_ring = get_noc_addr(succ_xy >> 16, succ_xy & 0xFFFF, x_ring);
    const uint64_t s_xarr = get_noc_addr(succ_xy >> 16, succ_xy & 0xFFFF, get_semaphore(xarr_id));
    auto sem = [](uint32_t id) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id)); };
    volatile tt_l1_ptr uint32_t* xarr = sem(xarr_id);
    volatile tt_l1_ptr uint32_t* sfree = sem(sfree_id);
    volatile tt_l1_ptr uint32_t* xcons = sem(xcons_id);
    volatile tt_l1_ptr uint32_t* xend = sem(xend_id);
    uint32_t issued = 0, forwarded = 0, reported = 0;  // pieces issued / acknowledged (+ XARR'd), blocks reported
    // gate/up weights read from here (gate_up_weight_read.hpp, RT 6..), into the landing slots the receiver (BRISC)
    // grants; the loads list comes from the receiver (its mailbox word, then the list in this RISC's half of CB 7)
    GateUpWeightReads weight_reads;
    bool g_done = true;
    volatile tt_l1_ptr uint32_t* mailbox =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(schedule_scratch(ScratchOwner::Ncrisc));
    {
        volatile tt_l1_ptr uint32_t* mailbox_set = sem(flat_options::gate_up_mailbox_sem);
        while (*mailbox_set == 0) {
            invalidate_l1_cache();
        }
        weight_reads.init(
            6,
            flat_options::gate_up_block_bytes,
            flat_options::gate_up_blocks_per_expert,
            flat_options::gate_up_reads_in_flight,
            mailbox[0],
            mailbox + 1,
            sem(flat_options::gate_up_grant_sem),
            sem(flat_options::gate_up_data_sem));
        g_done = weight_reads.total == 0;
    }
    // a short last sub-block: its blocks' pieces past its row tiles holding tokens carry nothing - counted (XARR), not
    // written. The entries (sub-blocks | last one's row tiles << 16) follow the loads list in the mailbox.
    const volatile tt_l1_ptr uint32_t* x_entries = mailbox + 1 + mailbox[0];
    const uint32_t n_ent = x_entries[0];
    uint32_t cursor_entry = 0, cursor_sub_block = 0, cursor_end_block = blocks_per_sub_block,
             cursor_pieces = x_pieces;  // the block cursor's entry / sub-block, its end block, its pieces
    auto set_np = [&]() {
        const uint32_t subs = cursor_entry < n_ent ? (x_entries[1 + cursor_entry] & 0xFFFF) : 1u;
        const uint32_t token_row_tiles =
            cursor_entry < n_ent && cursor_sub_block + 1 == subs ? (x_entries[1 + cursor_entry] >> 16) : row_tiles;
        cursor_pieces = (token_row_tiles * x_pieces + row_tiles - 1) / row_tiles;
    };
    set_np();
    while (true) {
        invalidate_l1_cache();
        if (!g_done) {
            g_done = weight_reads.step();
        }
        if (succ_xy) {
            const uint32_t arrived = *xarr;
            while (issued < arrived && issued - forwarded < depth &&
                   (issued / x_pieces < x_slots || *sfree >= issued / x_pieces + 1 - x_slots)) {
                while (issued / x_pieces >= cursor_end_block) {  // (the cursor onto the block's sub-block)
                    cursor_end_block += blocks_per_sub_block;
                    if (cursor_entry < n_ent && ++cursor_sub_block == (x_entries[1 + cursor_entry] & 0xFFFF)) {
                        cursor_sub_block = 0;
                        ++cursor_entry;
                    }
                    set_np();
                }
                if (issued % x_pieces >=
                    cursor_pieces) {  // (an empty piece: nothing to write, its count goes on in order)
                    ++issued;
                    continue;
                }
                const uint32_t off = ((issued / x_pieces) % x_slots) * x_bytes + (issued % x_pieces) * piece;
                for (uint32_t o = 0; o < piece; o += NOC_MAX_BURST_SIZE) {
                    const uint32_t n = piece - o < NOC_MAX_BURST_SIZE ? piece - o : NOC_MAX_BURST_SIZE;
                    noc_async_write_one_packet_with_trid(x_ring + off + o, s_ring + off + o, n, 1 + issued % depth);
                }
                ++issued;
            }
            // (the write command buffer drained first: a transaction id's outstanding count only covers requests the
            // NIU has taken - polled right after issuing, the last write can still sit in the command buffer and the
            // id reads "done" before its data has landed; the read-side twin: FLAT_COMBINE_OVERLAP_WORKLOG.md 6.14)
            while (forwarded < issued && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
            }
            while (forwarded < issued &&
                   ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + forwarded % depth)) {
                noc_semaphore_inc(s_xarr, 1);
                ++forwarded;
            }
        }
        const uint32_t cons = *xcons;
        const uint32_t freed = succ_xy && forwarded / x_pieces < cons ? forwarded / x_pieces : cons;
        if (freed > reported) {
            noc_semaphore_inc(pred_xy ? pred_free : relay_word, freed - reported);
            reported = freed;
        }
        const uint32_t end = *xend;
        if (g_done && end && reported == end - 1 && (!succ_xy || forwarded == (end - 1) * x_pieces)) {
            break;
        }
    }
    weight_reads.finish();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_write_set_trid(0);
}
