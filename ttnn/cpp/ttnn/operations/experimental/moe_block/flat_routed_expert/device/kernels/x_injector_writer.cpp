// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// x_injector_writer: feeds a gate/up rectangle's x snake from the primary relay's landing rings
// Runs on: x primary relays (p.relays[0..NR-1], one kernel per primary), RISCV_1 (NCRISC), NoC 0
// Responsibilities:
//  - reads the helpers' tilized super-blocks from its own L1 landing rings (written by x_injector_helper_writer)
//  - unicasts each x block as MT pieces into the snake head's x ring (a gate/up core), transaction-id tracked,
//    kChainDepth batches in flight; bumps the head's XARR as batches are acked
//  - waits on helper arrivals (sems 4 / 7), on the head's freed word (bumped by gate_up_x_forwarder), and returns
//    freed landing slots to the helper (flat_options::x_credit_sem there)

// Flat expert x injector writer (an x primary relay's NCRISC). Its source is the helpers' tilized super-blocks (MT row
// tiles x super_block_tiles K tiles, row tile major) in its landing rings (x read once by the helper relays:
// super-block b from helper b % kXnh, x_injector_helper_writer.cpp): each of a super-block's x blocks ([MT x KBLK]
// tiles, row tile major in the gate/up cores' x ring) goes out as MT unicast pieces into the rectangle's snake head,
// once the whole super-block's ring slots are free there (the head's freed word); the gate/up cores forward it along
// the snake (gate_up_core_reader.cpp / gate_up_x_forwarder.cpp). The head's XARR moves by a batch's pieces once its
// writes are acknowledged. Named CT args (flat_options): MT = row_tiles_per_sub_block, TILE_BYTES = x_tile_bytes,
// X_SLOTS = x_ring_blocks, XARR_SEM = x_arrived_sem, KBLK = x_block_k_tiles, NUM_E = local_experts, NSB =
// super_blocks_per_sub_block
// RT: 0 x ring address, 1 snake head xy, 2 the head's freed word address (here), 3 helper 0 xy, 4 landing rings
//     address, 5 landing slots, 6 helper 1 xy, 7.. the expert_schedule.hpp args
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/moe_block_args.hpp"
#include <stdint.h>
constexpr uint32_t kXnh = 2;  // the helper relays feeding a primary
#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "flat_options.hpp"
#include "zz_probes.hpp"
#include "expert_schedule.hpp"

void kernel_main() {
    moe_block_args::load();  // (this core's args from the DRAM table, or the runtime args: moe_block_args.hpp)
    constexpr uint32_t mt = flat_options::x_row_tiles;
    constexpr uint32_t tb = flat_options::x_tile_bytes;
    constexpr uint32_t x_slots = flat_options::x_ring_blocks;
    constexpr uint32_t xarr_sem = flat_options::x_arrived_sem;
    constexpr uint32_t kblk = flat_options::x_block_k_tiles;
    constexpr uint32_t num_e = flat_options::local_experts, nsb = flat_options::super_blocks_per_sub_block;
    constexpr uint32_t sbt = flat_options::super_block_tiles;
    constexpr uint32_t per_sb = sbt / kblk;
    constexpr uint32_t sb_tiles = mt * sbt;
    static_assert(sbt % kblk == 0);
    constexpr uint32_t blk_bytes = mt * kblk * tb, piece = kblk * tb;
    const uint32_t ring = moe_block_args::arg(0);
    const uint32_t hxy = moe_block_args::arg(1);
    // (MEASUREMENT FLAT_X_MCAST=1: x multicast over the whole rectangle - RT 1 its NOC0 corners, 8 bits each: start x,
    // y, end x, y - no snake; every gate/up core reports its freed blocks into its own word here (RT 2 + 4 i), the
    // slots free once the slowest has; the XARR count multicast from the last word)
    // (bits: 1 multicast, 2 linked within a batch, 4 posted data + the XARR multicast behind it on the same VC;
    // x_mcast_vc: the VC, 0 the default multicast VC)
    constexpr uint32_t mcast_bits = get_named_compile_time_arg_val("x_mcast");
    constexpr bool mcast = (mcast_bits & 1) != 0, mc_linked = (mcast_bits & 2) != 0, mc_posted = (mcast_bits & 4) != 0;
    constexpr uint32_t mc_vc = get_named_compile_time_arg_val("x_mcast_vc")
                                   ? get_named_compile_time_arg_val("x_mcast_vc")
                                   : NOC_MULTICAST_WRITE_VC;
    constexpr uint32_t kWords = 64;
    const uint32_t msx = hxy >> 24, msy = (hxy >> 16) & 0xFF, mex = (hxy >> 8) & 0xFF, mey = hxy & 0xFF;
    const uint32_t ndst = mcast ? (mex - msx + 1) * (mey - msy + 1) : 1;
    const uint64_t head_xarr = mcast ? get_noc_multicast_addr(msx, msy, mex, mey, get_semaphore(xarr_sem))
                                     : get_noc_addr(hxy >> 16, hxy & 0xFFFF, get_semaphore(xarr_sem));
    volatile tt_l1_ptr uint32_t* freed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(moe_block_args::arg(2));
    // K split (flat_options::gate_up_k_split): a snake a K part - super-block b of an x sub-block goes to part
    // (b % NSB) x KS / NSB's head (RT after the schedule args: part 1's head xy); each head its freed word (word kp),
    // its ring slots (sent_kp), its XARR
    constexpr uint32_t ks = flat_options::gate_up_k_split;
    static_assert(ks == 1 || !mcast, "the K split: unicast snakes");
    uint32_t hxy_k[2] = {hxy, 0};
    uint64_t head_xarr_k[2] = {head_xarr, 0};
    uint32_t sent_k[2] = {0, 0};
    // the freed words restart every launch: zeroed here (a head only bumps its word after consuming x this relay sent
    // in this launch), so no host write is needed between launches
    for (uint32_t i = 0; i < (mcast ? kWords : ks); ++i) {
        freed[i] = 0;
    }
    // (16-byte aligned, as the semaphore it is multicast to: the NoC needs source and destination alike modulo 16)
    volatile tt_l1_ptr uint32_t* xarr_word = freed + kWords - 4;
    uint32_t xarr_val = 0;
    auto freed_min = [&]() {
        uint32_t m = freed[0];
        for (uint32_t i = 1; i < ndst; ++i) {
            m = freed[i] < m ? freed[i] : m;
        }
        return m;
    };
    auto send_xarr = [&]() {
        *xarr_word = xarr_val;
        while (!noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
        ncrisc_noc_fast_write_any_len<noc_mode>(
            noc_index,
            write_cmd_buf,
            reinterpret_cast<uint32_t>(xarr_word),
            head_xarr,
            4,
            mc_vc,
            true,
            false,
            ndst,
            true);
    };
    // (multicast batches retire on the write-ack count: the software count after each batch is its target)
    uint32_t ack_target[8] = {0};
    ZZ_COUNTER(zz_ack);
    const uint32_t land = moe_block_args::arg(4), land_slots = moe_block_args::arg(5);
    const uint32_t helper_xy[kXnh] = {moe_block_args::arg(3), moe_block_args::arg(6)};
    // The schedule: the super-blocks follow from the active experts' sub-blocks (RT 7.. are the expert_schedule.hpp
    // args, CB 7's upper half this RISC's scratch)
    constexpr uint32_t schedule_args = 7;
    ExpertSchedule schedule;
    load_schedule<num_e>(schedule, schedule_args, schedule_scratch(ScratchOwner::Ncrisc));
    to_x_schedule(schedule);
    if constexpr (ks > 1) {
        hxy_k[1] = moe_block_args::arg(schedule_args + kScheduleArgs);
        head_xarr_k[1] = get_noc_addr(hxy_k[1] >> 16, hxy_k[1] & 0xFFFF, get_semaphore(xarr_sem));
    }
    uint8_t batch_kp[8] = {0};  // (each batch in flight: its K part's head)
    const uint32_t num_sb = schedule.total_sub_blocks * nsb;
    constexpr uint32_t kChainDepth = 4;  // batches of head writes in flight (transaction ids 1..kChainDepth)
    // Batch b's head writes carry transaction id 1 + b % DEPTH; once they are acknowledged the head's XARR moves by
    // that batch's pieces, in batch order (the head counts a contiguous prefix).
    uint32_t cb_iss = 0, cb_done = 0;
    auto chain_retire = [&](uint32_t keep) {  // retire acknowledged batches; block while more than `keep` are out
        // (the write command buffer drained first: an id's outstanding count only covers requests the NIU has taken -
        // right after issuing, the last write can still sit in the command buffer and the id reads "done" early)
        while (cb_done < cb_iss && !noc_cmd_buf_ready(noc_index, write_cmd_buf)) {
        }
        if constexpr (mcast) {
            auto acked = [&](uint32_t b) {
                return int32_t(NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED) - ack_target[b % 8]) >= 0;
            };
            while (cb_done < cb_iss && (cb_iss - cb_done > keep || acked(cb_done))) {
                ZZ_WAIT(zz_ack, while (!acked(cb_done)){});
                if constexpr (!mc_posted) {
                    xarr_val += per_sb * flat_options::x_chain_pieces;
                    send_xarr();
                }
                ++cb_done;
            }
            return;
        }
        while (cb_done < cb_iss && (cb_iss - cb_done > keep || ncrisc_noc_nonposted_write_with_transaction_id_flushed(
                                                                   noc_index, 1 + cb_done % kChainDepth))) {
            while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + cb_done % kChainDepth)) {
            }
            noc_semaphore_inc(head_xarr_k[batch_kp[cb_done % 8]], per_sb * flat_options::x_chain_pieces);
            ++cb_done;
        }
    };
    ZZ_COUNTER(zz_data);
    ZZ_COUNTER(zz_issue);
    ZZ_COUNTER(zz_flush);
    ZZ_COUNTER(zz_slots);
    [[maybe_unused]] const uint32_t zz_start = ZZ_NOW();
    for (uint32_t b = 0; b < num_sb; ++b) {
        // Super-block b was read and tilized by helper h = b % kXnh (x_injector_helper_writer.cpp) into landing ring h
        // here (ring h follows ring h - 1); DATA sem help_sem[h] here counts arrivals, the CREDIT sem x_credit_sem on
        // the helper counts freed slots.
        const uint32_t hh = b % kXnh, hb = b / kXnh;
        constexpr uint32_t help_sem[kXnh] = {4, 7};
        ZZ_WAIT(
            zz_data, while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(help_sem[hh])) < hb + 1) {
                invalidate_l1_cache();
                chain_retire(kChainDepth);  // (waiting: count acknowledged batches into the head's XARR)
            });
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(help_sem[hh])), hb + 1);
        const uint32_t src = land + (hh * land_slots + hb % land_slots) * sb_tiles * tb;
        const uint32_t kp = ks > 1 ? (b % nsb) * ks / nsb : 0;  // (its K part's snake)
        uint32_t& sent = sent_k[kp];
        const uint32_t ghxy = hxy_k[kp];
        // The whole super-block goes out as one batch once its last block's ring slot is free. Batches in flight are
        // only counted into the head's XARR on retirement: every wait retires the acknowledged ones, else a compute
        // that needs the whole ring (row passes: all Ht / KBLK blocks of a sub-block resident, GLM-5.3 24 of 24)
        // never sees the last batch and never frees one (deadlock).
        invalidate_l1_cache();
        ZZ_WAIT(
            zz_slots, while (sent + per_sb > (mcast ? freed_min() : freed[kp]) + x_slots) {
                chain_retire(kChainDepth);
                invalidate_l1_cache();
            });
        // the batch's pieces into the snake head (unicast, under the batch's transaction id); the head forwards them
        // along the snake, gate_up_core_reader.cpp
        chain_retire(kChainDepth - 1);  // (room for this batch's id)
        // (a short last sub-block: only its row tiles holding tokens go; the head's XARR still counts every piece)
        const uint32_t lr = sub_block_row_tiles(schedule, b / nsb);
        ASSERT(lr >= 1 && lr <= mt);  // (the pieces stay within the block's ring slot)
        [[maybe_unused]] const uint32_t zz_i0 = ZZ_NOW();
        for (uint32_t k = 0; k < per_sb; ++k) {
            const uint32_t dst = ring + ((sent + k) % x_slots) * blk_bytes;
            for (uint32_t m = 0; m < lr; ++m) {
                if constexpr (mcast) {
                    const bool last = k + 1 == per_sb && m + 1 == lr;
                    ncrisc_noc_fast_write_any_len<noc_mode>(
                        noc_index,
                        write_cmd_buf,
                        src + (m * sbt + k * kblk) * tb,
                        get_noc_multicast_addr(msx, msy, mex, mey, dst + m * piece),
                        piece,
                        mc_vc,
                        true,
                        mc_linked && !last,
                        ndst,
                        true,
                        mc_posted);
                    continue;
                }
                constexpr uint32_t x_vc = get_named_compile_time_arg_val("x_vc")
                                              ? get_named_compile_time_arg_val("x_vc")
                                              : NOC_UNICAST_WRITE_VC;
                if constexpr ((zz::iso & 16) != 0) {
                    continue;
                }
                noc_async_write_one_packet_with_trid(
                    src + (m * sbt + k * kblk) * tb,
                    get_noc_addr(ghxy >> 16, ghxy & 0xFFFF, dst + m * piece),
                    piece,
                    1 + cb_iss % kChainDepth,
                    write_cmd_buf,
                    noc_index,
                    x_vc);
            }
        }
        if constexpr (zz::on) {
            zz_issue += get_timestamp_32b() - zz_i0;
        }
        if constexpr (mc_posted) {
            // (the XARR multicast right behind the batch, same VC and path: it lands after the data)
            xarr_val += per_sb * flat_options::x_chain_pieces;
            send_xarr();
        }
        if constexpr (mcast) {
            ack_target[cb_iss % 8] = noc_nonposted_writes_acked[noc_index];
        }
        batch_kp[cb_iss % 8] = kp;
        ++cb_iss;
        chain_retire(kChainDepth);
        sent += per_sb;
        ZZ_WAIT(zz_flush, noc_async_writes_flushed());
        if constexpr (mc_posted) {
            ZZ_WAIT(zz_flush, noc_async_posted_writes_flushed());  // (the landing slot read out)
        }
        noc_semaphore_inc(
            get_noc_addr(helper_xy[hh] >> 16, helper_xy[hh] & 0xFFFF, get_semaphore(flat_options::x_credit_sem)),
            1);  // slot free
    }
    ZZ_EMIT("ZZ-XP-DATAWAIT", zz_data);
    ZZ_EMIT("ZZ-XP-SLOTWAIT", zz_slots);
    ZZ_EMIT("ZZ-XP-ISSUE", zz_issue);
    ZZ_EMIT("ZZ-XP-FLUSH", zz_flush);
    ZZ_EMIT("ZZ-XP-ACK", zz_ack);
    ZZ_EMIT("ZZ-XP-SPAN", ZZ_NOW() - zz_start);
    chain_retire(0);
    noc_async_write_set_trid(0);
    noc_async_write_barrier();
    // leave no NoC transaction in flight (reads, writes, atomics, posted writes): the next program starts clean
    noc_async_full_barrier();
}
