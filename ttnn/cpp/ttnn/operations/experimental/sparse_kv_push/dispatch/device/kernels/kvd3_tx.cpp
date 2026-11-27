// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch v3 TX(d, l) (BRISC): the only core on this (direction, link) with a fabric connection (direct EDM,
// no mux). In order:
//   0. chip barrier (every core of the op zeroes its control block first; see kvd3_common.hpp);
//   1. fence with the neighbour's TX facing back;
//   2. slice exchange (as v2): this chip's plan-table slot o -> owner o's inbox slot `me`, for the owners this link's
//      share of this direction reaches first;
//   3. the loop, which never waits except for an EDM slot:
//      - forward the credits my workers W(d, l, *) owe the neighbour (they freed landing slots of its packets);
//      - take a published staging slot (planners: class 0; workers: class >= 1), deepest class first, whose class has a
//        free landing slot on one of the neighbour's K workers: NoC-read it, free it on its producer, send it as one
//        fused write + increment into that worker's class pool;
//      - close class c (an END(c) packet to each of the K workers) once every producer is done with c and every slot it
//        published is sent; classes close in order;
//      - stop once every class is closed, my workers are done, every credit they owe is forwarded and every packet
//        this TX sent has been credited back.

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "api/debug/kernel_profiler.hpp"
#include "api/debug/dprint.h"
#include "kvd3_common.hpp"

#ifndef KVD_SKIP_XCHG
#define KVD_SKIP_XCHG 0
#endif

#include <cstdint>

using namespace kvd;
using namespace kvd3;
namespace fabric_mesh = tt::tt_fabric::mesh::experimental;

#ifdef KVD_FUSED_PLAN
// the fused plan (kvd4): this kernel runs the chip's dispatch-only plan first (sparse_push_plan.cpp, PLAN_FUSED)
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/plan/device/kernels/sparse_push_plan.cpp"
#endif

void kernel_main() {
#ifdef KVD_FUSED_PLAN
    plan_prologue();
#endif
#ifdef KVD_ABL_TX_DELAY
    const uint32_t tx_t_start = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
#endif
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t L = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr uint32_t P = get_compile_time_arg_val(3);
    constexpr uint32_t G = get_compile_time_arg_val(4);
    constexpr uint32_t B = get_compile_time_arg_val(5);
    constexpr uint32_t pool = get_compile_time_arg_val(6);
    constexpr uint32_t stage = get_compile_time_arg_val(7);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t stripe = get_compile_time_arg_val(9);
    constexpr uint32_t inbox_slot_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t max_payload = get_compile_time_arg_val(11) & ~63u;
    constexpr uint32_t owner_blocks_max = get_compile_time_arg_val(12);
    constexpr uint32_t cb_ctrl = get_compile_time_arg_val(13);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(14);  // (its address: the producers' staging)
    constexpr uint32_t cb_pool = get_compile_time_arg_val(15);   // (its address: the neighbour workers' pools)
    constexpr uint32_t cb_send = get_compile_time_arg_val(16);
    constexpr uint32_t n_cores = get_compile_time_arg_val(17);  // cores of the op on a chip (the barrier's count)
    constexpr uint32_t sem_go = get_compile_time_arg_val(18);   // the reader's release, ring counters, its done flag
    constexpr uint32_t sem_filled = get_compile_time_arg_val(19);
    constexpr uint32_t sem_drained = get_compile_time_arg_val(20);
    constexpr uint32_t sem_rdone = get_compile_time_arg_val(21);
    constexpr uint32_t lanes3 = get_compile_time_arg_val(22);  // 2: lane 1's reader is a helper core (ring B)
    constexpr uint32_t wsplit = get_compile_time_arg_val(23);  // workers' RISCs a core (each signals done)
    constexpr auto table_args = TensorAccessorArgs<24>();
    constexpr auto inbox_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    constexpr uint32_t sbytes = slot_bytes(B, row_bytes);
    constexpr uint32_t rows_off = slot_rows_offset(B);
    constexpr uint32_t words_per_block = stripe / 32;
    constexpr uint32_t bitmap_off = ::sparse_push::plan_table::kBitmapOff;
    constexpr uint32_t dest_chip_ids[N] = DEST_CHIP_ID;
    constexpr uint32_t dest_mesh_ids[N] = DEST_MESH_ID;
    static_assert(K <= kMaxK && G <= kMaxClasses && P + 4 * K <= kMaxProducers);

    const uint32_t inbox_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t fence_addr = get_common_arg_val<uint32_t>(8);
    const uint32_t go_addr = get_common_arg_val<uint32_t>(9);        // arrived_semaphore: the barrier's release
    const uint32_t barrier_addr = get_common_arg_val<uint32_t>(10);  // credit_semaphore on the barrier core
    const uint32_t inbox_sem_addr = get_common_arg_val<uint32_t>(11);
    const uint32_t table_addr = get_common_arg_val<uint32_t>(12);
#ifdef KVD_ACTIVE_FROM_BLOCK
    // (the stage CB: unused on this core until the gate is passed)
    if (skip_chunk(table_addr, align64(get_write_ptr(cb_stage)))) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go)) = 2;  // the reader skips too
        return;
    }
#endif

    uint32_t a = 0;
    const uint32_t present = get_arg_val<uint32_t>(a++);
    const uint32_t bar_x = get_arg_val<uint32_t>(a++);  // the chip barrier core (planner 0)
    const uint32_t bar_y = get_arg_val<uint32_t>(a++);
    const uint32_t ctrl = get_write_ptr(cb_ctrl);
    auto* cw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl);
#ifdef KVD_ABL_ZERO_LOOP
    for (uint32_t i = 0; i < kCtrlWords; ++i) {
        cw[i] = 0;
    }
#else
    zero_ctrl(ctrl);
#endif
    // chip barrier: count in, wait for the release, take it back to zero (persistent word)
    noc_semaphore_inc(get_noc_addr(bar_x, bar_y, barrier_addr), 1);
    noc_async_atomic_barrier();
    auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_addr);
    noc_semaphore_wait_min(go, 1);
    noc_semaphore_set(go, 0);
    if (!present) {
        return;
    }
    (void)n_cores;
    const uint32_t d = get_arg_val<uint32_t>(a++);
    const uint32_t l = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint32_t eth_dir = get_arg_val<uint32_t>(a++);
    const uint32_t nb_chip = get_arg_val<uint32_t>(a++);
    const uint32_t nb_mesh = get_arg_val<uint32_t>(a++);
    const uint32_t nbtx_x = get_arg_val<uint32_t>(a++);  // the neighbour's TX facing back: fence and credits
    const uint32_t nbtx_y = get_arg_val<uint32_t>(a++);
    const uint32_t sync_x = get_arg_val<uint32_t>(a++);  // planner 0 (inbox word), the same core on every chip
    const uint32_t sync_y = get_arg_val<uint32_t>(a++);
    uint32_t nbw_x[K], nbw_y[K];  // the neighbour's workers W(d ^ 1, l, k)
    for (uint32_t k = 0; k < K; ++k) {
        nbw_x[k] = get_arg_val<uint32_t>(a++);
        nbw_y[k] = get_arg_val<uint32_t>(a++);
    }
    uint32_t prod_x[P + 4 * K], prod_y[P + 4 * K];  // producers: planners, then my W(f, l, k) (absent: 0)
    for (uint32_t p = 0; p < P + 4 * K; ++p) {
        prod_x[p] = get_arg_val<uint32_t>(a++);
        prod_y[p] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t n_workers = get_arg_val<uint32_t>(a++);  // W(*, l, *) on this chip: producers of classes >= 1
    const uint32_t hlp_x = get_arg_val<uint32_t>(a++);      // my lane-1 helper (lanes3 = 2)
    const uint32_t hlp_y = get_arg_val<uint32_t>(a++);
    const uint32_t nbh_x = get_arg_val<uint32_t>(a++);  // the neighbour TX's helper: credits go to it too
    const uint32_t nbh_y = get_arg_val<uint32_t>(a++);
    size_t fabric_arg = a;

    const uint32_t t_k0 = stats_clock();
    const auto table = TensorAccessor(table_args, table_addr, inbox_slot_bytes);
    const auto inbox = TensorAccessor(inbox_args, inbox_addr, inbox_slot_bytes);
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fabric_arg);
    conn.open();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* chdr = PacketHeaderPool::allocate_header();
    cb_reserve_back(cb_send, 1);
    const uint32_t send_l1 = align64(get_write_ptr(cb_send));

    // ---- 1. fence
    fabric_mesh::fabric_unicast_noc_unicast_atomic_inc(
        &conn,
        chdr,
        nb_chip,
        nb_mesh,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{get_noc_addr(nbtx_x, nbtx_y, fence_addr), 1, true});
    noc_async_writes_flushed();
    auto* fence = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fence_addr);
    noc_semaphore_wait_min(fence, 1);
    noc_semaphore_inc(get_noc_addr(fence_addr), 0u - 1u);
    noc_async_atomic_barrier();

#ifdef KVD_ABL_FABBENCH
    {  // link probe (timing only, not exact): KVD_ABL_FABBENCH full-size plain writes back to back into the neighbour
       // worker's landing pool, no reader or worker in the loop: the bare per-packet cost of this link at this payload
        const uint64_t dst = get_noc_addr(nbw_x[0], nbw_y[0], align64(get_write_ptr(cb_pool)));
        auto* junk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(send_l1);
        junk[3] = 0;  // (no stale stamp: the probe lands in a real slot; KVD_SEQ's header word and END trailer)
        junk[12] = 0;
        const uint32_t tb0 = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
        for (uint32_t i = 0; i < KVD_ABL_FABBENCH; ++i) {
            fabric_mesh::fabric_unicast_noc_unicast_write(
                &conn, hdr, nb_chip, nb_mesh, send_l1, sbytes, tt::tt_fabric::NocUnicastCommandHeader{dst});
        }
        noc_async_writes_flushed();
        const uint32_t tb1 = *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
        DPRINT(
            "KVDFAB d={} l={} n={} bytes={} ns_per_pkt={}\n",
            d,
            l,
            KVD_ABL_FABBENCH,
            sbytes,
            static_cast<uint32_t>(static_cast<uint64_t>(tb1 - tb0) * 1000 / 1350 / KVD_ABL_FABBENCH));
    }
#endif

#ifdef KVD_ABL_TX_DELAY
    {  // front-end probe (fused plan, E6): no slice leaves before KVD_ABL_TX_DELAY cycles after this kernel started
        while (*reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L) - tx_t_start < KVD_ABL_TX_DELAY) {
        }
    }
#endif
#if defined(KVD_FUSED_PLAN) && !defined(KVD_ABL_FUSED_NOPLAN)
    // (the fused plan: every owner's slot is in the table)
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(KVD_PLAN_READY)), N);
#endif
    // ---- 2. slice exchange (v2's, split over the links instead of lanes); KVD_SKIP_XCHG: the inbox already holds
    // this plan's slices (an earlier call on the same plan build sent them)
#if !KVD_SKIP_XCHG
    {
        noc_async_read(table.get_noc_addr(0), send_l1, 64);
        noc_async_read_barrier();
        const uint32_t active_blocks =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(send_l1)[::sparse_push::plan_table::kWordActiveBlocks];
        // (slot 0's chunk start too: a dense plan writes only slot 0's header; the planners' hot split reads it)
        const uint32_t has_split =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(send_l1)[::sparse_push::plan_table::kWordHasSplit];
        const uint32_t split_block =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(send_l1)[::sparse_push::plan_table::kWordSplitBlock];
        auto owned_by = [&](uint32_t o) {
            const uint32_t b = active_blocks > o ? (active_blocks - o + N - 1) / N : 0u;
            return b < owner_blocks_max ? b : owner_blocks_max;
        };
        uint32_t nth = 0;
        for (uint32_t o = 0; o < N; ++o) {  // (table slot o: owner o, on chip owner_chip(o))
            const uint32_t pc = owner_chip(o);
            if (pc == me) {
                continue;
            }
            if (static_cast<uint32_t>(
                    tt::tt_fabric::get_next_hop_router_direction(dest_mesh_ids[pc], dest_chip_ids[pc])) != eth_dir) {
                continue;
            }
            if (nth++ % L != l) {
                continue;
            }
            const uint32_t bytes = ((bitmap_off + owned_by(o) * words_per_block) * 4 + 63) & ~63u;
            const uint64_t inbox_dst = tt::tt_fabric::linear::addrgen_detail::get_noc_address(inbox, me);
            const uint64_t remote_inbox_sem = get_noc_addr(sync_x, sync_y, inbox_sem_addr);
            for (uint32_t off = 0; off < bytes;) {
                const uint32_t n =
                    bytes - off < max_payload ? bytes - off : max_payload;  // (the send CB holds max_payload)
                noc_async_read(table.get_noc_addr(o) + off, send_l1, n);
                noc_async_read_barrier();
                if (off == 0) {
                    auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(send_l1);
                    h[::sparse_push::plan_table::kWordActiveBlocks] = active_blocks;
                    h[::sparse_push::plan_table::kWordHasSplit] = has_split;
                    h[::sparse_push::plan_table::kWordSplitBlock] = split_block;
                }
                if (off + n == bytes) {
                    fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                        &conn,
                        hdr,
                        dest_chip_ids[pc],
                        dest_mesh_ids[pc],
                        send_l1,
                        n,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            inbox_dst + off, remote_inbox_sem, 1, true});
                } else {
                    fabric_mesh::fabric_unicast_noc_unicast_write(
                        &conn,
                        hdr,
                        dest_chip_ids[pc],
                        dest_mesh_ids[pc],
                        send_l1,
                        n,
                        tt::tt_fabric::NocUnicastCommandHeader{inbox_dst + off});
                }
                noc_async_writes_flushed();
                off += n;
            }
        }
    }
#endif

    const uint32_t t_xchg = stats_clock();
    // ---- 3. send what the reader (kvd3_txr.cpp, NCRISC) hands over, in ring order; forward my workers' credits
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_go)) = 1;  // (the chip barrier is passed)
    auto* filled_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_filled));
    auto* drained_w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_drained));
    auto* rdone = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rdone));
    auto* desc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(desc_l1(send_l1, sbytes));
    const uint64_t nb_credit_base = get_noc_addr(nbtx_x, nbtx_y, ctrl);
    const uint32_t cred = cred_l1(send_l1, sbytes);
    auto* credw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cred);
    for (uint32_t i = 0; i < kMaxK * kMaxClasses; ++i) {
        credw[i] = 0;
    }
    // headers rotate over 4 (at most 2 sends are in flight: the writer flushes before a third), whatever the ring
    // depth: the packet header pool is small
    constexpr uint32_t kHdrs = 4;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* rhdr[kHdrs];
    for (uint32_t i = 0; i < kHdrs; ++i) {
        rhdr[i] = PacketHeaderPool::allocate_header();
    }
    namespace fx = tt::tt_fabric::common::experimental;
#ifndef KVD_ABL_NO_STATEFUL
    // the route and every field are set once a header; a packet only patches its destination, arrival word and size
    for (uint32_t i = 0; i < kHdrs; ++i) {
        fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc_set_state<
            fx::UnicastFusedAtomicIncUpdateMask::All>(
            rhdr[i],
            nb_chip,
            nb_mesh,
#if defined(KVD_ABL_NOFLUSH_ARRIVE) || defined(KVD_SEQ)
            // (NOFLUSH_ARRIVE: timing only, unsafe; SEQ: the worker waits for the slot's stamp instead)
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, 1, false},
#else
            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{0, 0, 1, true},
#endif
            sbytes);
    }
    constexpr auto kPatch = static_cast<fx::UnicastFusedAtomicIncUpdateMask>(
        static_cast<uint32_t>(fx::UnicastFusedAtomicIncUpdateMask::WriteDstAddr) |
        static_cast<uint32_t>(fx::UnicastFusedAtomicIncUpdateMask::SemaphoreAddr) |
        static_cast<uint32_t>(fx::UnicastFusedAtomicIncUpdateMask::PayloadSize));
#endif
#ifdef KVD_ABL_BATCH_ARRIVE
    // Batched arrivals (as sparse_dispatch_2d vouches for rows): data goes as plain writes; the S-th packet to a
    // (worker, class), and every END, is a fused write whose flushed increment carries the whole pending count (the
    // receiving router pushes out every earlier write before the increment); anything pending when the writer idles
    // (or after kArriveAge plain sends) goes as header-only flushed increments.

#if defined(KVD_ABL_NOFLUSH_ARRIVE) || defined(KVD_ABL_NO_STATEFUL) || defined(KVD_SEQ)
#error "BATCH_ARRIVE needs the flushed, stateful fused header (one increment vouches for the earlier plain writes)"
#endif
    constexpr uint32_t kArriveS = KVD_ABL_BATCH_ARRIVE;
    constexpr uint32_t kArriveAge = 32;
    static_assert(
        kArriveS >= 1 && kArriveS <= pool && kArriveS <= pool_depth(0, pool),
        "a (worker, class) never has more than its pool depth of packets unsignalled");
    volatile tt_l1_ptr PACKET_HEADER_TYPE* whdr[2] = {
        PacketHeaderPool::allocate_header(), PacketHeaderPool::allocate_header()};
    volatile tt_l1_ptr PACKET_HEADER_TYPE* ahdr = PacketHeaderPool::allocate_header();
    for (uint32_t i = 0; i < 2; ++i) {
        fabric_mesh::fabric_unicast_noc_unicast_write_set_state<fx::UnicastWriteUpdateMask::All>(
            whdr[i], nb_chip, nb_mesh, tt::tt_fabric::NocUnicastCommandHeader{0}, sbytes);
    }
    constexpr auto kWPatch = static_cast<fx::UnicastWriteUpdateMask>(
        static_cast<uint32_t>(fx::UnicastWriteUpdateMask::DstAddr) |
        static_cast<uint32_t>(fx::UnicastWriteUpdateMask::PayloadSize));
    constexpr auto kPatchVal = static_cast<fx::UnicastFusedAtomicIncUpdateMask>(
        static_cast<uint32_t>(kPatch) | static_cast<uint32_t>(fx::UnicastFusedAtomicIncUpdateMask::Val));
    uint32_t unsig[K][G];
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t c = 0; c < G; ++c) {
            unsig[k][c] = 0;
        }
    }
    uint32_t n_unsig = 0, since_sig = 0, n_plain = 0;
    auto signal_all = [&]() {  // header-only flushed increments for everything pending
        for (uint32_t k = 0; k < K && n_unsig; ++k) {
            for (uint32_t c = 0; c < G; ++c) {
                if (unsig[k][c]) {
                    noc_async_writes_flushed();  // (ahdr is reused)
                    fabric_mesh::fabric_unicast_noc_unicast_atomic_inc(
                        &conn,
                        ahdr,
                        nb_chip,
                        nb_mesh,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                            get_noc_addr(nbw_x[k], nbw_y[k], ctrl + w_arrived(c) * 4), unsig[k][c], true});
                    n_unsig -= unsig[k][c];
                    unsig[k][c] = 0;
                }
            }
        }
        since_sig = 0;
    };
#endif
    uint32_t fwd[K][G];
    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t c = 0; c < G; ++c) {
            fwd[k][c] = 0;
        }
    }
    uint32_t fwd_total = 0;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* chdr2 = PacketHeaderPool::allocate_header();
    const uint64_t nbh_credit_base = get_noc_addr(nbh_x, nbh_y, ctrl);
    // Credits go back as one snapshot: every (worker, class) count of freed slots (cumulative, so a later snapshot
    // replaces an earlier one) in one packet (and a copy for the neighbour's lane-1 helper), when at least `min_new`
    // were freed since the last one (1 when idle).
#ifdef KVD_ABL_PIGGY
    static_assert(lanes3 == 1, "piggybacked credits: one lane");
    constexpr uint32_t kCw = credit_area_words(B);
    // deltas owed since the last packet: into buffer `buf` (up to kCw of them); returns how many
    uint32_t fwd_k[K];
    for (uint32_t k = 0; k < K; ++k) {
        fwd_k[k] = 0;
    }
    auto piggyback = [&](uint32_t buf) -> uint32_t {
        uint32_t n = 0;
        auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf);
        if (static_cast<int32_t>(cw[kTxOwedTotal] - fwd_total) > 0) {
            auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buf + credit_area_offset(B));
            for (uint32_t k = 0; k < K && n < kCw; ++k) {
                if (static_cast<int32_t>(cw[tx_owed_k(k)] - fwd_k[k]) <= 0) {
                    continue;  // nothing new from worker k (its total is incremented after the class words)
                }
                for (uint32_t c = 0; c < G && n < kCw; ++c) {
                    const uint32_t delta = cw[tx_owed(k, c)] - fwd[k][c];
                    if (delta) {
                        w[n++] = pack_credit(k, c, delta);
                        fwd[k][c] += delta;
                        fwd_k[k] += delta;
                        fwd_total += delta;
                    }
                }
            }
        }
        h[kHdrCredits] = n;
        return n;
    };
    // idle: every delta owed as a header-only atomic increment on the neighbour TX's credit word
    auto forward_credits = [&](uint32_t) {
        bool any = false;
        for (uint32_t k = 0; k < K; ++k) {
            for (uint32_t c = 0; c < G; ++c) {
                const uint32_t delta = cw[tx_owed(k, c)] - fwd[k][c];
                if (delta) {
                    noc_async_writes_flushed();  // chdr is reused
                    fabric_mesh::fabric_unicast_noc_unicast_atomic_inc(
                        &conn,
                        chdr,
                        nb_chip,
                        nb_mesh,
                        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                            nb_credit_base + tx_credit(k, c) * 4, delta, true});
                    fwd[k][c] += delta;
                    fwd_k[k] += delta;
                    fwd_total += delta;
                    any = true;
                }
            }
        }
        return any;
    };
    (void)chdr2;
    (void)nbh_credit_base;
    (void)credw;
#else
    auto forward_credits = [&](uint32_t min_new) {
        // signed: the total word lags the per-class words (incremented after them), so a snapshot can already hold
        // more than the total says; that must not read as a wrapped, huge "fresh" and resend a stale snapshot
        const int32_t fresh = static_cast<int32_t>(cw[kTxOwedTotal] - fwd_total);
        if (fresh < static_cast<int32_t>(min_new) || fresh <= 0) {
            return false;
        }
        noc_async_writes_flushed();  // the previous snapshot has left
        uint32_t total = 0;
        for (uint32_t k = 0; k < K; ++k) {
            for (uint32_t c = 0; c < G; ++c) {
                fwd[k][c] = cw[tx_owed(k, c)];
                credw[tx_credit(k, c)] = fwd[k][c];
                total += fwd[k][c];
            }
        }
        fwd_total = total;
        fabric_mesh::fabric_unicast_noc_unicast_write(
            &conn,
            chdr,
            nb_chip,
            nb_mesh,
            cred,
            kCreditBytes,
            tt::tt_fabric::NocUnicastCommandHeader{nb_credit_base + tx_credit(0, 0) * 4});
        if constexpr (lanes3 == 2) {
            fabric_mesh::fabric_unicast_noc_unicast_write(
                &conn,
                chdr2,
                nb_chip,
                nb_mesh,
                cred,
                kCreditBytes,
                tt::tt_fabric::NocUnicastCommandHeader{nbh_credit_base + tx_credit(0, 0) * 4});
        }
        return true;
    };
#endif
#ifdef KVD_ABL_CREDIT_MIN
    constexpr uint32_t kCreditMin = KVD_ABL_CREDIT_MIN;
#else
    constexpr uint32_t kCreditMin = 32;
#endif
#ifdef KVD_ABL_CREDIT_IDLE
    constexpr uint32_t kCreditIdleMin = KVD_ABL_CREDIT_IDLE;  // credits owed before an idle writer sends them
#else
    constexpr uint32_t kCreditIdleMin = 1;
#endif
#ifdef KVD_ABL_CREDIT_AGE
    constexpr uint32_t kCreditAge = KVD_ABL_CREDIT_AGE;
    static_assert(kCreditAge <= (1u << 20), "an idle writer must forward what it owes eventually");
#else
    constexpr uint32_t kCreditAge = 64;
#endif
    uint32_t idle_rounds = 0;
    // ring A (lane 0, this core's reader: program semaphores) and ring B (lane 1, the helper: kTxFilledB here, the
    // drained count goes back to its kHDrained word); taken alternately when both have buffers
    uint32_t issued[2] = {0, 0}, drained[2] = {0, 0}, turn = 0, n_sent = 0, n_credit_pkts = 0;
    const uint64_t hlp_drained = get_noc_addr(hlp_x, hlp_y, ctrl + kHDrained * 4);
    auto flush = [&]() {  // every send so far has left L1: hand the buffers back
        noc_async_writes_flushed();
        if (issued[0] != drained[0]) {
            drained[0] = issued[0];
            *drained_w = drained[0];
        }
        if (issued[1] != drained[1]) {
            noc_semaphore_inc(hlp_drained, issued[1] - drained[1]);
            drained[1] = issued[1];
        }
    };
    auto filled_of = [&](uint32_t r) -> uint32_t { return r == 0 ? *filled_w : cw[kTxFilledB]; };
    const uint32_t t0 = stats_clock();
    uint32_t cy_fab = 0, cy_wait = 0, t_first = 0, t_last = 0, cy_slot = 0, cy_flush = 0, n_loop = 0;
    uint32_t tl_mark[32];
    // (KVD_STATS) data packets sent per 2^16-cycle (~48.5 us) bin from kernel start, bins 0-23 (~1.16 ms): class 0
    // (planners), then relayed classes 1-7 (END packets and classes 8+ are not counted)
    uint16_t hist0[24], hist1[24];
    for (uint32_t b = 0; b < 24; ++b) {
        hist0[b] = hist1[b] = 0;
    }
    (void)hist0;
    (void)hist1;
    for (uint32_t i = 0; i < 32; ++i) {
        tl_mark[i] = 0;
    }
    (void)tl_mark;
    uint32_t tl_end[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t pk_cls[8] = {0, 0, 0, 0, 0, 0, 0, 0}, rows_cls[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    (void)pk_cls;
    (void)rows_cls;
    (void)tl_end;
    while (true) {
#ifndef KVD_ABL_PIGGY
        if (forward_credits(kCreditMin)) {
            ++n_credit_pkts;
        }
#endif
        const bool ra = filled_of(0) != issued[0];
        const bool rb = lanes3 == 2 && filled_of(1) != issued[1];
        if (ra || rb) {
            const uint32_t r = (ra && (!rb || turn == 0)) ? 0 : 1;
            turn ^= 1;
            const uint32_t i = r * kRing + issued[r] % kRing;
#ifdef KVD_STATS
            {
                const uint32_t tw0 = stats_clock();
                conn.wait_for_empty_write_slot();
                cy_slot += stats_clock() - tw0;
            }
#endif
            const uint32_t tf = stats_clock();
            const uint32_t k = desc[i * 4 + 0], c = desc[i * 4 + 1], dst = desc[i * 4 + 2];
            uint32_t payload = desc[i * 4 + 3];
#ifdef KVD_STATS
            if (payload > rows_off && c < 8) {  // (stats: classes 0-7 only)
                ++pk_cls[c];
                rows_cls[c] += (payload - rows_off) / row_bytes;
                const uint32_t b = (tf - t_k0) >> 16;
                if (b < 24) {
                    ++(c == 0 ? hist0 : hist1)[b];
                }
            }
#endif
#ifdef KVD_ABL_PIGGY
            if (piggyback(send_l1 + i * sbytes) != 0 && payload < rows_off) {
                payload = rows_off;  // (an END packet: carry the credit words too)
            }
#endif
#if defined(KVD_ABL_BATCH_ARRIVE)
            {
                const bool is_end = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(send_l1 + i * sbytes)[kHdrEnd] != 0;
                if (is_end || unsig[k][c] + 1 >= kArriveS) {
                    fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<kPatchVal>(
                        &conn,
                        rhdr[n_sent % kHdrs],
                        send_l1 + i * sbytes,
                        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                            get_noc_addr(nbw_x[k], nbw_y[k], dst),
                            get_noc_addr(nbw_x[k], nbw_y[k], ctrl + w_arrived(c) * 4),
                            static_cast<uint16_t>(unsig[k][c] + 1),
                            true},
                        static_cast<uint16_t>(payload));
                    n_unsig -= unsig[k][c];
                    unsig[k][c] = 0;
                } else {
                    fabric_mesh::fabric_unicast_noc_unicast_write_with_state<kWPatch>(
                        &conn,
                        whdr[n_plain & 1],
                        send_l1 + i * sbytes,
                        tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(nbw_x[k], nbw_y[k], dst)},
                        static_cast<uint16_t>(payload));
                    ++n_plain;
                    ++unsig[k][c];
                    ++n_unsig;
                    if (++since_sig >= kArriveAge) {
                        signal_all();
                    }
                }
            }
#elif !defined(KVD_ABL_NO_STATEFUL)
            fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc_with_state<kPatch>(
                &conn,
                rhdr[n_sent % kHdrs],
                send_l1 + i * sbytes,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    get_noc_addr(nbw_x[k], nbw_y[k], dst),
                    get_noc_addr(nbw_x[k], nbw_y[k], ctrl + w_arrived(c) * 4),
                    1,
                    true},
                static_cast<uint16_t>(payload));
#else
            fabric_mesh::fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &conn,
                rhdr[n_sent % kHdrs],
                nb_chip,
                nb_mesh,
                send_l1 + i * sbytes,
                payload,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    get_noc_addr(nbw_x[k], nbw_y[k], dst),
                    get_noc_addr(nbw_x[k], nbw_y[k], ctrl + w_arrived(c) * 4),
                    1,
                    true});
#endif
            ++issued[r];
            ++n_sent;
            idle_rounds = 0;
#ifdef KVD_STATS
            if (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(send_l1 + i * sbytes)[kHdrEnd] != 0 && c < 8 &&
                tl_end[c] == 0) {
                tl_end[c] = stats_clock() / 135;  // the first END(c) sent (0.1 us, absolute: vs the reader's marks)
            }
            if ((n_sent & 63) == 0 && n_sent / 64 <= 32) {
                tl_mark[n_sent / 64 - 1] = (stats_clock() - t_k0) / 1350;
            }
#endif
            cy_fab += stats_clock() - tf;
            t_last = stats_clock();
            if (t_first == 0) {
                t_first = t_last;
            }
            // buffers go back once their sends have left L1 (at most 2 sends unflushed: 4 headers rotate)
            if ((issued[0] - drained[0]) + (issued[1] - drained[1]) >= 2) {
                const uint32_t tq = stats_clock();
                flush();
                cy_flush += stats_clock() - tq;
            }
            ++n_loop;
            continue;
        }
        const uint32_t tw = stats_clock();
        flush();
#ifdef KVD_ABL_BATCH_ARRIVE
        if (n_unsig) {
            signal_all();  // nothing to send: everything pending is signalled now
        }
#endif
        // idle: credits go back once kCreditIdleMin are owed, or anything owed once idle for kCreditAge rounds (an
        // idle writer must never sit on credits its neighbour may be waiting for)
        if (forward_credits(++idle_rounds > kCreditAge ? 1 : kCreditIdleMin)) {
            ++n_credit_pkts;
            idle_rounds = 0;
        }
        if (*rdone && (lanes3 == 1 || cw[kTxRdoneB] != 0) && cw[kTxWorkersDone] == K * wsplit &&
            filled_of(0) == issued[0] && (lanes3 == 1 || filled_of(1) == issued[1])) {
            forward_credits(1);
            bool all = true;
            for (uint32_t k = 0; k < K && all; ++k) {
                for (uint32_t c = 0; c < G; ++c) {
                    if (cw[tx_owed(k, c)] != fwd[k][c]) {
                        all = false;
                        break;
                    }
                }
            }
            if (all) {
                break;
            }
        }
        cy_wait += stats_clock() - tw;
    }
    noc_async_atomic_barrier();
    const uint32_t t_end = stats_clock();  // (the loop's end, before the stats prints)
#ifdef KVD_STATS
    DPRINT(
        "KVD3TP {} {} {} {} {} {} {} {} | {} {} {} {} {} {} {} {}\n",
        pk_cls[0],
        pk_cls[1],
        pk_cls[2],
        pk_cls[3],
        pk_cls[4],
        pk_cls[5],
        pk_cls[6],
        pk_cls[7],
        rows_cls[0],
        rows_cls[1],
        rows_cls[2],
        rows_cls[3],
        rows_cls[4],
        rows_cls[5],
        rows_cls[6],
        rows_cls[7]);
    DPRINT("KVD3TH d={} l={} c0", d, l);
    for (uint32_t b = 0; b < 24; ++b) {
        DPRINT(" {}", hist0[b]);
    }
    DPRINT(" | rel");
    for (uint32_t b = 0; b < 24; ++b) {
        DPRINT(" {}", hist1[b]);
    }
    DPRINT("\n");
    if (d == 0 && l == 0) {
        DPRINT(
            "KVD3TE {} {} {} {} {} {} {} {} {}\n",
            tl_end[0],
            tl_end[1],
            tl_end[2],
            tl_end[3],
            tl_end[4],
            tl_end[5],
            tl_end[6],
            tl_end[7],
            (stats_clock() - t_k0) / 135);
        DPRINT(
            "KVD3TL {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
            tl_mark[0],
            tl_mark[1],
            tl_mark[2],
            tl_mark[3],
            tl_mark[4],
            tl_mark[5],
            tl_mark[6],
            tl_mark[7],
            tl_mark[8],
            tl_mark[9],
            tl_mark[10],
            tl_mark[11],
            tl_mark[12],
            tl_mark[13],
            tl_mark[14],
            tl_mark[15],
            tl_mark[16],
            tl_mark[17]);
    }
    DPRINT(
        "KVD3T d={} l={} total={} fab={} wait={} slot={} flush={} first={} last={} pkts={} credit_pkts={} xchg={} "
        "k0={} end={} (k0, end: raw wall clock, same chip only)\n",
        d,
        l,
        stats_clock() - t0,
        cy_fab,
        cy_wait,
        cy_slot,
        cy_flush,
        (t_first - t_k0) / 1350,
        (t_last - t_k0) / 1350,
        n_sent,
        n_credit_pkts,
        (t_xchg - t_k0) / 1350,
        t_k0,
        t_end);
#endif
    (void)t0;
    (void)t_k0;
    (void)t_xchg;
    (void)t_end;
    (void)cy_fab;
    (void)cy_wait;
    (void)cy_slot;
    (void)cy_flush;
    (void)n_loop;
    (void)t_first;
    (void)t_last;
    (void)n_sent;
    (void)n_credit_pkts;
    (void)hdr;
    (void)prod_x;
    (void)prod_y;
    (void)n_workers;
    noc_async_writes_flushed();
    conn.close();
}
