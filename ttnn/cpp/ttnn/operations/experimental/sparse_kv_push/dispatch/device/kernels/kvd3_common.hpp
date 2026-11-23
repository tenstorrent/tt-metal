// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_kv_dispatch v3 (KV_DISPATCH_2D_PLAN.md 10.4-10.6, model tools/kvd3_model.py): full L1 packets relayed through
// worker cores, one fabric (TX) core per (direction, link), no DRAM queues, no mux. Plain constexpr / inline only: the
// host includes it too.
//
// Roles (each core one): TX(d, l) owns the fabric connection to the neighbour in direction d on link l; W(f, l, k) is
// worker k of the packets that arrive from the neighbour in direction f on link l (they move in f ^ 1); planners build
// the class-0 packets of this chip's own rows.
// Classes: a packet's class is its rows' hop generation c (0 at the owner). Every class has its own landing pool and
// credits end to end, staging is bounded per (out port, class), and a worker serves its classes independently (deepest
// first, a blocked class is skipped): the dependency graph is then acyclic (a class-c resource waits only on class c +
// 1).
#pragma once

#include <cstdint>

#include "kvd_common.hpp"

#ifdef KVD_ACTIVE_FROM_BLOCK
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/common/kernels/sparse_push_constants.hpp"
#endif

namespace kvd3 {

#ifdef KVD_ACTIVE_FROM_BLOCK
// active_from (SparseKvDispatchParams): true when this chunk starts before it, read from this chip's plan table header
// (page 0, bank 0) into 64 B of scratch L1. Every chip sees the same chunk start, so all skip together, and every
// kernel checks before any fabric or semaphore traffic.
FORCE_INLINE bool skip_chunk(uint32_t table_addr, uint32_t scratch_l1) {
    noc_async_read(get_noc_addr_from_bank_id<true>(0, table_addr), scratch_l1, 64);
    noc_async_read_barrier();
    const auto* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    return h[::sparse_push::plan_table::kWordHasSplit] != 0 &&
           h[::sparse_push::plan_table::kWordSplitBlock] < KVD_ACTIVE_FROM_BLOCK;
}
#endif

// ---- A packet slot (landing pools, staging, send buffers): header, B entries (kvd entry words), rows at align64.
// header word 0: entries, word 1: class, word 2: END flag (an END(class) packet has no entries).
constexpr uint32_t kHdrEntries = 0, kHdrClass = 1, kHdrEnd = 2;
// Piggybacked credits (KVD_ABL_PIGGY): header word 3 counts packed deltas in the slot's spare bytes between the entries
// and the rows; each word: worker k (3 bits) | class c << 3 (4 bits) | delta << 7. The landing worker adds them to its
// own chip's TX facing back (whose packets they credit). Deltas: order-independent across packets and workers.
constexpr uint32_t kHdrCredits = 3;
// KVD_SEQ: data lands without the router's per-packet write flush. The TX reader stamps the packet's sequence
// number (per landing worker and class) in header word 3 and in the packet's last 16 B: the padding of its last row
// (rows are aligned pages), or bytes 48-63 of a header-only END. The worker trusts the header once its stamp is there,
// then reads the slot once the trailer stamp is there too (the write landed: a packet's bytes land in address order).
// Before the credit goes back (and at start) it zeroes the header stamp, END's trailer word and every row's stamp word,
// so a trailer not yet written reads 0. Stamps also reach the output's row padding (never its logical bytes).
constexpr uint32_t kHdrSeq = 3;
constexpr uint32_t seq_trailer(uint32_t B, uint32_t row_bytes, uint32_t n, bool end) {
    return end ? 48 : kvd::slot_rows_offset(B) + n * row_bytes - 16;
}
constexpr uint32_t kSeqMagic = 0x5E000000;
#if defined(KVD_SEQ) && defined(KVD_ABL_PIGGY)
#error "KVD_SEQ uses header word 3 (PIGGY's credit count)"
#endif
constexpr uint32_t credit_area_offset(uint32_t B) { return kvd::kSlotHeaderBytes + B * kvd::kEntryBytes; }
constexpr uint32_t credit_area_words(uint32_t B) { return (kvd::slot_rows_offset(B) - credit_area_offset(B)) / 4; }
constexpr uint32_t pack_credit(uint32_t k, uint32_t c, uint32_t delta) { return k | (c << 3) | (delta << 7); }

// ---- Control block: one CB (index 0, on every core of the program: the same L1 address everywhere), zeroed by its
// core at the start of a call before the chip barrier; nothing writes it before that barrier.
constexpr uint32_t kCtrlWords = 2304;

#ifdef KERNEL_BUILD
// zero a control block with NoC reads of the hardware zero region (a store loop over the 2304 words cost ~5 us of
// the chip barrier)
inline void zero_ctrl(uint32_t ctrl) {
    static_assert((kCtrlWords * 4) % MEM_ZEROS_SIZE == 0, "the control block is a whole number of zero chunks");
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t off = 0; off < kCtrlWords * 4; off += MEM_ZEROS_SIZE) {
        noc_async_read(zeros, ctrl + off, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
}
#endif
constexpr uint32_t kMaxClasses = 16;
constexpr uint32_t kMaxK = 8;
constexpr uint32_t kMaxProducers = 112;  // planners + 4 x K workers
// TX(d, l)
constexpr uint32_t tx_credit(uint32_t k, uint32_t c) { return k * kMaxClasses + c; }      // returned by nbr's W
constexpr uint32_t tx_owed(uint32_t k, uint32_t c) { return 128 + k * kMaxClasses + c; }  // freed by my W(d, l, k)
constexpr uint32_t kTxWorkersDone = 288;                                                  // my W(d, l, *) finished
constexpr uint32_t kTxOwedTotal = 289;  // sum of tx_owed (incremented after them)
constexpr uint32_t tx_pub(uint32_t p, uint32_t c) { return 320 + p * kMaxClasses + c; }  // producer p, class c
static_assert(320 + kMaxProducers * kMaxClasses <= 2120);
// Lanes of a link: the TX's producers and the neighbour's K workers split in two halves (producer index parity, worker
// k < K / 2): lane 0 is read by the TX core's NCRISC, lane 1 by a helper core that pushes its packets into ring B.
// Per lane s: slots published a class, producers done with a class, all published (incremented after the class word).
constexpr uint32_t kMaxLanes3 = 2;
constexpr uint32_t tx_cls_pub(uint32_t s, uint32_t c) { return 2120 + s * kMaxClasses + c; }
constexpr uint32_t tx_done(uint32_t s, uint32_t c) { return 2152 + s * kMaxClasses + c; }
constexpr uint32_t tx_pub_total(uint32_t s) { return 2184 + s; }
constexpr uint32_t kTxFilledB = 2188;                          // ring B buffers handed over by the helper
constexpr uint32_t kTxRdoneB = 2189;                           // the helper has finished
constexpr uint32_t kHDrained = 2190;                           // (on the helper) ring B buffers the writer has sent
constexpr uint32_t tx_owed_k(uint32_t k) { return 2192 + k; }  // sum of tx_owed(k, *) (incremented after them)
static_assert(kHDrained < kCtrlWords);
// W(f, l, k)
constexpr uint32_t w_arrived(uint32_t c) { return c; }                                     // landed packets
constexpr uint32_t w_freed(uint32_t td, uint32_t c) { return 64 + td * kMaxClasses + c; }  // staging pulled by TX(td)
constexpr uint32_t w_flush_req(uint32_t td) { return 200 + td; }  // TX(td)'s reader is starving (KVD_ABL_HUNGRY)
// planner
constexpr uint32_t p_freed(uint32_t port) { return 64 + port; }  // staging pulled, port d*L+l

// TX send CB: kRing packet buffers (the reader fills, the writer sends), the credit snapshot, then kRing descriptors
// (worker, class, landing slot address, payload bytes).
#ifdef KVD_ABL_RING
constexpr uint32_t kRing = KVD_ABL_RING;
#else
constexpr uint32_t kRing = 8;
#endif
constexpr uint32_t kCreditBytes = kMaxK * kMaxClasses * 4;
// ring A buffers 0..kRing-1, ring B (LANES=2: the host defines KVD3_RINGS 2) kRing..2kRing-1; descriptors likewise
#ifndef KVD3_RINGS
#define KVD3_RINGS 2
#endif
constexpr uint32_t kRings = KVD3_RINGS;
constexpr uint32_t cred_l1(uint32_t send_l1, uint32_t sbytes) { return send_l1 + kRings * kRing * sbytes; }
constexpr uint32_t desc_l1(uint32_t send_l1, uint32_t sbytes) { return cred_l1(send_l1, sbytes) + kCreditBytes; }
constexpr uint32_t kMaxRing = 15;  // KVD_ABL_RING may set the ring up to this (streaming reads: transaction ids 1..15)
static_assert(kRing >= 1 && kRing <= kMaxRing);
// the send CB for `rings` rings of `ring` buffers (the host passes the ring depth and lane count it compiles with)
constexpr uint32_t send_cb_bytes(uint32_t sbytes, uint32_t ring, uint32_t rings) {
    return rings * ring * sbytes + kCreditBytes + rings * ring * 16 + 64;  // (+64: the kernels align the base up)
}

// Producers of TX(d, l): planners 0..P-1, then W(f, l, k) at P + f * K + k.
constexpr uint32_t worker_producer(uint32_t P, uint32_t K, uint32_t f, uint32_t k) { return P + f * K + k; }

// Staging layout (workers and planners share one CB, the same address on both): slot s of (port q, class c) at
// ((q * G + c) * stage + s) * sbytes. Workers: q = out direction (0..3); planners: q = d * L + l, class 0.
constexpr uint32_t stage_offset(uint32_t q, uint32_t c, uint32_t s, uint32_t G, uint32_t stage, uint32_t sbytes) {
    return ((q * G + c) * stage + s) * sbytes;
}
// Landing pool layout (workers): class c has pool_depth(c) slots, class 0 first. KVD_ABL=POOL0=n: class 0 (the
// planners' traffic, most of a busy port's backlog) gets n slots, the other classes `pool`.
#ifdef KVD_ABL_POOL0
constexpr uint32_t kPool0 = KVD_ABL_POOL0;
#endif
constexpr uint32_t pool_depth(uint32_t c, uint32_t pool) {
#ifdef KVD_ABL_POOL0
    return c == 0 ? kPool0 : pool;
#else
    (void)c;
    return pool;
#endif
}
constexpr uint32_t pool_offset(uint32_t c, uint32_t s, uint32_t pool, uint32_t sbytes) {
#ifdef KVD_ABL_POOL0
    return (c == 0 ? s : kPool0 + (c - 1) * pool + s) * sbytes;
#else
    return (c * pool + s) * sbytes;
#endif
}

}  // namespace kvd3
