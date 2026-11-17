// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/debug/kernel_profiler.hpp"

// The persistent op (FE_PERSISTENT): every kernel loops, one iteration per token row the requesters'
// input sockets bring.
//
// The control core (fe_ctrl.cpp) takes a page off any requester's input D2D socket -- a header whose first
// word is the layer id, then the hidden row -- and runs three stages, each with its own go and its own
// ready count, so up to three rows are in flight (the router stage up to one row per requester ahead):
//   1. the router row (router scores, this chip's top-k), counted on the control core's sem_route_ready, and the
//      expert-id merge of every chip's candidates, on its own count (FE_SEM_MERGE_GO, a ring of the routed rows'
//      headers): the merge core holds the ids until the grid's go (FE_SEM_EXPERT_GO);
//   2. the worker grid (gate/up, down, the within-chip reduce), counted on the control core's
//      sem_ready;
//   3. the reduce senders (the cross-chip reduce), a row behind the grid, counted on sem_reduce_ready;
//      their staging is double buffered, so the grid's go for iteration i waits for row i - 2.
// Each stage's go is the header words and the row (into the input replica) multicast to the stage's
// cores and, once those have landed, the go words: one per RISC of a worker core, set to iteration + 1
// (kFeStopBit set when the layer id is the stop id). Each worker core's header sits at FE_HDR_ADDR, a
// workload tensor -- the TRISCs cannot address program semaphores, and a RISC that clears only its own
// go word at stop leaves no stale go behind for the next launch. The reduce senders take the grid's go
// on program semaphore FE_SEM_GO.
//
// The chips need not take the requesters' rows in the same order: a chip's only input from the others is their
// router candidates, kept per requester (each requester has one request in flight), and its router stage never
// waits on them -- only the merge does, which the grid follows in this chip's own order. Every chip routes every
// row a requester sends, so every merge completes.
//
// Because a stage's go of iteration i + 1 needs that stage done with iteration i, every CB is empty at
// the top of an iteration, so each RISC rewinds its CB pointers there (fe_rewind_cbs) and every
// semaphore that counts up within an iteration counts up by the same amount per iteration (the
// kernels wait for cumulative targets).
//
// A core reports ready (FE_SEM_READY, which the factory points at its stage's count on the control
// core, one NoC atomic increment) from its reader: once its compute has popped the input -- the
// compute's last CB operation of an iteration -- and its writer has set FE_SEM_WRITER_DONE, so all five
// RISCs are past the iteration.

constexpr uint32_t kFeStopBit = 0x80000000u;

// Orders staged L1 stores before an MMIO register write (such as starting a NoC transfer) or a cross-RISC handoff.
FORCE_INLINE void fe_store_fence() { asm volatile("fence" ::: "memory"); }

// The most experts one group runs per iteration (the program factory's kMaxGroupExperts bound): 1 unless the
// persistent op takes several-row requests.
#ifndef FE_GROUP_EXPERTS
#define FE_GROUP_EXPERTS 1u
#endif
// The routed gate_up K-split's partners (FE_ROUTED_SPLIT): producer c's is the group's local core FE_PARTNER_BASE + c.
#ifndef FE_PARTNER_BASE
#define FE_PARTNER_BASE 8u
#endif

// Header words the control core writes at FE_HDR_ADDR before each go.
constexpr uint32_t kFeHdrLayer = 0;
constexpr uint32_t kFeHdrSharedGateUp = 1;
constexpr uint32_t kFeHdrSharedDown = 2;
constexpr uint32_t kFeHdrRouterGate = 3;
constexpr uint32_t kFeHdrRouterBias = 4;
// The layer's routed picks per row when fewer than the op's top_k (0: top_k): the DSpark drafter's layers on the
// target's server take 3 of their (padded) experts per row.
constexpr uint32_t kFeHdrTopK = 5;
// The row's requester, in the router row's header (the reducer's candidate slot) and the merge ring's.
constexpr uint32_t kFeHdrRequester = 6;
// The request's rows when fewer than the op's (0: the op's): a one-row request to an op built for several-row
// blocks, its row in row 0 of the block's (rows x 32) tiles (the other rows' contents are never used). Set in the
// request page's header by the requester and copied by the control core into the router row's header (the route
// merge drops the other rows' selections) and the source ring (the senders send row 0's partial only).
constexpr uint32_t kFeHdrRows = 9;
// FE_NATIVE_ONE_ROW (an op built for several-row blocks, DEEPSEEK_V41_MOE_NATIVE_ONE_ROW != 0): a one-row request is
// sent and run in plain one-row form instead -- the page's row is the plain row (k_tiles 1x32 tiles), and every
// tile-shaped CB of the op has a 1x32 alias, buffer index + kFeOneRowCbOffset, over the same memory (the program
// factory adds them), which every kernel uses for that iteration: the compute LLKs take the tile shape from the CB
// index, and the dataflow kernels take the one-row tile bytes (FE_ONE_ROW_*_TILE_BYTES). A CB index has its own
// pointers and page counts, so a CB's producer and consumer pick the same alias in an iteration (both read the
// header), and fe_rewind_cbs rewinds both. Without it (the padded mode) a one-row request rides row 0 of a block.
constexpr uint32_t kFeOneRowCbOffset = 32;
// Bit k set when requester k is served on demand rather than in turn (the same in every row; the control core reads
// row 0's): see fe_ctrl.cpp.
constexpr uint32_t kFeHdrOnDemand = 7;
// Several-row requests (FE_GROUP_EXPERTS > 1): the route merge's hit count -- the size of the union of the rows'
// selections -- as (iter + 1) << 8 | hits, written into every grid core's header after the control core's (whose
// layer-table row holds 0 here). Word 8 keeps it 16-byte aligned for the merge's multicast.
constexpr uint32_t kFeHdrHits = 8;
// Several-row requests: bit j set when the group's j-th expert's routed gate_up K-pieces are read in reverse (odd
// groups, the DRAM bank stagger). The compute takes an expert's pieces once all are in and chains them in K order,
// finding piece p at the CB position these bits say, so an expert's activation does not depend on which group the
// union deals it to. Written by each routed core's reader into its own header once the ids are in, before it pushes
// the first piece; the next iteration's header multicast comes only after this core is done.
constexpr uint32_t kFeHdrPieceFlags = 12;
constexpr uint32_t kFeHdrWords = 16;
// Then the go words, one per RISC of a worker core: BRISC, NCRISC, TRISC0-2.
constexpr uint32_t kFeGoWords = 5;
// Then, on the control core, each iteration's requester, for the root's output kernel. The control core
// runs at most a few iterations ahead of it.
constexpr uint32_t kFeSourceRing = 16;
// The merge ring: the headers of the rows the router stage takes, in order, kFeSourceRing deep (the router stage
// runs at most one row per requester ahead of the grid, whose go waits on the merge before it).
constexpr uint32_t kFeMergeRingBytes = kFeSourceRing * kFeHdrWords * 4u;
// A merge ring entry's layer word for the stop.
constexpr uint32_t kFeStopLayerWord = 0xFFFFFFFFu;

// Whether this iteration's request is a one-row one run natively (FE_NATIVE_ONE_ROW): its header's kFeHdrRows is 1.
// The header of an iteration stays put until every core of the stage is done with it, so any RISC may read it any
// time in the iteration.
#if defined(FE_NATIVE_ONE_ROW)
#ifndef FE_BLOCK_ROWS
#error "FE_NATIVE_ONE_ROW needs FE_BLOCK_ROWS (the op's rows)"
#endif
FORCE_INLINE bool fe_one_row() {
    invalidate_l1_cache();
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FE_HDR_ADDR)[kFeHdrRows] == 1u;
}
#else
FORCE_INLINE constexpr bool fe_one_row() { return false; }
#endif
// A tile-shaped CB's index for the iteration: its one-row alias in a one-row iteration.
FORCE_INLINE constexpr uint32_t fe_cb(uint32_t cb, bool one_row) { return one_row ? cb + kFeOneRowCbOffset : cb; }
// The one-row tile bytes (a 1x32 tile of the format), defined with FE_NATIVE_ONE_ROW.
#ifndef FE_ONE_ROW_ACT_TILE_BYTES
#define FE_ONE_ROW_ACT_TILE_BYTES 0u
#endif
#ifndef FE_ONE_ROW_OUT_TILE_BYTES
#define FE_ONE_ROW_OUT_TILE_BYTES 0u
#endif
#ifndef FE_ONE_ROW_SCALAR_TILE_BYTES
#define FE_ONE_ROW_SCALAR_TILE_BYTES 0u
#endif

#if defined(FE_PERSISTENT)

#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t kFeMyGoWord = 0;
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint32_t kFeMyGoWord = 1;
#else
constexpr uint32_t kFeMyGoWord = 2u + COMPILE_FOR_TRISC;
#endif

FORCE_INLINE uint32_t fe_header(uint32_t word) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FE_HDR_ADDR)[word];
}

// Waits for the go of iteration `iter`; false when it is the stop, after clearing this RISC's go word.
FORCE_INLINE bool fe_wait_go(uint32_t iter) {
    DeviceZoneScopedN("FE_GO_WAIT");
    volatile tt_l1_ptr uint32_t* go =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FE_HDR_ADDR) + kFeHdrWords + kFeMyGoWord;
    uint32_t v;
    do {
        invalidate_l1_cache();
        v = *go;
    } while ((v & ~kFeStopBit) < iter + 1u);
    if ((v & kFeStopBit) != 0u) {
        *go = 0;
        return false;
    }
    return true;
}

// The hit count of iteration `iter` (see kFeHdrHits), once the route merge has written it.
FORCE_INLINE uint32_t fe_wait_hits(uint32_t iter) {
    volatile tt_l1_ptr uint32_t* word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FE_HDR_ADDR) + kFeHdrHits;
    uint32_t v;
    do {
        invalidate_l1_cache();
        v = *word;
    } while ((v >> 8) != iter + 1u);
    return v & 0xFFu;
}

// Expert group `group`'s share of `hits` hits dealt round robin over `groups` groups (hit h to group h % groups).
FORCE_INLINE uint32_t fe_group_hits(uint32_t hits, uint32_t group, uint32_t groups) {
    return group < hits ? (hits - group + groups - 1u) / groups : 0u;
}

#ifndef FE_REWIND_CBS_HI
#define FE_REWIND_CBS_HI 0u
#endif

// Back to the start of every CB in FE_REWIND_CBS (indices 0-31) and FE_REWIND_CBS_HI (32-63: the one-row aliases), as
// the firmware left them at launch. The math TRISC has no CB interface (its firmware defines none) and never moves a CB
// pointer.
FORCE_INLINE void fe_rewind_cbs() {
#if !defined(UCK_CHLKC_MATH)
    for (uint32_t half = 0; half < 2u; ++half) {
        const uint32_t mask =
            half == 0u ? static_cast<uint32_t>(FE_REWIND_CBS) : static_cast<uint32_t>(FE_REWIND_CBS_HI);
        for (uint32_t cb = 0; cb < 32u; ++cb) {
            if (((mask >> cb) & 1u) != 0u) {
                LocalCBInterface& cb_if = get_local_cb_interface(half * 32u + cb);
                cb_if.fifo_rd_ptr = cb_if.fifo_limit - cb_if.fifo_size;
                cb_if.fifo_wr_ptr = cb_if.fifo_rd_ptr;
                cb_if.fifo_wr_tile_ptr = 0;
            }
        }
    }
#endif
}

// Start of iteration `iter` for a core's ready reporter; false on stop. `cb_input_id` is the CB the
// local compute pops last (k_tiles pages), `has_writer` whether the core runs a writer.
#if !defined(COMPILE_FOR_TRISC)
// Whether the previous iteration was a one-row one (FE_NATIVE_ONE_ROW): its input went through the one-row alias.
inline bool fe_prev_one_row = false;
FORCE_INLINE bool fe_begin_iteration(uint32_t iter, uint32_t cb_input_id, uint32_t k_tiles, bool has_writer) {
    if (iter > 0u) {
        // The whole CB, through whichever index the previous iteration used: free again once the compute popped it
        // (the one-row alias holds FE_BLOCK_ROWS times the pages).
#if defined(FE_NATIVE_ONE_ROW)
        cb_reserve_back(fe_cb(cb_input_id, fe_prev_one_row), fe_prev_one_row ? k_tiles * FE_BLOCK_ROWS : k_tiles);
#else
        cb_reserve_back(cb_input_id, k_tiles);
#endif
        if (has_writer) {
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_SEM_WRITER_DONE)), iter);
        }
        noc_semaphore_inc(get_noc_addr(FE_CTRL_X, FE_CTRL_Y, get_semaphore(FE_SEM_READY)), 1);
        noc_async_atomic_barrier();
    }
    if (!fe_wait_go(iter)) {
        return false;
    }
    fe_rewind_cbs();
    fe_prev_one_row = fe_one_row();
    return true;
}

// The writer's end of an iteration: lets the reader report the core ready.
FORCE_INLINE void fe_writer_done(uint32_t iter) {
    fe_store_fence();
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_SEM_WRITER_DONE)) = iter + 1u;
}
#endif

#endif
