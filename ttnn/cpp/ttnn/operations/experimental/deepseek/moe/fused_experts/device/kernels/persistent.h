// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/debug/kernel_profiler.hpp"

// The persistent op (FE_PERSISTENT): every kernel loops, one iteration per token row the host sends.
//
// The control core (fe_h2d_ctrl.cpp) takes a page off this chip's H2D socket -- a header whose first
// word is the layer id, then the hidden row -- and waits until every participant has reported ready,
// i.e. finished the previous iteration. It then multicasts the header words and the row (into the
// input replica) to the worker grid and, once those have landed, the go words: one per RISC of a
// worker core, set to iteration + 1 (kFeStopBit set when the layer id is the stop id). Each worker
// core's header sits at FE_HDR_ADDR, a workload tensor -- the TRISCs cannot address program
// semaphores, and a RISC that clears only its own go word at stop leaves no stale go behind for the
// next launch. The reduce senders, outside the grid, take the go on program semaphore FE_SEM_GO.
//
// Because the go of iteration i + 1 needs every participant done with iteration i, every CB is empty
// at the top of an iteration, so each RISC rewinds its CB pointers there (fe_rewind_cbs) and every
// semaphore that counts up within an iteration counts up by the same amount per iteration (the
// kernels wait for cumulative targets).
//
// A core reports ready (FE_SEM_READY on the control core, one NoC atomic increment) from its reader:
// once its compute has popped the input -- the compute's last CB operation of an iteration -- and its
// writer has set FE_SEM_WRITER_DONE, so all five RISCs are past the iteration.

constexpr uint32_t kFeStopBit = 0x80000000u;

// Header words the control core writes at FE_HDR_ADDR before each go.
constexpr uint32_t kFeHdrLayer = 0;
constexpr uint32_t kFeHdrSharedGateUp = 1;
constexpr uint32_t kFeHdrSharedDown = 2;
constexpr uint32_t kFeHdrRouterGate = 3;
constexpr uint32_t kFeHdrRouterBias = 4;
constexpr uint32_t kFeHdrWords = 16;
// Then the go words, one per RISC of a worker core: BRISC, NCRISC, TRISC0-2.
constexpr uint32_t kFeGoWords = 5;

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

// Back to the start of every CB in FE_REWIND_CBS, as the firmware left them at launch. The math TRISC
// has no CB interface (its firmware defines none) and never moves a CB pointer.
FORCE_INLINE void fe_rewind_cbs() {
#if !defined(UCK_CHLKC_MATH)
    for (uint32_t cb = 0; cb < 32u; ++cb) {
        if (((static_cast<uint32_t>(FE_REWIND_CBS) >> cb) & 1u) != 0u) {
            LocalCBInterface& cb_if = get_local_cb_interface(cb);
            cb_if.fifo_rd_ptr = cb_if.fifo_limit - cb_if.fifo_size;
            cb_if.fifo_wr_ptr = cb_if.fifo_rd_ptr;
            cb_if.fifo_wr_tile_ptr = 0;
        }
    }
#endif
}

// Start of iteration `iter` for a core's ready reporter; false on stop. `cb_input_id` is the CB the
// local compute pops last (k_tiles pages), `has_writer` whether the core runs a writer.
#if !defined(COMPILE_FOR_TRISC)
FORCE_INLINE bool fe_begin_iteration(uint32_t iter, uint32_t cb_input_id, uint32_t k_tiles, bool has_writer) {
    if (iter > 0u) {
        cb_reserve_back(cb_input_id, k_tiles);
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
    return true;
}

// The writer's end of an iteration: lets the reader report the core ready.
FORCE_INLINE void fe_writer_done(uint32_t iter) {
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FE_SEM_WRITER_DONE)) = iter + 1u;
}
#endif

#endif
