// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the TailEngine engine-core dataflow kernels: compile-time args 0-10, the CB and semaphore ids.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t KH = get_compile_time_arg_val(0), PNMAX = get_compile_time_arg_val(1),
                   RT = get_compile_time_arg_val(2);
constexpr uint32_t NCK = get_compile_time_arg_val(3), GX = get_compile_time_arg_val(4),
                   WTB = get_compile_time_arg_val(5);
constexpr uint32_t NSLOT = get_compile_time_arg_val(6), KT = get_compile_time_arg_val(7),
                   XT = get_compile_time_arg_val(8);
constexpr uint32_t PAIRS = get_compile_time_arg_val(9), NSTMAX = get_compile_time_arg_val(10);
// Token loops (LOOPS, the kernels' own loop): a step's global index g = l nst + s (nst: my pair's steps a loop, the
// same on every core of a row and of its partner) - the slots other cores address absolutely (the ring CB_A and the E
// slots CB_PART by g % NSLOT, the exchange CB_XRECV by g % NSTMAX) follow the CBs' pointers, which advance a slot a
// step across loops; the landing slots CB_L0 are a loop's (step s at slot s): the compute pushes a whole CB a loop
// (its own steps' tiles, then filler) and the writer pops it whole, so the pointer is back at the base every loop.
// (SX: the partner's exchanges (cumulative over the loops: wait g + 1); SFREE: the row's cores' steps done
// (cumulative: GX (g - NSLOT + 1)); SW: W' K ranges in, a bit each (once a call). The per-step words - landed (my
// share), SA (the row's shares), SP (the row's E tiles) - CB_WD, at words + 16 (kind NSTMAX + s), the SA source word
// at kind 1 step 0, each step's SA words a column after them: zeroed once a call (loop 0) by every engine core's
// reader before the zero gate; then counted per loop - loop l's thresholds are l + 1 (landed, SA) and (l + 1) GX (SP).
// The gate a loop (SZ: the lead's count, SZGO: the writers' go; PZR / PZW: the port readers' / writers' go): loop 0
// once every engine core zeroed its words; loop l > 0 once every engine writer and every port reader is done with loop
// l - 1 - nothing reads loop l - 1's landing slots any more: the writers pop them, the port writers send their ready
// (the previous chip lands loop l's shares only after it) - not persistent: no L1_SMALL a step)
constexpr uint32_t SX = 0, SFREE = 1, SW = 2, SG = 3, SGO = 4;  // (SG / SGO: the W' loads' gate - BRISC)
constexpr uint32_t SZ = 5, SZGO = 6, PZR = 0, PZW = 1;
constexpr uint32_t CB_WD = 16;  // (fused_tail_program_factory: the words' CB, after the landing CB on engine and ports)
constexpr uint32_t WD_BYTES = 16 * (3 * NSTMAX + 1 + NSTMAX * GX);
constexpr uint32_t CB_W = 0, CB_A = 1, CB_PART = 2, CB_PMY = 3, CB_ONES = 4, CB_XSEND = 6, CB_XRECV = 8, CB_SE = 9;
constexpr uint32_t CB_OUT = 10, CB_L0 = 11, CB_RM = 12;
FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t i) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(i));
}
// step s of pair p: the call's row tile t, its chunk position and chunk-local row tile
FORCE_INLINE uint32_t step_t(uint32_t s, uint32_t p) { return s * PAIRS + p; }
