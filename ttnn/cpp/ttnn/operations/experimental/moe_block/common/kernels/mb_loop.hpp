// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Token loops inside one program (MoEBlockConfig.token_loops k): a kernel's body runs k times, each time as if it were
// a new launch of the loop's own program - the host merges the k per-loop programs into one (loop_program.hpp) and the
// loop's argument differences become delta tables this header applies in place.
//
// A kernel defines its body with MB_LOOP_BODY { ... } in place of void kernel_main() { ... } and includes
// mb_loop_main.hpp after it. Without MB_LOOPS (one loop) the body is kernel_main and nothing here runs.
//
// Between loop l and l + 1 (what a program boundary did):
//  * every RISC finishes loop l (its NoC transactions flushed / the Tensix pipeline idle) and marks it;
//  * the chip's barrier, phase A: each core's lead RISC (NCRISC, else BRISC) waits for its core's RISCs, then counts on
//    the coordinator core, which releases every core once all came: no core still runs loop l, nothing of loop l is
//    still in flight towards another core of the chip;
//  * the lead resets its core's program semaphores (their initial value 0) and CB counters (stream registers); phase B
//    of the barrier: every core reset;
//  * every RISC re-initialises its CB interfaces, applies loop l + 1's argument deltas to its own args and resets
//    its packet header pool.
// Chips are not barriered here: the ops' cross-chip fences are per launch, they run per loop as they ran per call.
//
// MB_CHIP (one program for the mesh, common/chip_program.hpp): kernel_main first writes this chip's arg values
// (mb_chip.hpp), then runs the body (its loops).
//
// The loop table: common args from MB_LT (the kernel's own common args come first, unchanged):
//   [0] W - the loop words (persistent, L1_SMALL, the same address on every core: done words a RISC 16 B apart at
//       W + 16 rid, the go word at W + 80, the coordinator's count at W + 96)
//   [1], [2] 0
//   [3] the coordinator: logical x | y << 8 | virtual x << 16 | y << 24
//   [4] the chip's cores (the barrier's count)
//   [5] nc, then nc entries (index | last << 31 (only the last loop), delta) of my common args
//   then np per-core sections: (logical x | y << 8, n, n entries (index | last << 31, delta) of my args)
//   then nl cores this kernel leads: (logical x | y << 8, the core's RISCs (bit rid: 0 BRISC, 1 NCRISC, 2..4 TRISC
//   0..2), the program semaphores to reset (bit id)); or, when it leads every core of its own alike, 1 << 31 and that
//   one role (RISCs, semaphores)
//   then (coordinator kernel) the cores to release: n, n x (virtual x | y << 16)
#pragma once

#include <cstdint>

#ifdef MB_LOOPS
#define MB_LOOP_BODY void mb_loop_body()
void mb_loop_body();
namespace mb_loop {
inline uint32_t g_loop = 0;
inline __attribute__((always_inline)) uint32_t index() { return g_loop; }
inline __attribute__((always_inline)) bool is_last() { return g_loop + 1 == MB_LOOPS; }
}  // namespace mb_loop
#elif defined(MB_CHIP) || defined(MB_CHIP_WAIT)  // (one program for the mesh: the chip's args first)
#define MB_LOOP_BODY void mb_loop_body()
void mb_loop_body();
namespace mb_loop {
constexpr uint32_t index() { return 0; }
constexpr bool is_last() { return true; }
}  // namespace mb_loop
#else
#define MB_LOOP_BODY void kernel_main()
namespace mb_loop {
constexpr uint32_t index() { return 0; }
constexpr bool is_last() { return true; }
}  // namespace mb_loop
#endif
