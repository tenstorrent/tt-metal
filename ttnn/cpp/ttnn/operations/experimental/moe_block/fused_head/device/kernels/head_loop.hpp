// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Token loops of the fused head, run by the kernels themselves (MoEBlockConfig.token_loops k).
//
// Why: a chunk run as k token loops processes k slices of the rows one after another. The head runs them all in one
// launch: every kernel runs its body k times (HL_BODY, a loop in head_loop_main.hpp) instead of the host enqueuing
// the head k times.
//
// What changes from one loop to the next - each kernel computes it from the loop index (hl::index()):
//   * x and the shared expert's output: loop l's rows are HL_DX / HL_DY bytes further in every bank;
//   * the dispatch buffer's rows of this chip's tokens: HL_DROW rows further (the prep units);
//   * the call's first row in the SP row (a padded forward's span): HL_DSUB rows further;
//   * the router's owner page and route words: HL_DPAGE pages / HL_DROUTE bytes further;
//   * the routing planner plans in the last loop only (every loop's tokens); earlier loops run it empty.
// Everything else (CB layouts, slot rings, counters) carries on from loop to loop as from one row chunk to the next:
// the kernels apply their chunk-boundary rules at a loop boundary too, and count ring slots across the loops.
//
// Between two loops every RISC of the program finishes its loop and the chip's cores pass a barrier
// (head_loop_main.hpp); the barrier's lead on each core resets the core's program semaphores (their initial 0) in the
// middle of it. That is plain synchronization with public APIs: semaphores and NoC atomics, nothing of the launch
// state is touched.
#pragma once

#include <cstdint>
// (every kernel's args: mb_args::arg / mb_args::common, read from this core's block of the DRAM tables -
// head_program.cpp head_args_to_dram; head_loop_main.hpp loads it first)
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_args.hpp"

#ifndef HL_LOOPS
#define HL_LOOPS 1
#endif
#ifndef HL_DX
#define HL_DX 0
#endif
#ifndef HL_DY
#define HL_DY 0
#endif
#ifndef HL_DROW
#define HL_DROW 0
#endif
#ifndef HL_DSUB
#define HL_DSUB 0
#endif
#ifndef HL_DPAGE
#define HL_DPAGE 0
#endif
#ifndef HL_DROUTE
#define HL_DROUTE 0
#endif

#define HL_BODY void hl_body()
void hl_body();

namespace hl {
constexpr uint32_t LOOPS = HL_LOOPS;
inline uint32_t g_loop = 0;
inline __attribute__((always_inline)) uint32_t index() { return g_loop; }
inline __attribute__((always_inline)) bool is_last() { return g_loop + 1 == LOOPS; }
}  // namespace hl
