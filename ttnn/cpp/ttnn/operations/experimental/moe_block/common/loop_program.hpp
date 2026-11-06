// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Token loops inside one program (MoEBlockConfig.token_loops k, kernels/mb_loop.hpp): an op whose call ran as k
// programs - the same kernels, CBs and semaphores, only their args differ by the loop - runs them as one program whose
// kernels loop. The host builds loop 0's, loop 1's (an earlier loop's) and the last loop's program of a chip; the
// merge keeps loop 0's and gives every kernel its loop table: the loop's argument deltas (linear in the loop, and the
// last loop's own), its core's RISCs and semaphores, and the chip barrier between loops.
#pragma once

#include <cstdint>
#include <functional>

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::moe_block {

// the loop words a core needs (mb_loop.hpp): 5 done words, the go word, the coordinator's count, 16 B apart
inline constexpr uint32_t kLoopWordsBytes = 112;

struct LoopProgramSpec {
    uint32_t loops = 1;  // k >= 2
    uint32_t words = 0;  // the loop words' base: persistent, zero, the same address on every core of the program
    std::function<CoreCoord(const CoreCoord&)> virt;  // a logical worker core's NoC (virtual) coordinates
};

// p0 / p1 / plast: loop 0's, loop 1's and the last loop's program of one chip (k = 2: p1 is the last loop's). They must
// match but for their runtime args (kernels, compile-time args, defines, CBs, semaphores). Returns p0 with every
// kernel's loop table appended to its common args (MB_LOOPS, MB_LT). Kernels on compute RISCs must have no deltas.
tt::tt_metal::ProgramDescriptor merge_loop_programs(
    tt::tt_metal::ProgramDescriptor p0,
    const tt::tt_metal::ProgramDescriptor& p1,
    const tt::tt_metal::ProgramDescriptor& plast,
    const LoopProgramSpec& spec);

}  // namespace ttnn::operations::experimental::moe_block
