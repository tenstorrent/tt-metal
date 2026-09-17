// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stream core, compute RISCs: one lane of the routing-index build each. The op has no tile math, so
// without this the three TRISCs idle for the whole launch; the pass is nothing but L1 loads and
// stores, which they issue as well as the reader does.
//
// Built from the reader's compile-time arguments, unchanged, so the carve of the control region and
// every constant the pass reads are the reader's own. The JIT compiles this file three times, once per
// TRISC, with exactly one of TRISC_UNPACK / TRISC_MATH / TRISC_PACK defined (jit_build's TRISC
// prolog); each build is one lane.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
// The reader's argument struct names its tensor accessors, which a compute build does not pull in.
#include "api/tensor/tensor_accessor_args.h"
#include "../dispatch_fabric2d_prologue.hpp"

#if defined(TRISC_UNPACK)
constexpr uint32_t kLane = dspf2d::kLaneUnpack;
#elif defined(TRISC_MATH)
constexpr uint32_t kLane = dspf2d::kLaneMath;
#elif defined(TRISC_PACK)
constexpr uint32_t kLane = dspf2d::kLanePack;
#else
#error "prologue lane kernel built for an unknown TRISC"
#endif
// The lane set is the reader plus these three builds and nothing else; a lane no RISC runs would leave
// the reader waiting forever.
static_assert(dspf2d::PROLOGUE_LANES == 4u, "one lane per RISC that runs run_lane: the reader and three TRISCs");
static_assert(kLane != dspf2d::kLaneReader && kLane < dspf2d::PROLOGUE_LANES);

void kernel_main() {
    const dspf2d::prologue::Control c = dspf2d::prologue::carve_control();
    dspf2d::prologue::run_lane(c, kLane);
}
