// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stream core, compute RISCs: one RISC of the routing-index build each. The op has no tile math, so
// without this the three TRISCs idle for the whole launch; the pass is nothing but L1 loads and
// stores, which they issue as well as the reader does.
//
// Built from the reader's compile-time arguments, unchanged, so the layout of the scratch and
// every constant the pass reads are the reader's own. The JIT compiles this file three times, once per
// TRISC, with exactly one of TRISC_UNPACK / TRISC_MATH / TRISC_PACK defined (jit_build's TRISC
// prolog); each build is one RISC.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
// The reader's argument struct names its tensor accessors, which a compute build does not pull in.
#include "api/tensor/tensor_accessor_args.h"
#include "../dispatch_fabric2d_routing_index.hpp"

#if defined(TRISC_UNPACK)
constexpr uint32_t kRisc = dspf2d::kRiscUnpack;
#elif defined(TRISC_MATH)
constexpr uint32_t kRisc = dspf2d::kRiscMath;
#elif defined(TRISC_PACK)
constexpr uint32_t kRisc = dspf2d::kRiscPack;
#else
#error "routing index risc kernel built for an unknown TRISC"
#endif
// The RISC set is the reader plus these three builds and nothing else; a RISC no RISC runs would leave
// the reader waiting forever.
static_assert(
    dspf2d::INDEX_RISCS == 4u, "one routing index risc per RISC that runs run_risc: the reader and three TRISCs");
static_assert(kRisc != dspf2d::kRiscReader && kRisc < dspf2d::INDEX_RISCS);

void kernel_main() {
    const dspf2d::routing_index::Control c = dspf2d::routing_index::layout_scratch();
    dspf2d::routing_index::run_risc(c, kRisc);
}
