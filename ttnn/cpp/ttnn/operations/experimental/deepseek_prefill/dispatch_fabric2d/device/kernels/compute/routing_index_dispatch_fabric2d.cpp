// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compute kernel of a stream core: each TRISC builds its slice of the routing index. Built from the reader's
// compile-time arguments so the scratch layout matches, and compiled once per TRISC with one of
// TRISC_UNPACK / TRISC_MATH / TRISC_PACK defined.

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
#error "routing index kernel built for an unknown TRISC"
#endif
// run_risc waits for all INDEX_RISCS, so each one must be run: by the reader and these three TRISCs.
static_assert(dspf2d::INDEX_RISCS == 4u, "INDEX_RISCS must be the reader plus three TRISCs");
static_assert(kRisc != dspf2d::kRiscReader && kRisc < dspf2d::INDEX_RISCS);

void kernel_main() {
    const dspf2d::routing_index::Control c = dspf2d::routing_index::layout_scratch();
    dspf2d::routing_index::run_risc(c, kRisc);
}
