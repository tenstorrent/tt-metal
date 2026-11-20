// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A matmul core's N blocks, shared by in0, in1 and compute.
//
// - n_stride == 0: block b covers [N_start + b * N_block, ...) clipped at N_end (minimal_matmul's split).
// - n_stride > 0 (N groups): the weight columns are groups of n_stride tiles and block b covers this core's part of
//   group q = (n_perm >> 4b) & 15, [q * n_stride + N_start, ...) clipped at the group's end. N_start is then the core's
//   offset inside every group. The blocks visit the groups in n_perm's order, so the output lands group by group (e.g.
//   in the order a reduce-scatter consumes it).
//
// signal_n_blocks: the output writer increments L1 counter word 16 * b on each target core once block b's output
// tiles have landed (RT at sig: counters, num_targets, x0, y0, ...).

#pragma once

#include <stdint.h>
#include "ttnn/operations/experimental/lego/kernels/lego.hpp"
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#include "api/dataflow/dataflow_api.h"
#endif

namespace lego::mm {

template <typename A>
inline void n_block_range(uint32_t N_start, uint32_t N_end, uint32_t b, uint32_t& n0, uint32_t& n1) {
    if constexpr (A::n_stride > 0) {
        const uint32_t q = (A::n_perm >> (4 * b)) & 0xF;
        n0 = q * A::n_stride + N_start;
        n1 = n0 + A::N_block_tiles;
        if (n1 > (q + 1) * A::n_stride) {
            n1 = (q + 1) * A::n_stride;
        }
    } else {
        n0 = N_start + b * A::N_block_tiles;
        n1 = n0 + A::N_block_tiles;
        if (n1 > N_end) {
            n1 = N_end;
        }
    }
}

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
template <typename A>
inline void signal_n_block(uint32_t sig, uint32_t b) {
    noc_async_write_barrier();
    const uint32_t counters = rt<A>(sig);
    const uint32_t n = rt<A>(sig + 1);
    for (uint32_t t = 0; t < n; t++) {
        noc_semaphore_inc(get_noc_addr(rt<A>(sig + 2 + 2 * t), rt<A>(sig + 3 + 2 * t), counters + 16 * b), 1);
    }
}
#endif

}  // namespace lego::mm
