// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The AttnRes mixture's MAC, shared by every AttnRes compute kernel that mixes: after this
// init, `mul_tiles_bcast_cols(a, b, ia, ib, d)` adds `a * column 0 of b` into dest tile `d`
// rather than overwriting it.
//
// The compute API has no init for a column-broadcast multiply that accumulates into dest, so
// this is `bcast_init` with the math side re-initialized for `acc_to_dest`.
//
// Include after the compute API's `bcast.h`.

#pragma once

#include <cstdint>

namespace ckernel {

ALWI void attn_res_mul_bcast_cols_acc_init(uint32_t icb0, uint32_t icb1) {
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(icb0, icb1);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(
        icb0, icb1, 1 /*acc_to_dest*/)));
}

}  // namespace ckernel
