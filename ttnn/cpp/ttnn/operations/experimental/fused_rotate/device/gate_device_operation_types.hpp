// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Fused SO(3) gate activation (column-split elementwise, NO reduction). Per edge row of nsph*H cols:
//   out[:, :H]  = silu(a[:, :H])                       (mode 0, forward)   OR
//                 a[:, :H] * silu'(b[:, :H])           (mode 1, backward = silu_bw(a, b))
//   out[:, H:]  = a[:, H:] * gate[:, :]                (both modes; gate is [E,(nsph-1)*H])
// Collapses the slice+silu+slice+multiply+concat chain into one kernel. b is only read in mode 1.
struct GateParams {
    uint32_t Wt;    // total tiles per row = nsph*H/32
    uint32_t Gt;    // gate tiles per row  = (nsph-1)*H/32
    uint32_t Ht;    // scalar(l=0) tiles   = H/32
    uint32_t mode;  // 0 = forward (silu), 1 = backward (silu_bw)
};

struct GateInputs {
    Tensor a;     // [E, nsph*H] TILE bf16  (fwd: x;  bw: g_out)
    Tensor gate;  // [E, (nsph-1)*H] TILE bf16  (expanded sigmoid gate)
    Tensor b;     // [E, nsph*H] TILE bf16  (bw: x for silu'; fwd: pass a, unused)
};

}  // namespace ttnn::experimental::prim
