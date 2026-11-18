// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifdef GROUPED_N
// Grouped N split (swiglu_last_chunk): a core's virtual N range is [A part (GRP_A_PER_CORE) | B part (GRP_B_PER_CORE)];
// map a virtual column to the physical weight / output column (A at c * A_pc, B at GRP_A_TILES + c * B_pc).
inline uint32_t grp_phys_n(uint32_t v) {
    constexpr uint32_t per_core = GRP_A_PER_CORE + GRP_B_PER_CORE;
    const uint32_t c = v / per_core;
    const uint32_t l = v - c * per_core;
    return l < GRP_A_PER_CORE ? c * GRP_A_PER_CORE + l : GRP_A_TILES + c * GRP_B_PER_CORE + (l - GRP_A_PER_CORE);
}

// Physical weight column -> output column: group A unchanged, group B (SwiGLU pairs) halves.
inline uint32_t grp_out_n(uint32_t p) { return p < GRP_A_TILES ? p : GRP_A_TILES + (p - GRP_A_TILES) / 2; }
#endif
