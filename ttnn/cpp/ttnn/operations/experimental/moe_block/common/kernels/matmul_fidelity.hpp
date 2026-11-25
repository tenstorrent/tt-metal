// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Matmuls that name their fidelity at the call site (compute kernels): matmul_block_init_at<fidelity> /
// matmul_block_at<fidelity>, so a kernel that mixes fidelities (the MoE head's router logits at HiFi3, the tail's sums
// of squares at HiFi4) shows every transition where it happens. configured_fidelity: the kernel's compute-config
// fidelity (MATH_FIDELITY).
#pragma once

#include <cstdint>
#include "api/compute/matmul.h"

// The kernel's configured fidelity: the compute config's math_fidelity (MATH_FIDELITY). The build generates
// MATH_FIDELITY for the MATH and PACK threads only; the UNPACK thread's half of a matmul (the state and the unpacker's
// init / call) is the same at every fidelity, so there it is a placeholder.
#if defined(TRISC_MATH) || defined(TRISC_PACK)
constexpr MathFidelity configured_fidelity = MATH_FIDELITY;
#else
constexpr MathFidelity configured_fidelity = MathFidelity::LoFi;  // (UNPACK: unused - see above)
#endif

// matmul_block_init / matmul_block at a fidelity named by the call site. The kernel's configured fidelity: the public
// matmul_block_init / matmul_block (on Blackhole with their firmware-controlled dynamic throttle); another fidelity:
// the same LLK calls with that fidelity fixed (no public matmul takes a fidelity), the throttle level MM_THROTTLE.
// The arguments as the public calls': transpose B's tiles, the output block's column / row tiles, the inner tiles.
template <MathFidelity fidelity>
ALWI void matmul_block_init_at(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t transpose,
    uint32_t column_tiles,
    uint32_t row_tiles,
    uint32_t inner_tiles,
    uint32_t call_line = __builtin_LINE()) {
    if constexpr (fidelity == configured_fidelity) {
        ckernel::matmul_block_init(in0_cb, in1_cb, transpose, column_tiles, row_tiles, inner_tiles, call_line);
    } else {
        state_configure(in1_cb, in0_cb, call_line);
        UNPACK((llk_unpack_AB_matmul_init(in0_cb, in1_cb, transpose, column_tiles, row_tiles, inner_tiles)));
        MATH((llk_math_matmul_init<fidelity, MM_THROTTLE>(in0_cb, in1_cb, transpose, column_tiles, row_tiles)));
    }
}
template <MathFidelity fidelity>
ALWI void matmul_block_at(
    uint32_t in0_cb,
    uint32_t in1_cb,
    uint32_t in0_tile,
    uint32_t in1_tile,
    uint32_t dest_tile,
    uint32_t transpose,
    uint32_t column_tiles,
    uint32_t row_tiles,
    uint32_t inner_tiles,
    uint32_t call_line = __builtin_LINE()) {
    if constexpr (fidelity == configured_fidelity) {
        ckernel::matmul_block(
            in0_cb, in1_cb, in0_tile, in1_tile, dest_tile, transpose, column_tiles, row_tiles, inner_tiles, call_line);
    } else {
        state_configure(in1_cb, in0_cb, call_line);
        UNPACK((llk_unpack_AB_matmul(in0_cb, in1_cb, in0_tile, in1_tile, column_tiles, row_tiles, inner_tiles)));
        MATH((llk_math_matmul<fidelity, MM_THROTTLE>(dest_tile, column_tiles, row_tiles)));
    }
}
