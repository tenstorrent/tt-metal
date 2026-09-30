// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused SwiGLU (GLU) epilogue helpers for bmm_large_block_zm_fused_bias_activation.cpp (define FUSE_GLU).
// The subblock in DEST holds tile-pair interleaved [gate | up] tiles: DST[2j] = gate, DST[2j + 1] = up.
// Each helper writes DST[2j] = silu(DST[2j]) * DST[2j + 1], with the same SFPU silu and the same SFPU multiply
// (fp32 product, software RNE to bf16, 0 * x = 0) as minimal_matmul's swiglu_block.
//
// Two variants:
//  - MATH thread (default): glu_init_math() once per pass, glu_pairs_math<N>() between the copy and the commit.
//  - PACK thread (define GLU_SFPU_ON_PACK): glu_init_pack() once per kernel, glu_pairs_from_pack<N>() in place of
//    tile_regs_wait(), as apply_activation_from_pack() does for the fused unary activation.
#pragma once

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"

#ifdef TRISC_PACK
// The TRISC_PACK build of compute_kernel_api.h includes only the unary SFPU kernels. The pack-thread
// multiply needs the binary SFPU kernel and its call macros too.
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"
#endif

// ---------------------------------------------------------------------------------------------------------------
// MATH thread
// ---------------------------------------------------------------------------------------------------------------
// minimal_matmul re-inits silu and mul for every pair. The mul init only sets ADDR_MOD_7 = {0, 0, 0}; the silu
// init sets the same address mode and its programmable constants. Doing mul first and silu last leaves the
// state each op sees identical to the per-pair order, so one init per pass is numerics-neutral.
FORCE_INLINE void glu_init_math() {
    mul_binary_tile_init();
    silu_tile_init();
}

template <uint32_t num_tiles>
FORCE_INLINE void glu_pairs_math() {
    static_assert(num_tiles % 2 == 0, "fused SwiGLU needs an even number of tiles per subblock");
    for (uint32_t j = 0; j < num_tiles / 2; j++) {
        silu_tile(2 * j);
        mul_binary_tile(2 * j, 2 * j + 1, 2 * j);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// PACK thread
// ---------------------------------------------------------------------------------------------------------------
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
FORCE_INLINE void mul_binary_tile_pack(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    PACK((SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        calculate_sfpu_binary_mul,
        (APPROX, ckernel::BinaryOp::MUL, 8 /* ITERATIONS */, is_fp32_dest_acc_en),
        idst0,
        idst1,
        odst,
        VectorMode::RC)));
}

#ifdef GLU_FUSED_SFPU
// One pass y = silu(g) * u over the tile pair (2j, 2j + 1), result over tile 2j (see calculate_swiglu).
FORCE_INLINE void swiglu_tile_pack(uint32_t idst) {
    PACK(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_swiglu, (DST_ACCUM_MODE, 8 /* ITERATIONS */), idst, VectorMode::RC));
}
#endif

FORCE_INLINE void mul_binary_tile_init_pack() {
    PACK((SFPU_BINARY_INIT_FN(unused, sfpu::sfpu_binary_init, (APPROX, ckernel::BinaryOp::MUL))));
}

FORCE_INLINE void glu_init_pack() {
    mul_binary_tile_init_pack();
    silu_tile_init_pack();
#ifdef GLU_FUSED_SFPU
    // calculate_swiglu clamps |gate| with SFPSWAP against Prgm2 (LREG14) = 87.5f. silu_init programs Prgm2 only when
    // SILU_BF16_IMPL is 2 or 3 (the default is 0), so set it here; same value silu_init used to write.
    PACK(sfpi::vConstFloatPrgm2 = 87.5f);
#endif
}

template <uint32_t num_tiles>
FORCE_INLINE void glu_pairs_from_pack() {
    static_assert(num_tiles % 2 == 0, "fused SwiGLU needs an even number of tiles per subblock");
    // Replaces tile_regs_wait(): wait for MATH to commit the DEST half (see apply_activation_from_pack).
    PACK(TTI_SEMWAIT(
        p_stall::STALL_TDMA | p_stall::STALL_CFG, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_ZERO));
    // Point the SFPU at the DEST half that the packer owns.
    PACK(TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, ckernel::packer::get_packer_dest_offset()));
    for (uint32_t j = 0; j < num_tiles / 2; j++) {
#ifdef GLU_FUSED_SFPU
        swiglu_tile_pack(2 * j);
#else
        silu_tile_pack(2 * j);
        mul_binary_tile_pack(2 * j, 2 * j + 1, 2 * j);
#endif
    }
    // Wait for the SFPU before packing.
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
}
