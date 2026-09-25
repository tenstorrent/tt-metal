// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"

#ifdef TRISC_MATH
namespace ckernel::sfpu {

// The fixed-register body deliberately does not mix live SFPI compiler-managed
// vectors with raw instructions. LREG0..7 are scratch; no constant is rewritten.
// ADDR_MOD_7 is the zero-increment SFPU address modifier initialized by startup.
inline void bfp4_max_r2_r3() {
    // In VEC_MIN_MAX mode VC gets max and VD gets min: retain max in LREG2.
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, 1);
    TTI_SFPNOP;
}

template <int ROTATIONS>
inline void bfp4_rotate_r2_to_r3() {
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG3, 3);
    TTI_SFPNOP;
    for (int i = 1; i < ROTATIONS; ++i) {
        TTI_SFPSHFT2(0, p_sfpu::LREG3, p_sfpu::LREG3, 3);
        TTI_SFPNOP;
    }
}

template <bool FP32_DST, int BASE>
inline void bfp4_round_four_face_rows() {
    // Each load is four rows x eight columns of one parity. Their pair covers
    // four independent native BFP groups of 16 adjacent columns, not 32 columns.
    constexpr int dst_format = FP32_DST ? 3 : 2;  // explicit FP32 / BF16
    TTI_SFPLOAD(p_sfpu::LREG0, dst_format, ADDR_MOD_7, BASE);
    TTI_SFPLOAD(p_sfpu::LREG1, dst_format, ADDR_MOD_7, BASE + 2);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 1);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG3, 1);
    bfp4_max_r2_r3();

    // A cyclic max butterfly broadcasts each group's max to all eight lanes.
    // SHFLROR1 never crosses an eight-lane subgroup (one face row).
    bfp4_rotate_r2_to_r3<1>();
    bfp4_max_r2_r3();
    bfp4_rotate_r2_to_r3<2>();
    bfp4_max_r2_r3();
    bfp4_rotate_r2_to_r3<4>();
    bfp4_max_r2_r3();

    // Let E=floor(log2(max(abs(x)))). The BFP4 grid is delta=2^(E-2).
    // magic=2^(E+21) has FP32 ulp=delta. Two SEPARATE rounded additions
    // therefore implement RNE to this grid. LREG4 holds magic.
    TTI_SFPEXEXP(0, p_sfpu::LREG2, p_sfpu::LREG4, 1);  // biased exponent
    TTI_SFPIADD(21, p_sfpu::LREG4, p_sfpu::LREG4, 5);  // immediate, no CC
    TTI_SFPSETEXP(0, p_sfpu::LCONST_1, p_sfpu::LREG4, 0);

    // cap=7*delta=1.75*2^E. Use the maximum's exponent, avoiding a multiply.
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_FLOATB, 0x3fe0);
    TTI_SFPSETEXP(0, p_sfpu::LREG3, p_sfpu::LREG2, 2);
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG5, 1);
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG6, 1);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LREG5, 0);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG6, 0);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG5, p_sfpu::LREG4, p_sfpu::LREG5, 2);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG6, p_sfpu::LREG4, p_sfpu::LREG6, 2);

    // Copy cap before destructive min/max. MOV hides the MAD->SWAP hazard;
    // an explicit NOP follows each SWAP as required by the Blackhole ISA.
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG7, 0);
    TTI_SFPSWAP(0, p_sfpu::LREG7, p_sfpu::LREG5, 1);
    TTI_SFPNOP;
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG6, 1);
    TTI_SFPNOP;
    TTI_SFPSETSGN(0, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPSETSGN(0, p_sfpu::LREG6, p_sfpu::LREG1, 0);
    TTI_SFPSTORE(p_sfpu::LREG0, dst_format, ADDR_MOD_7, BASE);
    TTI_SFPSTORE(p_sfpu::LREG1, dst_format, ADDR_MOD_7, BASE + 2);
}

template <bool FP32_DST>
inline void round_native_bfp4_face() {
    bfp4_round_four_face_rows<FP32_DST, 0>();
    bfp4_round_four_face_rows<FP32_DST, 4>();
    bfp4_round_four_face_rows<FP32_DST, 8>();
    bfp4_round_four_face_rows<FP32_DST, 12>();
}
}  // namespace ckernel::sfpu
#endif

void kernel_main() {
    constexpr uint32_t batch = get_compile_time_arg_val(0);
    const uint32_t count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 16);
    copy_init(0);
    negative_tile_init();
    // FP32-DST copies use FPU ELWADD, so its SrcB zero must really be zero.
    MATH((ckernel::math::_configure_src_zero_flag_(false)));
    for (uint32_t i = 0; i < count; i += batch) {
        cb_wait_front(0, batch);
        cb_reserve_back(16, batch);
        tile_regs_acquire();
        for (uint32_t j = 0; j < batch; ++j) {
            copy_tile(0, j, j);
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE, DST_ACCUM_MODE, round_native_bfp4_face, (DST_ACCUM_MODE), j, VectorMode::RC));
        }
        tile_regs_commit();
        cb_pop_front(0, batch);
        tile_regs_wait();
        for (uint32_t j = 0; j < batch; ++j) {
            // Pre-rounded values are exactly representable in BFP4, so the
            // native E8M6 -> BFP8 -> BFP4 packing conversions are identities.
            pack_tile(j, 16);
        }
        tile_regs_release();
        cb_push_back(16, batch);
    }
}
