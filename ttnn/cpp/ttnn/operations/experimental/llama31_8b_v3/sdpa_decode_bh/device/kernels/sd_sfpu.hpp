// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_BLACKHOLE)
#ifdef TRISC_MATH
namespace ckernel {

template <typename F, typename... A>
inline __attribute__((always_inline)) void sdv_run(F&& fn, uint32_t idst, VectorMode vm, A&&... a) {
    sdo_sfpu_start(idst);
    if (vm == VectorMode::RC || vm == VectorMode::R || vm == VectorMode::C) {
        const int n = vm == VectorMode::RC ? 4 : 2;
        const int step = vm == VectorMode::C ? 2 : 1;
#pragma GCC unroll 0
        for (int f = 0; f < n; f++) {
            fn(a...);
            for (int s = 0; s < step; s++) {
                sdo_sfpu_next_face();
            }
        }
        if (vm == VectorMode::R) {
            sdo_sfpu_next_face();
            sdo_sfpu_next_face();
        }
    } else {
        fn(a...);
    }
    sdo_sfpu_done();
}

ALWI void sdv_seek_rows4(uint32_t row0) {
    if (row0 >= 16) {
        sdo_sfpu_next_face();
        sdo_sfpu_next_face();
    }
    switch (row0 & 15) {
        case 4: ckernel::math::inc_dst_addr<4>(); break;
        case 8: ckernel::math::inc_dst_addr<8>(); break;
        case 12: ckernel::math::inc_dst_addr<12>(); break;
        default: break;
    }
}

template <uint32_t DEST, uint32_t BITS>
inline void sdv_cfg32() {
    TTI_SFPLOADI(0, 10, BITS & 0xFFFFu);
    TTI_SFPLOADI(0, 8, BITS >> 16);
    TTI_SFPCONFIG(0, DEST, 0);
}

template <uint32_t LREG, uint32_t BITS>
inline void sdv_imm32() {
    TTI_SFPLOADI(LREG, 10, BITS & 0xFFFFu);
    TTI_SFPLOADI(LREG, 8, BITS >> 16);
}

inline void sdv_recip_init() { static_assert(APPROX); }

template <bool APPROXIMATION_MODE, uint32_t scale, bool CLAMP_NEGATIVE, bool is_fp32_dest_acc_en>
inline void sdv_exp_init() {
    static_assert(!APPROXIMATION_MODE && is_fp32_dest_acc_en);
    TTI_SFPCONFIG(0, 0xF, 1);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    math::reset_counters(p_setrwc::SET_ABD_F);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    sdv_cfg32<p_sfpu::LREG12, 0x3FB8AA3Bu>();
    sdv_cfg32<p_sfpu::LREG13, 0xBF317200u>();
    sdv_cfg32<p_sfpu::LREG14, 0xB5BFBE8Eu>();
}

template <bool SCALE_EN>
ALWI void sdv_exp_head(const std::uint32_t s) {
    TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0);
    if constexpr (SCALE_EN) {
        TT_SFPMULI(s, p_sfpu::LREG1, 0);
    }
    TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG12, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
    TTI_SFPLOADI(p_sfpu::LREG0, 1, 0x15A5);
    TTI_SFP_STOCH_RND(0, 0, p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG3, 7);
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG2, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG13, p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG14, p_sfpu::LREG1, p_sfpu::LREG1, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG4, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG5, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG6, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG7, p_sfpu::LREG0, 0);
    TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 3);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG0, 0);
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
    TTI_SFPMULI(0x7F80, p_sfpu::LREG0, 0);
}

ALWI void sdv_exp_tail() {
    TTI_SFPEXEXP(0, p_sfpu::LREG1, p_sfpu::LREG2, 1);
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, 4);
    TTI_SFPIADD(0xF01, p_sfpu::LREG2, p_sfpu::LREG3, 1);
    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
    TTI_SFPSETEXP(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
    TTI_SFPIADD(0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, 1);
    TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_6, 0);
}

template <bool SCALE_EN, int ROWS>
inline void sdv_exp(const int faces, const int step, const int skip, const std::uint32_t s) {
    constexpr std::uint32_t HEAD_LEN = 15 + (SCALE_EN ? 1 : 0);
    static_assert(HEAD_LEN <= ckernel::math::replay_buf_offset);
    sdv_imm32<p_sfpu::LREG4, 0x3C092F6Eu>();
    sdv_imm32<p_sfpu::LREG5, 0x3D2AADADu>();
    sdv_imm32<p_sfpu::LREG6, 0x3E2AAA28u>();
    sdv_imm32<p_sfpu::LREG7, 0x3EFFFFFBu>();
    TTI_REPLAY(0, HEAD_LEN, 1, 1);
    sdv_exp_head<SCALE_EN>(s);
    sdv_exp_tail();
#pragma GCC unroll 0
    for (int f = 0; f < faces; f++) {
#pragma GCC unroll 8
        for (int d = (f == 0) ? 1 : 0; d < ROWS; d++) {
            TTI_REPLAY(0, HEAD_LEN, 0, 0);
            sdv_exp_tail();
        }
        for (int k = 0; k < step; k++) {
            sdo_sfpu_next_face();
        }
    }
    for (int k = 0; k < skip; k++) {
        sdo_sfpu_next_face();
    }
}

template <bool SCALE_EN, int ROWS>
inline void sdv_exp_vm(const std::uint32_t idst, VectorMode vm, const std::uint32_t s) {
    sdo_sfpu_start(idst);
    if (vm == VectorMode::RC) {
        sdv_exp<SCALE_EN, ROWS>(4, 1, 0, s);
    } else if (vm == VectorMode::R) {
        sdv_exp<SCALE_EN, ROWS>(2, 1, 2, s);
    } else if (vm == VectorMode::C) {
        sdv_exp<SCALE_EN, ROWS>(2, 2, 0, s);
    } else {
        sdv_exp<SCALE_EN, ROWS>(1, 0, 0, s);
    }
    sdo_sfpu_done();
}

template <uint32_t P, int ITERATIONS>
inline void sdv_mul_rows() {
    TTI_SFPLOADI(0, 2, P & 0xFFFFu);
    TTI_SFPLOADI(0, 8, P >> 16);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(1, 0, ADDR_MOD_7, 0);
        TTI_SFPMUL(1, 0, p_sfpu::LCONST_0, 1, 0);
        TTI_SFPSTORE(1, 0, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
}

template <uint32_t S>
ALWI void sdv_corr_body() {
    TTI_SFPLOAD(0, 0, 7, 0);
    TTI_SFPLOAD(4, 0, 7, 64);
    TTI_SFPLOAD(3, 0, 7, 192);
    TTI_SFPLOAD(2, 0, 7, 128);
    TTI_SFPGT(0, 0, 4, 1);
    TTI_SFPSTORE(4, 0, 7, 128);
    TTI_SFPCOMPC(0, 0, 0, 0);
    TTI_SFPSTORE(0, 0, 7, 128);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPLOAD(1, 0, 7, 128);
    TTI_SFPADD(10, 1, 0, 0, 1);
    TTI_SFPADD(10, 1, 4, 1, 1);
    TTI_SFPMULI(S, 0, 0);
    TTI_SFPLOADI(5, 2, 43579);
    TTI_SFPLOADI(5, 8, 16312);
    TTI_SFPMUL(0, 5, 9, 5, 0);
    TTI_SFPLOADI(7, 1, 5541);
    TTI_SFP_STOCH_RND(0, 0, 9, 5, 5, 7);
    TTI_SFPCAST(5, 6, 0);
    TTI_SFPLOADI(4, 2, 29184);
    TTI_SFPLOADI(4, 8, 48945);
    TTI_SFPMAD(6, 4, 0, 0, 0);
    TTI_SFPLOADI(4, 2, 48782);
    TTI_SFPLOADI(4, 8, 46527);
    TTI_SFPMAD(6, 4, 0, 4, 0);
    TTI_SFPLOADI(0, 2, 12142);
    TTI_SFPLOADI(0, 8, 15369);
    TTI_SFPMAD(7, 4, 0, 0, 0);
    TTI_SFPLOADI(7, 2, 44461);
    TTI_SFPLOADI(7, 8, 15658);
    TTI_SFPMAD(0, 4, 7, 0, 0);
    TTI_SFPLOADI(7, 2, 43560);
    TTI_SFPLOADI(7, 8, 15914);
    TTI_SFPMAD(0, 4, 7, 0, 0);
    TTI_SFPLOADI(7, 2, 65531);
    TTI_SFPLOADI(7, 8, 16127);
    TTI_SFPMAD(0, 4, 7, 0, 0);
    TTI_SFPABS(0, 5, 5, 0);
    TTI_SFPMAD(0, 4, 10, 0, 0);
    TTI_SFPSETSGN(0, 5, 6, 0);
    TTI_SFPMAD(0, 4, 10, 4, 0);
    TTI_SFPMULI(32640, 0, 0);
    TTI_SFPEXEXP(0, 4, 5, 1);
    TTI_SFPIADD(0, 6, 5, 4);
    TTI_SFPIADD(3841, 5, 6, 1);
    TTI_SFPMOV(0, 5, 0, 0);
    TTI_SFPSETEXP(0, 4, 0, 0);
    TTI_SFPIADD(4095, 5, 5, 1);
    TTI_SFPMOV(0, 9, 0, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPMULI(S, 1, 0);
    TTI_SFPMOV(0, 1, 4, 2);
    TTI_SFPLOADI(1, 2, 43579);
    TTI_SFPLOADI(1, 8, 16312);
    TTI_SFPMUL(4, 1, 9, 1, 0);
    TTI_SFPLOADI(5, 1, 5541);
    TTI_SFP_STOCH_RND(0, 0, 9, 1, 7, 7);
    TTI_SFPCAST(7, 6, 0);
    TTI_SFPLOADI(1, 2, 29184);
    TTI_SFPLOADI(1, 8, 48945);
    TTI_SFPMAD(6, 1, 4, 1, 0);
    TTI_SFPLOADI(4, 2, 48782);
    TTI_SFPLOADI(4, 8, 46527);
    TTI_SFPMAD(6, 4, 1, 4, 0);
    TTI_SFPLOADI(1, 2, 12142);
    TTI_SFPLOADI(1, 8, 15369);
    TTI_SFPMAD(5, 4, 1, 1, 0);
    TTI_SFPLOADI(5, 2, 44461);
    TTI_SFPLOADI(5, 8, 15658);
    TTI_SFPMAD(1, 4, 5, 1, 0);
    TTI_SFPLOADI(5, 2, 43560);
    TTI_SFPLOADI(5, 8, 15914);
    TTI_SFPMAD(1, 4, 5, 1, 0);
    TTI_SFPLOADI(5, 2, 65531);
    TTI_SFPLOADI(5, 8, 16127);
    TTI_SFPMAD(1, 4, 5, 1, 0);
    TTI_SFPABS(0, 7, 7, 0);
    TTI_SFPMAD(1, 4, 10, 1, 0);
    TTI_SFPSETSGN(0, 7, 6, 0);
    TTI_SFPMAD(1, 4, 10, 4, 0);
    TTI_SFPMULI(32640, 1, 0);
    TTI_SFPEXEXP(0, 4, 5, 1);
    TTI_SFPIADD(0, 6, 5, 4);
    TTI_SFPIADD(3841, 5, 6, 1);
    TTI_SFPMOV(0, 5, 1, 0);
    TTI_SFPSETEXP(0, 4, 1, 0);
    TTI_SFPIADD(4095, 5, 5, 1);
    TTI_SFPMOV(0, 9, 1, 0);
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_SFPSTORE(0, 0, 7, 0);
    TTI_SFPSTORE(1, 0, 7, 64);
    TTI_SFPMUL(0, 3, 9, 0, 0);
    TTI_SFPSTORE(0, 0, 7, 192);
    TTI_SFPLOAD(0, 0, 7, 192);
    TTI_SFPMAD(1, 2, 0, 1, 0);
    TTI_SFPSTORE(1, 0, 7, 192);
}

template <uint32_t S>
inline void sdv_corr_rows() {
    for (int d = 0; d < 4; d++) {
        sdv_corr_body<S>();
        TTI_INCRWC(0, 4, 0, 0);
    }
}

ALWI void sdv_recip_one() {
    TTI_SFPLOAD(0, 0, ADDR_MOD_7, 0);
    TTI_SFPARECIP(0, 0, 0, 0);
    TTI_SFPSTORE(0, 0, ADDR_MOD_7, 0);
}

inline void sdv_recip_rows() {
    for (int d = 0; d < 4; d++) {
        sdv_recip_one();
        TTI_INCRWC(0, 4, 0, 0);
    }
}

template <int ITERATIONS>
ALWI void sdv_rnd_rows(uint32_t idst, uint32_t row0) {
    sdo_sfpu_start(idst);
    sdv_seek_rows4(row0);
    TTI_SFPLOADI(2, 2, 0x8000);
    TTI_SFPLOADI(1, 0, 0xFFFF);
    for (int face = 0; face < 2; face++) {
#pragma GCC unroll 8
        for (int it = 0; it < ITERATIONS; it++) {
            TTI_SFPLOAD(0, 0, ADDR_MOD_7, 0);
            TTI_SFPIADD(0, 2, 0, 4);
            TTI_SFPAND(0, 1, 0, 1);
            TTI_SFPSTORE(0, 0, ADDR_MOD_7, 0);
            TTI_INCRWC(0, 2, 0, 0);
        }
        sdo_sfpu_next_face();
    }
    sdo_sfpu_done();
}

template <std::uint16_t SCALE_BF16>
inline void sdv_exp_fc() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    TTI_SFPLOADI(p_sfpu::LREG3, 0xA, 56745);
    TTI_SFPLOADI(p_sfpu::LREG3, 0x8, 15659);
    TTI_SFPLOADI(p_sfpu::LREG4, 0xA, 62370);
    TTI_SFPLOADI(p_sfpu::LREG4, 0x8, 15915);
    TTI_SFPLOADI(p_sfpu::LREG5, 0xA, 64988);
    TTI_SFPLOADI(p_sfpu::LREG5, 0x8, 16127);
    TTI_SFPLOADI(p_sfpu::LREG6, 0xA, 64903);
    TTI_SFPLOADI(p_sfpu::LREG6, 0x8, 16255);
    TTI_SFPLOADI(p_sfpu::LREG7, 0xA, 1);
    TTI_SFPLOADI(p_sfpu::LREG7, 0x8, 16256);
    for (int d = 0; d < 4; d++) {
        TTI_SFPLOAD(p_sfpu::LREG2, 3, ADDR_MOD_7, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, 0, SCALE_BF16);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPLOADI(p_sfpu::LREG1, 0xA, 43579);
        TTI_SFPLOADI(p_sfpu::LREG1, 0x8, 16312);
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
        TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG0, p_sfpu::LREG1, 3);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPLOADI(p_sfpu::LREG0, 0xA, 29208);
        TTI_SFPLOADI(p_sfpu::LREG0, 0x8, 48945);
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG0, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG6, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LREG7, p_sfpu::LREG0, 0);
        TT_SFPADDI(0x42fe, p_sfpu::LREG1, 0);
        TTI_SFP_STOCH_RND(0, 0, p_sfpu::LCONST_0, p_sfpu::LREG1, p_sfpu::LREG2, 2);
        TTI_SFPSETEXP(0, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG2, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, 6);
        TTI_SFPLOADI(p_sfpu::LREG2, 0, 0);
        TTI_SFPENCC(0, 0, 0, 0);
        TTI_SFPSTORE(p_sfpu::LREG2, 3, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 4, 0, 0);
    }
}

inline void sdv_bmax_init() {
    TTI_SFPSWAP(0, p_sfpu::LREG2, 12, 9);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, 0, 13, 6);
    TTI_SFPLOADI(0, 10, 0x80 | 0x00 | (1 << 3) | 4);
    TTI_SFPLOADI(0, 8, 0x80 | 0x40 | (3 << 3) | 5);
    TTI_SFPCONFIG(0, 4 + 0, 0);
    TTI_SFPLOADI(0, 10, 0);
    TTI_SFPLOADI(0, 8, (0x00 | 0x40 | (2 << 3) | 3) << 8);
    TTI_SFPCONFIG(0, 4 + 1, 0);
    TTI_SFPCONFIG(0x330, 8, 1);
}

inline void sdv_bmax(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;
    constexpr int b = p_sfpu::LREG2;
    constexpr int c = p_sfpu::LREG3;
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        int a = i & 1;
        TT_SFPLOADMACRO((0 << 2) | (a & 3), 0, ADDR_MOD_7, offset0 | (a >> 2));
        TT_SFPLOAD(b, 0, ADDR_MOD_7, offset1);
        TT_SFPLOADMACRO((1 << 2) | (c & 3), 0, ADDR_MOD_6, offset2 | (c >> 2));
    }
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
}

}  // namespace ckernel
#endif
#endif
