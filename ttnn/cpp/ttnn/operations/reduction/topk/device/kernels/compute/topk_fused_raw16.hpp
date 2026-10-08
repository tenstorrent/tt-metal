// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Fused [bf16 value | u16 index] keys for kernels whose bf16 values travel as raw u16 words in a 32-bit DEST section
// (the unpacker, the datacopy and the packer keep the u16 index format): the comparator-stable network's order from
// the plain network, with the comparator path's handling of zeros, denormals and NaNs reproduced where it happens.

#include <cstdint>
#include "api/compute/topk.h"
#include "ckernel_sfpu.h"

#ifdef TRISC_MATH
namespace topk_fused_raw16 {
using namespace ckernel;
using namespace ckernel::sfpu;

// What the fuse does to a value first: nothing; -0 to +0; or every zero and denormal to +0.
enum class Zeros { Keep, FoldNegative, ToPositive };

// _topk_fuse_tile_ for value tiles moved in as raw u16 words: DEST 0,1 hold [garbage | bf16 bits].
template <bool largest, Zeros zeros>
inline void fuse_raw16_slab() {
    constexpr int body = zeros == Zeros::Keep ? 10 : (zeros == Zeros::FoldNegative ? 15 : 14);
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = TOPK_LO16_MASK;
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    load_replay_buf<Exec>(0, body, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 128);
        TTI_SFPSHFT(16, 0, p_sfpu::LREG0, 1);
        if constexpr (zeros == Zeros::FoldNegative) {
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPSHFT(1, 0, p_sfpu::LREG2, 1);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            TOPK_SFPENCC_ALL_LANES_ON();
        } else if constexpr (zeros == Zeros::ToPositive) {
            TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
            TOPK_SFPENCC_ALL_LANES_ON();
        }
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, largest ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TOPK_SFPENCC_ALL_LANES_ON();
        TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG0, 0);
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    });
    for (int i = 1; i < 64; i++) {
        lltt::replay(0, body);
    }
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    topk_replay_init = 0;
}

// Flushes the denormal values of the keys in DEST 0,1 to signed zero, index kept, as a BF16 store does. No replay: a
// local sort's later phases replay the loads and stores its phase 0 recorded.
inline void flush_key_denormals() {
    TOPK_SFPENCC_ALL_LANES_ON();
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0x8000);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    for (int i = 0; i < 64; i++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
        TOPK_SFPENCC_ALL_LANES_ON();
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    }
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

// Makes a NaN value of the keys in DEST 0,1 the infinity of its sign, index kept, as a pack out of a 16-bit DEST does.
inline void keys_nan_to_inf() {
    constexpr int body = 8;
    TOPK_SFPENCC_ALL_LANES_ON();
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    load_replay_buf<Exec>(0, body, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
        TTI_SFPIADD(
            (-255) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
        TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
        TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
        TOPK_SFPENCC_ALL_LANES_ON();
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_INCRWC(0, 2, 0, 0);
    });
    for (int i = 1; i < 64; i++) {
        lltt::replay(0, body);
    }
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    topk_replay_init = 0;
}

// Splits keys into u16 value and index words in the packer-visible high half (mode 9, as the u16 index pack).
// nan_to_inf: a NaN value becomes the infinity of its sign, as a pack out of a 16-bit DEST makes it.
template <bool largest, bool nan_to_inf>
inline void defuse_raw16(const int num_tiles) {
    constexpr std::uint32_t pack_u16 = TOPK_SFPSTORE_MODE_PACK_UINT16;
    constexpr int body = nan_to_inf ? 15 : 10;
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = TOPK_LO16_MASK;
    if constexpr (nan_to_inf) {
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0xFF80);
        TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0x0000);
    }
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    load_replay_buf<Exec>(0, body, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, largest ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TOPK_SFPENCC_ALL_LANES_ON();
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        if constexpr (nan_to_inf) {
            TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
            TTI_SFPIADD(
                (-255) & 0xFFF, p_sfpu::LREG2, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_IMM | sfpi::SFPIADD_MOD1_CC_NONE);
            TTI_SFPSETCC(0, p_sfpu::LREG2, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
            TTI_SFPAND(0, p_sfpu::LREG3, p_sfpu::LREG0, 0);
            TOPK_SFPENCC_ALL_LANES_ON();
        }
        TTI_SFPSHFT((-16) & 0xFFF, 0, p_sfpu::LREG0, 1);
        TTI_SFPSTORE(p_sfpu::LREG0, pack_u16, ADDR_MOD_7, 0);
        TTI_SFPSTORE(p_sfpu::LREG1, pack_u16, ADDR_MOD_7, 128);
        TTI_INCRWC(0, 2, 0, 0);
    });
    const int n = 32 * num_tiles;
    for (int i = 1; i < n; i++) {
        lltt::replay(0, body);
    }
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    topk_replay_init = 0;
}
}  // namespace topk_fused_raw16
#endif
