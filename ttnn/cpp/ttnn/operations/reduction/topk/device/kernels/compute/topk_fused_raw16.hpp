// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Fused [bf16 value | u16 index] keys for kernels whose bf16 values travel as raw u16 words in a 32-bit DEST section
// (the unpacker, the datacopy and the packer keep the u16 index format): the comparator-stable network's order from
// the plain network, with the comparator path's -0 and denormal handling reproduced where it happens.

#include <cstdint>
#include "api/compute/topk.h"
#include "ckernel_sfpu.h"

#ifdef TRISC_MATH
namespace topk_fused_raw16 {
using namespace ckernel;
using namespace ckernel::sfpu;

// _topk_fuse_tile_ for value tiles moved in as raw u16 words: DEST 0,1 hold [garbage | bf16 bits].
// canonicalize_negzero: -0 becomes +0 first, as the comparator path's bf16 copy_tile does before each merge (its
// transposes before the local sort keep -0, which sorts below +0 there).
template <bool largest, bool canonicalize_negzero>
inline void fuse_raw16_slab() {
    constexpr int body = canonicalize_negzero ? 15 : 10;
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = TOPK_LO16_MASK;
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    load_replay_buf<Exec>(0, body, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_7, 128);
        TTI_SFPSHFT(16, 0, p_sfpu::LREG0, 1);
        if constexpr (canonicalize_negzero) {
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPSHFT(1, 0, p_sfpu::LREG2, 1);
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

// Flushes the denormal values of the keys in DEST 0,1 to signed zero, index kept: the comparator path's BF16
// SFPSTORE does this after the local sort's phase 0, so its later phases compare the flushed values.
inline void flush_key_denormals() {
    constexpr int body = 7;
    TOPK_SFPENCC_ALL_LANES_ON();
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, 0x8000);
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, 0xFFFF);
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    load_replay_buf<Exec>(0, body, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_NODEBIAS);
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
template <bool largest>
inline void defuse_raw16(const int num_tiles) {
    constexpr std::uint32_t pack_u16 = TOPK_SFPSTORE_MODE_PACK_UINT16;
    constexpr int body = 10;
    TOPK_SFPENCC_ALL_LANES_ON();
    sfpi::vConstIntPrgm0 = TOPK_LO16_MASK;
    set_dst_write_addr(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    load_replay_buf<Exec>(0, body, [] {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_7, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, largest ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        TTI_SFPXOR(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
        TOPK_SFPENCC_ALL_LANES_ON();
        TTI_SFPAND(0, p_sfpu::LREG12, p_sfpu::LREG1, 0);
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
