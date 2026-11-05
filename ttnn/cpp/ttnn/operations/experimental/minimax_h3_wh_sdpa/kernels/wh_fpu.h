// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The WH matrix pipeline shared by the TRISC kernels: unpackers feeding 64-row fills, the math thread's fills of
// MVMULs and the four packers draining 64-row column faces of Dst. A fill is 32 MVMULs a fidelity phase: SrcA face c
// (16 x 16) times SrcB row group g (8 x 16) into Dst row 64 c + 8 g past the math thread's Dst offset.
#pragma once

#include "wh_isa.h"

// Unpacker 0 reads chunk W * z0 + Z (1024 BF16) into SrcA rows 0-63, unpacker 1 face W * z1 + Z into SrcB rows 0-63;
// each UNPACR steps Z.
static constexpr uint32_t wh_unpack_a = TT_OP_UNPACR(0u, 1u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u);
static constexpr uint32_t wh_unpack_b = TT_OP_UNPACR(1u, 1u, 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 1u);
// Gives the matrix unit unpacker 0's next SrcA bank for MOVD2A to fill: a clear that waits, as an UNPACR does, for
// the bank to come back from the matrix unit, then the hand-over, which does not wait (UNPACR_NOP_SETDVALID.md).
static constexpr uint32_t wh_unpack_a_clear = TT_OP_UNPACR_NOP(0u, 0x11u);
static constexpr uint32_t wh_unpack_a_valid = TT_OP_UNPACR_NOP(0u, 7u);

static inline void wh_unpack_setup(
    uint32_t base0, uint32_t z0, uint32_t throttle0, uint32_t base1, uint32_t z1, uint32_t throttle1) {
    static constexpr uint32_t zero[] = {36u, 37u, 38u, 39u, 40u, 41u, 44u, 45u, 46u, 47u,  48u,  49u,
                                        54u, 55u, 62u, 63u, 73u, 75u, 80u, 94u, 95u, 102u, 103u, 120u};
    for (const uint32_t index : zero) {
        wh_cfg(index, 0u);
    }
    const uint32_t descriptor = wh_fmt_bf16 | 1u << 4 | 1024u << 16;
    wh_cfg(THCON_SEC0_REG0_TileDescriptor_ADDR32, descriptor);
    wh_cfg(THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1u, 1u | z0 << 16);
    wh_cfg(THCON_SEC0_REG2_Out_data_format_ADDR32, wh_fmt_bf16 | throttle0 << 4);
    wh_cfg(THCON_SEC0_REG2_Disable_zero_compress_cntx0_ADDR32, 1u);
    wh_cfg(THCON_SEC0_REG3_Base_address_ADDR32, wh_l1_base(base0));
    wh_cfg(THCON_SEC0_REG5_Dest_cntx0_address_ADDR32, 64u | 64u << 16);
    wh_cfg(THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 1024u | 1024u << 16);
    wh_cfg(THCON_SEC1_REG0_TileDescriptor_ADDR32, descriptor);
    wh_cfg(THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1u, 1u | z1 << 16);
    wh_cfg(THCON_SEC1_REG2_Out_data_format_ADDR32, wh_fmt_bf16 | throttle1 << 4);
    wh_cfg(THCON_SEC1_REG2_Disable_zero_compress_cntx0_ADDR32, 1u);
    wh_cfg(THCON_SEC1_REG3_Base_address_ADDR32, wh_l1_base(base1));
    wh_thread_reset();
    wh_push(TT_OP_SETC16(SRCA_SET_Base_ADDR32, 4u));  // SetOvrdWithAddr: unpacker 0 addresses all 64 SrcA rows
    wh_push(TT_OP_SETADCXX(wh_adc_unpack0, 1023u, 0u));
    wh_push(TT_OP_SETADCXX(wh_adc_unpack1, 1023u, 0u));
}

// BF16 sources into an FP32 Dst, or a 16-bit BF16 Dst, zeroed. AddrMods: 0 next row group, 1 next column face, 2 next
// fidelity phase, 3 the fill's end (Src counters and the phase cleared).
static inline void wh_math_setup(bool dst16) {
    wh_cfg(ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0u);
    const uint32_t dst = dst16 ? wh_fmt_bf16 << 25 : wh_fmt_fp32 << 25 | 1u << 29;
    wh_cfg(ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, wh_fmt_bf16 << 17 | wh_fmt_bf16 << 21 | dst);
    wh_cfg(ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32, 0u);  // Src denormals flush
    wh_thread_reset();
    wh_math_addr_mod(0u, 8u << 8, 0u);
    wh_math_addr_mod(1u, 1u << 15 | 16u, 0u);
    wh_math_addr_mod(2u, 1u << 15 | 1u << 7, 1u << 13);
    wh_math_addr_mod(3u, 1u << 15 | 1u << 7, 1u << 15);
    if (dst16) {
        wh_push(TT_OP_ZEROACC(wh_zeroacc_all_16b, 0u, 0u));
    } else {
        wh_push(TT_OP_ZEROACC(wh_zeroacc_all_32b, 0u, 0u));
    }
}

// MVMUL bank flips at a fill's end: SrcB's alone (its SrcA serves the next fill too), or both.
static constexpr uint32_t wh_flip_b = 2u, wh_flip_ab = 3u;

// Where a fill puts row group g of column face c: row 64 c + 8 g (face major), or 64 (g / 2) + 16 c + 8 (g % 2)
// (slice major: 16-row slices of every face, so a 64-row packer face is one 16-deep SrcA fill of four faces).
enum class WhFillRows { face, slice };

static constexpr uint32_t wh_fill_row(WhFillRows rows, uint32_t c, uint32_t g) {
    return rows == WhFillRows::face ? 64u * c + 8u * g : 64u * (g / 2u) + 16u * c + 8u * (g % 2u);
}

// A fill's MVMULs at each phase: replay slots 0-30, then the last row group of column face 3 (row 248 in both
// layouts), which steps the phase or ends the fill with `flip`. The first fill records the replay as it runs.
static inline void wh_fill(uint32_t phases, bool record, uint32_t flip, WhFillRows rows = WhFillRows::face) {
    for (uint32_t phase = 0; phase < phases; ++phase) {
        if (record && phase == 0u) {
            wh_push(TT_OP_REPLAY(0u, 31u, 1u, 1u));
            for (uint32_t i = 0; i < 31u; ++i) {
                wh_push(TT_OP_MVMUL(0u, 0u, i % 8u < 7u ? 0u : 1u, wh_fill_row(rows, i / 8u, i % 8u)));
            }
        } else {
            wh_push(TT_OP_REPLAY(0u, 31u, 0u, 0u));
        }
        if (phase + 1u < phases) {
            wh_push(TT_OP_MVMUL(0u, 0u, 2u, 248u));
        } else if (flip == wh_flip_ab) {
            wh_push(TT_OP_MVMUL(wh_flip_ab, 0u, 3u, 248u));
        } else {
            wh_push(TT_OP_MVMUL(wh_flip_b, 0u, 3u, 248u));
        }
    }
}

// Per packer: L1_Dest_addr, the format word after it, the word with Pack_L1_Acc.
static constexpr uint32_t wh_packer_dest[4] = {
    THCON_SEC0_REG1_L1_Dest_addr_ADDR32,
    THCON_SEC0_REG8_L1_Dest_addr_ADDR32,
    THCON_SEC1_REG1_L1_Dest_addr_ADDR32,
    THCON_SEC1_REG8_L1_Dest_addr_ADDR32};

// Packer c packs Dst rows 64 c to 64 c + 63 past the pack Y counter (16 rows a step) in one PACR of 1024 datums:
// FP32 from an FP32 Dst, or BF16 from a 16-bit Dst.
static inline void wh_pack_setup(bool dst16) {
    static constexpr uint32_t zero[] = {
        9u, 10u, 11u, 12u, 13u, 15u, 16u, 17u, 18u, 19u, 56u, 84u, 88u, 89u, 90u, 91u, 96u, 124u};
    for (const uint32_t index : zero) {
        wh_cfg(index, 0u);
    }
    wh_cfg(PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32, (dst16 ? 512u : 1024u) << 12);
    wh_cfg(PCK_DEST_RD_CTRL_Read_32b_data_ADDR32, dst16 ? 0u : 1u);
    for (uint32_t i = 0; i < 4u; ++i) {
        wh_cfg(PCK_EDGE_OFFSET_SEC0_mask_ADDR32 + i, 0xFFFFu);
        wh_cfg(PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i, 0x100u);
        wh_cfg(wh_packer_dest[i] + 1u, dst16 ? 1u | wh_fmt_bf16 << 4 | wh_fmt_bf16 << 8 : 1u);  // uncompressed, out, in
        wh_cfg(wh_packer_dest[i] + 2u, 0u);
        wh_cfg(DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32 + i, 64u * i);
    }
    wh_thread_reset();
    wh_push(TT_OP_SETADCXX(wh_adc_pack, 1023u, 0u));
}

// Sets packer i's configuration word `offset` past its L1_Dest_addr to value + i step, through GPR pair `gpr`.
static inline void wh_set_packer_cfg(uint32_t gpr, uint32_t offset, uint32_t value, uint32_t step) {
    for (uint32_t i = 0; i < 4u; ++i) {
        wh_set_gpr(gpr, value + i * step);
        wh_push(TT_OP_WRCFG(gpr, 0u, wh_packer_dest[i] + offset));
    }
}

// Points the packers at the 256 Dst rows from row 16 y and L1 `addr`, packer c's column face at addr + c face_bytes,
// once earlier packs are done. The DMANOP keeps the configuration writes two instructions ahead of a PACR.
static inline void wh_pack_prepare(uint32_t gpr, uint32_t addr, uint32_t face_bytes, uint32_t y) {
    wh_push(TT_OP_STALLWAIT(wh_block_sync | wh_block_cfg | wh_block_thcon, wh_wait_pack));
    wh_set_packer_cfg(gpr, 0u, wh_l1_base(addr), face_bytes / 16u);
    wh_push(TT_OP_SETADC(wh_adc_pack, 0u, wh_adc_y, y));
    wh_push(TT_OP_DMANOP);
}

// The pack wh_pack_prepare set up.
static inline void wh_pack_prepared() { wh_push(TT_OP_PACR(0u, 0u, 0xFu, 0u, 0u, 0u, 1u)); }

static inline void wh_pack_region(uint32_t gpr, uint32_t addr, uint32_t face_bytes, uint32_t y) {
    wh_pack_prepare(gpr, addr, face_bytes, y);
    wh_pack_prepared();
}
