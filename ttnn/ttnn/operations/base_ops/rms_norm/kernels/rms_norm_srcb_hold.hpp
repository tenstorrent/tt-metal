// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// "Hold srcB" broadcast multiply, raw Blackhole LLK (Perf 2, bcast_srcb_reuse) — the γ pass (x * γ, Row broadcast)
// and the inv pass ((x·γ | x) * inv_rms, Col broadcast) of rms_norm's normalize stage.
//
// HELPER BYPASS — measured justification (do NOT "fix" back to the chain without re-measuring):
//   helper:   ckl::eltwise_chain + ckl::BinaryFpu<Mul, x, BroadcastDim::Row | Col> (underneath: the stock
//             llk_unpack_AB<BType> + llk_math_eltwise_binary<ELWMUL, BType> pair, per tile).
//   gap:      capability. BinaryFpu re-unpacks the broadcast operand into srcB for EVERY tile and hands both source
//             registers over one 16x16 face at a time (4 srcA + 4 srcB UNPACRs per tile for Row, 4 + 2 for Col, one
//             data-valid handshake per face, one unpacker config-context round trip per tile). No InputSpec / policy
//             can keep srcB valid across tiles, stream whole-tile srcA under one context, or address per-face srcB rows
//             inside a resident broadcast tile.
//   measured: (1,1,8192,1024) BLOCK_SHARDED [1024,128] 8x8, bf16, HiFi2, 16-bit DEST, Blackhole p150b: whole op
//             15,685 -> 14,801 ns (chain -> hold, both passes; medians of 3), bit-exact. Inside the op the UNPACK
//             thread (TRISC0) is the longest one during these passes (γ 3,959 vs math 3,540 ns), so halving its
//             per-tile work moves the wall. Stock per-tile LLK without the chain measured flat (15,732 ns): the win is
//             the hold, not chain overhead. Single-core bench (16 reps x 128 tiles): γ 57,732 -> 54,893 ns, inv
//             60,813 -> 55,131 ns.
// MECHANISM: the broadcast tile is unpacked into srcB ONCE per run (whole tile, rows 0..63 of one srcB bank, one
// data-valid) and kept resident while n srcA tiles stream through (whole tile per UNPACR, one data-valid per tile,
// all n in ONE unpacker context); the math addresses the right srcB rows per face itself and never clears srcB until
// the run ends. Same ELWMUL instruction, same broadcast mode, same fidelity phases, same operand bits: the products
// are bit-identical to the stock path.
// Prior art: llk_unpack_AB_sub_bcast_col_custom (whole-tile UNPACR of 1 srcB + ct_dim srcA tiles in one context,
// W-counter advance) and llk_math_eltwise_binary_custom.h::_llk_math_bcast_cols_reuse_custom_ (LoFi Col-bcast with
// srcB face reuse). This adds: the HiFi fidelity phases, the Row-broadcast srcB addressing, a strided srcA walk, and
// the block-float srcA address path.
// Blackhole only (raw Blackhole LLK sequences); the compute kernel keeps the chain on other architectures.

#pragma once

#include <stdint.h>

namespace rms_norm_hold {

#ifdef TRISC_UNPACK
// Both unpackers move a whole 32x32 tile (4 faces, 1024 datums) per UNPACR. No transpose.
// The next stock init (any llk_unpack_*_init) re-establishes its own X span.
ALWI void unpack_hold_init() {
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(0);
    TTI_SETADCXX(p_setadc::UNP_A, 1024 - 1, 0x0);
    TTI_SETADCXX(p_setadc::UNP_B, 1024 - 1, 0x0);
}

// One run: srcB <- tile at addr_b (once), then n srcA tiles at addr_a + k * stride_tiles tiles (k < n),
// all in one unpacker config context. Addresses are the unpacker's (16 B units, fifo_rd_ptr - 1 based).
ALWI void unpack_hold_run(uint32_t addr_a, uint32_t addr_b, uint32_t n, uint32_t stride_tiles) {
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);  // reset both unpackers' Z/W counters
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(addr_a, addr_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    for (uint32_t k = 0; k < n; ++k) {
        TTI_UNPACR(SrcA, 0b00000000, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        // advance the srcA L1 pointer by stride_tiles tiles (W counter; the field is 3 bits wide)
        uint32_t s = stride_tiles;
        while (s > 7) {
            TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 7, 0);
            s -= 7;
        }
        TT_INCADCZW(p_setadc::UNP_A, 0, 0, s, 0);
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

// Prefetch form: n srcA tiles first, THEN (if load_b) the NEXT run's srcB tile (it lands in the other srcB bank
// while the math still holds this run's). The first run's srcB is loaded by a call with n = 0, load_b = true.
ALWI void unpack_hold_run_pf(uint32_t addr_a, uint32_t addr_b_next, uint32_t n, uint32_t stride_tiles, bool load_b) {
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(addr_a, addr_b_next, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (uint32_t k = 0; k < n; ++k) {
        TTI_UNPACR(SrcA, 0b00000000, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
        uint32_t s = stride_tiles;
        while (s > 7) {
            TTI_INCADCZW(p_setadc::UNP_A, 0, 0, 7, 0);
            s -= 7;
        }
        TT_INCADCZW(p_setadc::UNP_A, 0, 0, s, 0);
    }
    if (load_b) {
        TTI_UNPACR(SrcB, 0b00000000, 0, 0, 0, 1, 1 /*dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

// Address of tile `idx` of CB `cb` as the unpacker sees it.
ALWI uint32_t unpack_addr(uint32_t cb, uint32_t idx) {
    const uint32_t id = get_operand_id(cb);
    return get_local_cb_interface(id).fifo_rd_ptr - 1 + get_local_cb_interface(id).fifo_page_size * idx;
}
#endif

#ifdef TRISC_MATH
// COL = srcB column 0 broadcast across the row (inv_rms); !COL = srcB row 0 broadcast down the column (gamma).
// srcB holds the WHOLE broadcast tile: face f at rows 16f..16f+15. Per srcA face f the math reads
//   Col: srcB rows of face 0 (f = 0, 1) / face 2 (f = 2, 3), stepping 8 rows like srcA;
//   Row: srcB row 0 of face 0 (f even) / face 1 (f odd), i.e. srcB row 0 / 16.
// Addr mods 4..7 (0..3 stay the stock binary ones):
//   AM4  step 8 rows                        srcA +8, srcB +8 (Col) / 0 (Row), dest +8
//   AM5  end of a non-last fidelity phase   srcA -8, srcB -8 (Col) / 0 (Row), dest -> CR, fidelity +1
//   AM6  end of an even face (0, 2)         srcA +8, srcB -8 (Col) / +16 (Row), dest +8 -> CR, fidelity clr
//   AM7  end of an odd face (1, 3)          srcA +8, srcB +24 (Col) / -16 (Row), dest +8 -> CR, fidelity clr
template <bool COL>
ALWI void math_hold_init() {
    constexpr uint8_t m8 = 0x3F & -8;
    addr_mod_t{
        .srca = {.incr = 8},
        .srcb = {.incr = static_cast<uint8_t>(COL ? 8 : 0)},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_4);
    addr_mod_t{
        .srca = {.incr = m8},
        .srcb = {.incr = static_cast<uint8_t>(COL ? m8 : 0)},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = 1},
    }
        .set(ADDR_MOD_5);
    addr_mod_t{
        .srca = {.incr = 8},
        .srcb = {.incr = static_cast<uint8_t>(COL ? m8 : 16)},
        .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_6);
    addr_mod_t{
        .srca = {.incr = 8},
        .srcb = {.incr = static_cast<uint8_t>(COL ? 24 : (0x3F & -16))},
        .dest = {.incr = 8, .clr = 0, .cr = 0, .c_to_cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_7);
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// One srcA tile (already data-valid) times the resident srcB, into DEST slot dst. Releases srcA only.
template <bool COL>
ALWI void math_hold_tile(uint32_t dst) {
    constexpr uint32_t BC = COL ? p_elwise::SRCB_BCAST_COL : p_elwise::SRCB_BCAST_ROW;
    constexpr uint32_t FID = static_cast<uint32_t>(MATH_FIDELITY);
    constexpr uint32_t NPH = FID == 0 ? 1 : FID;  // fidelity phases (LoFi 1, HiFi2 2, HiFi3 3, HiFi4 4)
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst);
#pragma GCC unroll 4
    for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll 4
        for (uint32_t p = 0; p + 1 < NPH; ++p) {
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, BC, ADDR_MOD_4, 0);
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, BC, ADDR_MOD_5, 0);
        }
        TTI_ELWMUL(p_setrwc::CLR_NONE, 0, BC, ADDR_MOD_4, 0);
        if (f & 1) {
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, BC, ADDR_MOD_7, 0);
        } else {
            TTI_ELWMUL(p_setrwc::CLR_NONE, 0, BC, ADDR_MOD_6, 0);
        }
    }
    // release this srcA bank; srcA / srcB / dest counters back to 0 for the next tile
    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_ABD);
}

// End of a run: release the resident srcB bank.
ALWI void math_hold_release_b() { TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD); }
#endif

}  // namespace rms_norm_hold
