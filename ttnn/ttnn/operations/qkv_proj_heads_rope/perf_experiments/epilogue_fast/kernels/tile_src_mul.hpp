// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Whole-tile-src, phase-major ELWMUL (raw LLK; bypasses mul_tiles and the eltwise_chain BinaryFpu
// element, which only issue the face-major LLK sequence).
//
// mul_tiles runs each fidelity phase per 16-row face (MOP per face: phase 0 rows 0-7, 8-15, phase 1 rows
// 0-7, 8-15, ...), so the ELWMUL that accumulates into DEST rows r follows the one that wrote rows r two
// instructions earlier, and the fp32 DEST read-modify-write stalls on it. Measured on the rotary epilogue
// (p100a): HiFi3 and LoFi then cost ~15 vs ~8 cycles per face, i.e. ~2.5 cycles per ELWMUL.
// phase_major = true unpacks each operand as one 64-row tile into one src bank (one UNPACR, dvalid once,
// as matmul's full-tile unpack does) and runs ONE MOP per tile product with the fidelity phase outermost
// over all 8 row chunks, so a DEST row recurs only every 8th ELWMUL: 1.95x on the epilogue, and LoFi ==
// HiFi3 time (the multiply is off the critical path; the per-product unpack call now binds).
// phase_major = false keeps the face-major order inside the whole-tile bank (measured: no gain - the
// per-face src handshake is not the cost).
// Same ELWMUL instructions, same fidelity phases, same DEST accumulation order per element -> output is
// bit-identical to mul_tiles (checked vs the baseline on every shape / fidelity run).
//
// Call order: mul_init(cb_a, cb_b) (formats, hw config), then tile_src_mul_init<>(), then tile_src_mul<>()
// per tile product. A later op's init must reprogram unpacker x_end, ADDR_MOD_0/2/3 and the math MOP
// (mul_init / copy_init / matmul init all do).

#pragma once

#include <cstdint>

#include "api/compute/eltwise_binary.h"

namespace tile_src {

#ifdef TRISC_UNPACK
ALWI void unpack_init() { TTI_SETADCXX(p_setadc::UNP_AB, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0); }

#ifndef TILE_SRC_DIAG_B_NOP
#define TILE_SRC_DIAG_B_NOP 0
#endif
ALWI void unpack_tile(uint32_t cb_a, uint32_t tile_a, uint32_t cb_b, uint32_t tile_b) {
    const uint32_t id_a = get_operand_id(cb_a);
    const uint32_t id_b = get_operand_id(cb_b);
    const uint32_t address_a =
        get_local_cb_interface(id_a).fifo_rd_ptr - 1 + get_local_cb_interface(id_a).fifo_page_size * tile_a;
    const uint32_t address_b =
        get_local_cb_interface(id_b).fifo_rd_ptr - 1 + get_local_cb_interface(id_b).fifo_page_size * tile_b;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    _llk_unpack_configure_addresses_(address_a, address_b, cfg);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    if constexpr (TILE_SRC_DIAG_B_NOP == 2) {
        TTI_UNPACR_NOP(
            SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    } else {
        TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    }
    if constexpr (TILE_SRC_DIAG_B_NOP) {
        // Diagnostic only (wrong output): hand SrcB over without unpacking it.
        TTI_UNPACR_NOP(
            SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    } else {
        TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
// phase_major = false: face-major (as mul_tiles): per face, fidelity phases over its 2 8-row chunks; one
//   MOP run per face, so the same DEST rows are re-accumulated every 2nd ELWMUL.
// phase_major = true: one MOP per tile, fidelity phase outermost over all 8 chunks of the 64-row tile, so
//   the same DEST rows recur only every 8th ELWMUL.
template <bool phase_major>
ALWI void math_init() {
    constexpr auto fidelity = static_cast<MathFidelity>(MATH_FIDELITY);
    // MOP outer loop = fidelity phases (LoFi: one).
    constexpr uint32_t phases = is_high_fidelity(fidelity) ? to_underlying(fidelity) : 1;
    constexpr uint8_t fidelity_incr = is_high_fidelity(fidelity) ? 1 : 0;
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_0);
    if constexpr (phase_major) {
        // End of a phase: back to tile row 0, next phase.
        addr_mod_t{.srca = {.clr = 1}, .srcb = {.clr = 1}, .dest = {.clr = 1}, .fidelity = {.incr = fidelity_incr}}.set(
            ADDR_MOD_2);
        addr_mod_t{.srca = {.clr = 1}, .srcb = {.clr = 1}, .dest = {.clr = 1}, .fidelity = {.clr = 1}}.set(ADDR_MOD_3);
        ckernel_template tmp(phases, 8, TT_OP_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
        // The tile's last ELWMUL releases both whole-tile src banks.
        tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_AB, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0));
        tmp.program();
    } else {
        // End of a non-final phase: back to the face start (src / dest carriage return), next phase.
        addr_mod_t{
            .srca = {.incr = 0, .cr = 1},
            .srcb = {.incr = 0, .cr = 1},
            .dest = {.incr = 0, .cr = 1},
            .fidelity = {.incr = fidelity_incr}}
            .set(ADDR_MOD_2);
        // End of the face: src carriage return advances to the next face (Cr += 16), dest moves on.
        addr_mod_t{
            .srca = {.incr = 16, .cr = 1},
            .srcb = {.incr = 16, .cr = 1},
            .dest = {.incr = 8, .c_to_cr = 1},
            .fidelity = {.clr = 1}}
            .set(ADDR_MOD_3);
        ckernel_template tmp(phases, 2, TT_OP_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0));
        tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
        tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0));
        tmp.program();
    }
    ckernel::math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool phase_major>
ALWI void math_tile(uint32_t dst_index) {
    ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    if constexpr (phase_major) {
        ckernel_template::run();
    } else {
#pragma GCC unroll 4
        for (uint32_t face = 0; face < 4; ++face) {
            ckernel_template::run();
        }
        // Release both whole-tile src banks; counters and carriage returns back to row 0.
        TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_AB, 0, 0, 0, p_setrwc::SET_AB);
    }
    ckernel::math::clear_dst_reg_addr();
}
#endif

template <bool phase_major>
ALWI void tile_src_mul_init() {
    UNPACK((unpack_init()));
    MATH((math_init<phase_major>()));
}

template <bool phase_major>
ALWI void tile_src_mul(uint32_t cb_a, uint32_t cb_b, uint32_t tile_a, uint32_t tile_b, uint32_t dst) {
    UNPACK((unpack_tile(cb_a, tile_a, cb_b, tile_b)));
    MATH((math_tile<phase_major>(dst)));
}

}  // namespace tile_src
