// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Whole-tile-src, phase-major ELWMUL (raw LLK).
//
// Bypasses mul_tiles and the eltwise_chain BinaryFpu element, which only issue the face-major sequence: per
// 16-row face, each fidelity phase over its two 8-row chunks, so the ELWMUL accumulating into DEST rows r
// follows the one that wrote rows r two instructions earlier and the fp32 DEST read-modify-write stalls on it
// (measured on the rotary epilogue, p100a: ~2.5 cycles per ELWMUL, HiFi3 ~15 vs LoFi ~8 cycles per face).
// Here each operand is unpacked as one 64-row tile into one src bank (one UNPACR, dvalid once, as matmul's
// full-tile unpack does) and ONE MOP per tile product runs the fidelity phase outermost over all 8 row
// chunks, so a DEST row recurs only every 8th ELWMUL. Perf 1: rotary epilogue 8422 -> 4319 ns per 11x8 block
// (isolated, 4 heads), LoFi time == HiFi3 time (the per-product unpack call binds now).
// Same ELWMUL instructions, fidelity phases and per-element DEST accumulation order -> bit-identical to
// mul_tiles (checked on HiFi3 / HiFi4 / LoFi, full and half sync).
//
// Perf 2 (measured, 8x512 94.9 -> 94.1 us alone, bit-identical): each A tile (the fp32 partial) is unpacked
// into srcA once and multiplied by two B tiles; only srcB is re-unpacked for the second product. The rotary
// epilogue is unpack-bound (forcing one fidelity phase measured the same), and mul_tiles / eltwise_chain
// BinaryFpu unpack both operands for every product, so no helper can keep srcA across products.
//
// Call order: mul_init(cb_a, cb_b) (formats, hw config), then tile_src_mul_init(), then tile_src_mul_pair() per
// A tile. A later op's init must reprogram unpacker x_end, ADDR_MOD_0/2/3 and the math MOP
// (mul_init / copy_init / matmul init all do).

#pragma once

#include <cstdint>

#include "api/compute/eltwise_binary.h"

namespace tile_src {

#ifdef TRISC_UNPACK
ALWI void unpack_init() { TTI_SETADCXX(p_setadc::UNP_AB, 4 * FACE_R_DIM * FACE_C_DIM - 1, 0x0); }

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
    TTI_UNPACR(SrcA, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

// srcB only; the srcA bank of the previous unpack_tile stays valid for math.
ALWI void unpack_tile_b(uint32_t cb_b, uint32_t tile_b) {
    const uint32_t id_b = get_operand_id(cb_b);
    const uint32_t address_b =
        get_local_cb_interface(id_b).fifo_rd_ptr - 1 + get_local_cb_interface(id_b).fifo_page_size * tile_b;
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    } else {
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    TTI_UNPACR(SrcB, 0, 0, 0, 0, 1 /*OvrdThreadId*/, 1 /*Dvalid*/, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
// One MOP per tile: outer loop = fidelity phases (LoFi: one), inner loop = the 8 row chunks of the tile.
ALWI void math_init() {
    constexpr auto fidelity = static_cast<MathFidelity>(MATH_FIDELITY);
    constexpr uint32_t phases = is_high_fidelity(fidelity) ? to_underlying(fidelity) : 1;
    constexpr uint8_t fidelity_incr = is_high_fidelity(fidelity) ? 1 : 0;
    addr_mod_t{.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 8}}.set(ADDR_MOD_0);
    // End of a phase: back to tile row 0, next phase.
    addr_mod_t{.srca = {.clr = 1}, .srcb = {.clr = 1}, .dest = {.clr = 1}, .fidelity = {.incr = fidelity_incr}}.set(
        ADDR_MOD_2);
    addr_mod_t{.srca = {.clr = 1}, .srcb = {.clr = 1}, .dest = {.clr = 1}, .fidelity = {.clr = 1}}.set(ADDR_MOD_3);
    ckernel_template tmp(phases, 8, TT_OP_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0));
    tmp.set_last_inner_loop_instr(TT_OP_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0));
    // The product's last ELWMUL releases srcB only; math_release_a() releases srcA after its second product.
    tmp.set_last_outer_loop_instr(TT_OP_ELWMUL(p_setrwc::CLR_B, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0));
    tmp.program();
    ckernel::math::reset_counters(p_setrwc::SET_ABD_F);
}

ALWI void math_tile(uint32_t dst_index) {
    ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    ckernel_template::run();
    ckernel::math::clear_dst_reg_addr();
}

ALWI void math_release_a() { TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, 0); }
#endif

ALWI void tile_src_mul_init() {
    UNPACK((unpack_init()));
    MATH((math_init()));
}

// DEST[dst_a] += A[tile_a] * B[tile_b0]; DEST[dst_b] += A[tile_a] * B[tile_b1] (ELWMUL accumulates; a freshly
// acquired DEST half is zero). A is unpacked once.
ALWI void tile_src_mul_pair(
    uint32_t cb_a, uint32_t cb_b, uint32_t tile_a, uint32_t tile_b0, uint32_t tile_b1, uint32_t dst_a, uint32_t dst_b) {
    UNPACK((unpack_tile(cb_a, tile_a, cb_b, tile_b0)));
    MATH((math_tile(dst_a)));
    UNPACK((unpack_tile_b(cb_b, tile_b1)));
    MATH((math_tile(dst_b)));
    MATH((math_release_a()));
}

}  // namespace tile_src
