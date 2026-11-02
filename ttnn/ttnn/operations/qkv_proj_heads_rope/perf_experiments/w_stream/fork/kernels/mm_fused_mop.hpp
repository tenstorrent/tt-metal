// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Perf 2 (raw LLK, measured): one math MOP per matmul_block call instead of one per tile product (full 32x32
// tiles, no transpose).
//
// Bypasses ckernel::matmul_block's math half (llk_math_matmul), which issues per tile product a SETC16 of the
// DEST target offset plus one MOP run. Measured on p100a (8x512, compute isolated): that per-tile issue costs
// ~4.4 cycles on top of 16 cycles per fidelity phase, independent of fidelity (52.4 -> 49.3 cycles per
// product at HiFi3; whole op 94.9 -> 92.1 us). The MOP's outer loop walks the call's streamed tiles; DEST is
// addressed by the RWC counter against its carriage-return base: ADDR_MOD_5 phase end (dest = cr),
// ADDR_MOD_6 tile end (cr += one tile, the last MVMUL releases the streamed src bank, fidelity reset),
// ADDR_MOD_7 call end (everything cleared, CLR_AB releases both src banks). The DEST target offset is written
// only when it changes (successive K steps of one subblock hit the same tiles). Same MVMUL sequence, fidelity
// phases and per-element accumulation order as llk_math_matmul: bit-identical (HiFi4 / HiFi3 / HiFi2 / LoFi,
// half and full sync, sb 1xN and 2x2). The unpack half is unchanged (llk_unpack_AB_matmul).
//
// Call order: ckernel::matmul_block_init(in0, in1, 0, ct_dim, rt_dim, kt_dim), then fused_init(ct_dim, rt_dim),
// then fused_matmul_block() per K step. Any init in between (eltwise / copy / bcast) reprograms ADDR_MOD_5..7
// and the math MOP, so fused_init must follow the matmul init again.
// Blackhole dynamic throttle: when the firmware throttle flag is set, falls back to the stock throttled path.

#pragma once

#include <cstdint>

#include "api/compute/matmul.h"

namespace mmk {

#ifdef TRISC_MATH
inline bool fused_active = false;
inline uint32_t fused_dst_addr = 0xffffffff;  // last DEST target offset written

ALWI void fused_mop_config(uint32_t ct_dim, uint32_t rt_dim) {
    constexpr bool high_fidelity = ckernel::math::is_high_fidelity(MATH_FIDELITY);
    constexpr uint32_t phases = high_fidelity ? static_cast<uint32_t>(MATH_FIDELITY) : 1;
    const bool reuse_a = ct_dim >= rt_dim;
    const uint32_t rut_dim = reuse_a ? ct_dim : rt_dim;
    const uint32_t tile_stride = (reuse_a ? 1 : ct_dim) << 6;  // DEST rows between consecutive streamed tiles

    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = static_cast<uint8_t>(high_fidelity ? 1 : 0), .clr = 0},
    }
        .set(ADDR_MOD_5);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = static_cast<int16_t>(tile_stride), .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_6);
    addr_mod_t{
        .srca = {.incr = 0, .clr = 1, .cr = 1},
        .srcb = {.incr = 0, .clr = 1, .cr = 1},
        .dest = {.incr = 0, .clr = 1, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_7);

    // The stock init left the full-tile MVMUL sequence at replay_buf_offset; its first 15 entries are the
    // fidelity-independent body, the 16th (phase end) is replaced by the loop / last-iteration MVMULs.
    ckernel_template tmp(
        rut_dim,
        phases,
        lltt::replay_insn(ckernel::math::replay_buf_offset, 15),
        TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_5, 0));
    const uint32_t release = reuse_a ? p_setrwc::CLR_A : p_setrwc::CLR_B;
    tmp.set_last_inner_loop_instr(TT_OP_MVMUL(release, 0, ADDR_MOD_6, 0));
    tmp.set_last_outer_loop_instr(TT_OP_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_7, 0));
    tmp.program();
    math::reset_counters(p_setrwc::SET_ABD_F);
    fused_dst_addr = 0xffffffff;
}

ALWI void fused_math(uint32_t dst_index, uint32_t ct_dim, uint32_t rt_dim) {
    const bool reuse_a = ct_dim >= rt_dim;
    const uint32_t t_dim = reuse_a ? rt_dim : ct_dim;
    for (uint32_t t = 0; t < t_dim; ++t) {
        const uint32_t addr = ((dst_index + (reuse_a ? ct_dim * t : t)) << 6) + get_dest_buffer_base();
        if (addr != fused_dst_addr) {
            TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, addr);
            fused_dst_addr = addr;
        }
        ckernel_template::run();
    }
}
#endif

// After ckernel::matmul_block_init with the same ct_dim / rt_dim.
ALWI void fused_init(uint32_t ct_dim, uint32_t rt_dim) {
    MATH((fused_active = true));
    MATH((fused_mop_config(ct_dim, rt_dim)));
}

ALWI void fused_matmul_block(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t in0_tile_index,
    uint32_t in1_tile_index,
    uint32_t idst,
    uint32_t ct_dim,
    uint32_t rt_dim,
    uint32_t kt_dim) {
    UNPACK((llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
#ifdef TRISC_MATH
#ifdef ARCH_BLACKHOLE
    if (*(throttle_ptr) % 2) {
        if (fused_active) {
            fused_active = false;
            throttled_mop_status = 0;  // force the stock path to re-init its (throttled) MOP
        }
        matmul_block_math_dynamic_throttle(in0_cb_id, in1_cb_id, idst, 0, ct_dim, rt_dim);
        return;
    }
    if (!fused_active) {
        llk_math_matmul_init<MATH_FIDELITY, MM_THROTTLE>(in0_cb_id, in1_cb_id, 0, ct_dim, rt_dim);
        fused_mop_config(ct_dim, rt_dim);
        fused_active = true;
    }
#endif
    fused_math(idst, ct_dim, rt_dim);
#endif
}

}  // namespace mmk
