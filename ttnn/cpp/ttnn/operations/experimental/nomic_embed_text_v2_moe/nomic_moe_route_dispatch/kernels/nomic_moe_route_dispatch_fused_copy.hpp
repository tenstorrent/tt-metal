// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The G contributor faces of one split-softmax pass unpacked to DEST (fp32, unpack to
// dest) under ONE math <-> unpack handshake, into face 0 of DEST tiles 0 .. G-1 (same placement as G copy_tile calls
// through the 16x16-tile view). Raw LLK, bypassing copy_tile: each copy_tile is a full math <-> unpack round trip
// (mailbox dest address, MATH_DONE / UNPACK_TO_DEST semaphores, config restore), ~90 cycles a 1 KB face on MATH;
// here the unpacker runs the G one-face MOPs back to back (alternating its two config contexts, the dest address
// set per context) and posts UNPACK_TO_DEST once. Perf R4.2 (perf_experiments/r42_softmax_pass): 8x512 pass
// 1027 -> 915 cycles over G copy_tile calls of the same faces, bit-identical. Must end by re-enabling the SrcA
// address swizzle, as the stock unpack_to_dest_tile_done does: without it the later untilize corrupts every row.

#pragma once

#include <cstdint>

namespace route_dispatch_fused_copy {

#ifdef TRISC_UNPACK
// operand: the 16x16-tile view; face tiles first_tile + c * tile_stride, c < G (never popped: rd_ptr = base).
template <uint32_t G>
inline void unpack_contributors(uint32_t operand, uint32_t first_tile, uint32_t tile_stride) {
    using namespace ckernel;
    using namespace ckernel::unpacker;
    const uint32_t operand_id = get_operand_id(operand);
    const uint32_t base = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    const uint32_t page = get_local_cb_interface(operand_id).fifo_page_size;
    const uint32_t dst_format = unpack_dst_format[operand_id];
    const uint32_t z_stride = canonical_unpA_z_stride(dst_format) << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT;
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    const uint32_t dst_row = mailbox_read(ThreadId::MathThreadId);
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0);
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride), 0, LO_16(p_gpr_unpack::TMP_LO));
    TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
#pragma GCC unroll 3
    for (uint32_t c = 0; c < G; ++c) {
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        wait_for_next_context(2);
        const uint32_t dst_byte_addr = 16 * (4 + dst_row + 64 * c);
        if (unp_cfg_context == 0) {
            cfg[THCON_SEC0_REG3_Base_address_ADDR32] = base + page * (first_tile + c * tile_stride);
        } else {
            cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = base + page * (first_tile + c * tile_stride);
        }
        semaphore_post(semaphore::UNPACK_SYNC);
        if (unp_cfg_context == 0) {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);
        } else {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dst_byte_addr);
        }
        if (c == 0) {
            wait_for_dest_available();
        }
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
        ckernel_template::run();
        t6_semaphore_get(semaphore::UNPACK_SYNC);
        // restore this context (unpack to SrcA, canonical dest address) before the switch, as the stock tile_done
        if (c == G - 1) {
            t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
            TT_SETDMAREG(0, LOWER_HALFWORD(z_stride), 0, LO_16(p_gpr_unpack::TMP_LO));
            TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
        } else {
            // the context restore below must not land while this face's UNPACR still reads the context
            TTI_STALLWAIT(p_stall::STALL_THREAD, p_stall::UNPACK0);
        }
        if (unp_cfg_context == 0) {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(0);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(4 * 16);
        } else {
            cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(0);
            cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(4 * 16);
        }
        switch_config_context(unp_cfg_context);
    }
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);  // re-enable the SrcA address bit swizzle (stock tile_done)
}
#endif

#ifdef TRISC_MATH
// Math side of unpack_contributors, split so the next pass's unpack overlaps this pass's SFPU band (Perf R4.3,
// perf_experiments/r43_softmax_overlap: 8x512 cmp_softmax 3290 -> 2990 cycles, bit-identical). The stock
// math_unpack_to_dest_math_ready + set_dst_write_addr pair drains MATH / SFPU and targets only the current DEST half,
// so the unpacker could start a pass only after the previous band ended.
//   first pass:  contributors_send (drain, this half's dest address to the unpacker)
//   every pass:  contributors_wait, then (if a pass follows) pass_marker; send_next_half mid-band
// The mailbox carries the dest address the unpacker reads (unpack_contributors).

inline void contributors_send() {
    using namespace ckernel;
    math::math_unpack_to_dest_math_ready();
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(0);
}

// The UNPACK_TO_DEST token, then the BH zero-flag re-clear (budabackend#2730) of the faces written (face 0 of DEST
// tiles 0 .. G-1).
template <uint32_t G>
inline void contributors_wait() {
    using namespace ckernel;
    math::math_unpack_to_dest_tile_ready();
#pragma GCC unroll 3
    for (uint32_t c = 0; c < G; ++c) {
        TT_ZEROACC(p_zeroacc::CLR_16, 1, 1 /*clear zero flags*/, ADDR_MOD_3, math::get_dest_index_in_faces(c, 0));
    }
}

// Marker behind the previous commit's MATH_PACK post (the math thread is in order): once the RISC sees it, MATH_PACK
// counts only the previous pass's pack still holding the other half.
inline void pass_marker() { ckernel::t6_semaphore_post<ckernel::p_stall::NONE>(ckernel::semaphore::MATH_DONE); }

// The next pass's dest address (tile 0 of the other DEST half) to the unpacker, once the marker is seen and the packer
// has released every committed half.
inline void send_next_half() {
    using namespace ckernel;
    while (semaphore_read(semaphore::MATH_DONE) == 0) {
    }
    semaphore_get(semaphore::MATH_DONE);
    while (semaphore_read(semaphore::MATH_PACK) > 0) {
    }
    mailbox_write(ThreadId::UnpackThreadId, DEST_REGISTER_HALF_SIZE - get_dest_buffer_base());
}
#endif

}  // namespace route_dispatch_fused_copy
