// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Blackhole fast tilize with a lean pack stream. RAW LLK: bypasses compute_kernel_lib::tilize (tilize_helpers.inl ->
// fast_tilize_block) and the pack half of the BH fast-tilize LLK (llk_pack_fast_tilize.h), on a measured win.
//
// The BH fast tilize (bf16 row-major -> bf8_b / bf16 tiles, fp32 DEST) is PACK-bound. Its pack MOP runs, per tile, the
// 16-PACR tile replay plus
//   start_op  ADDRCRZW (restore Z/W from the CR shadows)
//   end_ops   ADDDMAREG(OUTPUT_ADDR += tile); STALLWAIT(CFG, THCON); WRCFG(L1_Dest_addr); NOP
// The STALLWAIT waits on the ADDDMAREG issued right before it, so every tile's address update sits on the packer's
// critical path. Here, with the same PACRs, counters and addresses (output bit-identical):
//   * the update replay is STALLWAIT; WRCFG; ADDDMAREG: the GPR moves to the NEXT tile's address after the WRCFG, and
//     row_begin pre-increments it once, so each WRCFG reads a GPR written a whole tile (16 PACRs) earlier;
//   * no MOP start_op: every tile already ends with Z cleared (ADDR_MOD_1) and W == W_Cr (ADDRCRZW), so the restore is
//     a no-op once each chunk starts from an explicit SETADCZW reset of Z and W (the stock row_chunk resets W only);
//   * the MOP is programmed once per call for ONE tile and run `chunk` times per chunk, so a chunk change never
//     reprograms it.
// A bf8_b tile needs its own Last PACR, so the per-tile address update stays. Measured (Blackhole p150b): 1.06-1.13x
// per tile in isolation for tw 2..16; aspp.convs.1 @ LoFi compute_tilize 12.1 -> 11.0 us.
// The same change belongs in llk_pack_fast_tilize.h (replay order, row_begin pre-increment, no start_op), where every
// BH fast tilize would get it.
//
// Exit state: as the stock fast tilize (X = Y = Z = 0, W advanced); fast_tilize_uninit restores the default pack MOP.
// The replay entries [16..18] are only valid until the next pack init that loads its own.

#pragma once

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

namespace conv2d_3x3::lean_tilize {

#if defined(TRISC_PACK) && defined(ARCH_BLACKHOLE)
constexpr uint32_t kUpdateLen = 3;

// After fast_tilize_init (which loads the tile replay [0..15]): the hoisted address update and the 1-tile MOP.
inline void pack_init() {
    ckernel::load_replay_buf(REPLAY_ADDR_UPDATE_OFFSET, kUpdateLen, [] {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    });
    ckernel::ckernel_template tmp(
        1,
        1,
        lltt::replay_insn(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN),
        TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_inner_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_outer_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_end_ops(lltt::replay_insn(REPLAY_ADDR_UPDATE_OFFSET, kUpdateLen), TT_OP_NOP);
    tmp.program();
}

// Once per row block: program the address of output tile `tile` (of the reserved block) and move the GPR to the next.
inline void pack_row_begin(uint32_t cb_out, uint32_t tile) {
    llk_pack_fast_tilize_row_begin(cb_out, tile);
    TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
}

// One DEST section of `chunk` tiles: Z/W from 0, one 1-tile MOP run per tile.
inline void pack_row_chunk(uint32_t chunk) {
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    for (uint32_t i = 0; i < chunk; ++i) {
        ckernel::ckernel_template::run();
    }
}
#endif

// fast_tilize_block's split: 2..4 tiles per DEST section, 5 -> 2 + 3.
constexpr uint32_t chunk_of(uint32_t remaining) { return (remaining > 5) ? 4 : (remaining == 5) ? 2 : remaining; }

// Where ckl::tilize takes the BH fast tilize (its can_use_fast_tilize for this op's formats: bf16 in, bf16 / bf8_b
// out).
template <uint32_t tw>
constexpr bool usable() {
#ifdef ARCH_BLACKHOLE
    return tw >= 2 && tw < 256 && !compute_kernel_lib::get_dst_full_sync_enabled();
#else
    return false;
#endif
}

// compute_kernel_lib::tilize's contract (WaitBlock; init / uninit per `mode`; reconfig = UnpackAndPackReconfigure or
// NoReconfigure), same tiles in the same order; falls back to it where the fast tilize does not apply.
template <
    uint32_t tw,
    uint32_t cb_in,
    uint32_t cb_out,
    compute_kernel_lib::tilize_config::InitUninitMode mode =
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
    bool reconfig = true>
ALWI void tilize(DataflowBuffer& in, DataflowBuffer& out, uint32_t rows) {
    namespace tc = compute_kernel_lib::tilize_config;
    if constexpr (!usable<tw>()) {
        compute_kernel_lib::tilize<
            tw,
            cb_in,
            cb_out,
            mode,
            tc::WaitMode::WaitBlock,
            reconfig ? tc::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure
                     : tc::ReconfigureRegisterDatatypeMode::NoReconfigure>(in, out, rows);
    } else {
#ifdef ARCH_BLACKHOLE
        if constexpr (reconfig) {
            reconfig_data_format_srca(cb_in);
            pack_reconfig_data_format(cb_out);
        }
        if constexpr (mode == tc::InitUninitMode::InitAndUninit || mode == tc::InitUninitMode::InitOnly) {
            fast_tilize_init(cb_in, tw, cb_out);
            PACK((pack_init()));
        }
        // The unpack x-dim is one SETADCXX: set it on the first chunk of every call (no state across calls).
        [[maybe_unused]] uint32_t prev_chunk = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            in.wait_front(tw);
            out.reserve_back(tw);
            PACK((pack_row_begin(cb_out, 0)));
            for (uint32_t done = 0; done < tw;) {
                const uint32_t chunk = chunk_of(tw - done);
                MATH((llk_math_wait_for_dest_available()));
                PACK((llk_packer_wait_for_math_done()));
                if (chunk != prev_chunk) {
                    UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                    prev_chunk = chunk;
                }
                UNPACK((llk_unpack_fast_tilize_block(cb_in, 0, chunk, done)));
                MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, cb_in, 4)));
                PACK((pack_row_chunk(chunk)));
                MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                done += chunk;
            }
            out.push_back(tw);
            in.pop_front(tw);
        }
        if constexpr (mode == tc::InitUninitMode::InitAndUninit || mode == tc::InitUninitMode::UninitOnly) {
            fast_tilize_uninit(cb_in, cb_out, tw);
        }
#endif
    }
}

}  // namespace conv2d_3x3::lean_tilize
