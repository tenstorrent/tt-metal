// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Isolated BH fast-tilize bench (compute only, one core): bf16 row-major tap blocks [32, tw * 32] -> bf8_b tiles,
// fp32 DEST, half sync (conv2d_3x3's N-slice prelude contract). `blocks` tap blocks per call, read round-robin from
// `nb` resident source blocks (the op tilizes from published L1 addresses); `reps` calls (init .. uninit each).
//
// split 0 "helper": fast_tilize_block(icb, tw, ocb, tile_index) per tap block, i.e. ckl::tilize's inner loop. Its
//                   chunk state resets per call, so it reprograms the pack MOP at every block entry and on every
//                   change.
// split 1 "base":   conv2d_3x3 direct_tilize: fast_tilize_block's split (5 -> 2 + 3, else 4s then the rest), the chunk
//                   state persists across blocks, the pack MOP is reprogrammed whenever the chunk changes (tw 6: 4, 2,
//                   4, 2 ... = every chunk).
// split 2 "balanced": ceil(tw / 4) chunks of near-equal size, larger first (6 -> 3 + 3, 9 -> 3 + 3 + 3, 10 -> 4 + 3
//                   + 3, 5 -> 3 + 2, 7 -> 4 + 3): the same DEST-section count as base, the per-section pack load even.
// mop1: the pack MOP is programmed ONCE with outerloop 1 (one tile) and run `chunk` times per chunk, so a chunk
//       change never reprograms it (no mop_sync drain of the pack thread). Same PACR stream, same counters: one MOP
//       run of n outer iterations == n runs of a 1-outer-iteration MOP (start_op restores W from W_Cr, which each
//       tile's ADDRCRZW advanced). The unpack x-dim change stays (SETADCXX: one instruction, no sync).
// pack mode (CT arg 1): 0 stock (fast-tilize MOP, reprogrammed per chunk change), 1 mop1 (above),
//       2 stream: the chunk is ONE packer stream: L1 address programmed once per chunk, the per-tile replay issued
//         directly (15 PACRs + the 16th with Last only on the chunk's final tile), no per-tile address update. Correct
//         only if the BH packer lays BFP tiles out back to back without a per-tile Last (the bit-exact gate decides).
//       3 diag noaddr: mop1 without the per-tile L1 address update (all tiles of a chunk to one address; wrong output).
//       4 hoist (+ mop1): the per-tile address update replay [16..19] becomes STALLWAIT(CFG, THCON); WRCFG; NOP;
//         ADDDMAREG(next tile) and row_begin pre-increments the GPR once, so each tile's WRCFG reads a GPR written a
//         whole tile (16 PACRs) earlier and its STALLWAIT finds THCON idle. Same addresses, same order (bit-identical).
//       5 hoist with the stock per-change MOP reprogram (isolates the hoist from mop1).
//       6 hoist3 (+ mop1): the update replay without the NOP: STALLWAIT; WRCFG; ADDDMAREG (2 instructions between the
//         WRCFG and the next PACR, as program_packer_destination's WRCFG; SETDMAREG; DMANOP).
//       7 diag hoist_nostall: WRCFG; NOP; ADDDMAREG without the STALLWAIT (a bound only: the GPR producer is then
//         ordered by distance, not by the ISA).
//       8 lean (+ mop1): hoist3 and no MOP start_op (the per-tile Z/W restore from the CR shadows). Every tile already
//         ends with Z cleared (ADDR_MOD_1) and W == W_Cr (ADDRCRZW), so the restore is a no-op once each chunk starts
//         from an explicit SETADCZW reset of Z AND W (the stock row_chunk resets W only and relies on the restore).
// diag 1 NoPack: unpack + math only (the packer still hands DEST back). diag 2 PackOnly: pack only (unpack idle, math
//       only syncs DEST). Both timed only (the output is wrong by construction).
//
// Raw LLK (bench): the variants are fast_tilize_block's loop with the unpacker base taken from an address (as
// conv2d_3x3's direct_tilize / window_tilize) and, for mop1, the pack row chunk issued as per-tile MOP runs.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"

namespace {

constexpr uint32_t CB_IN = 0;
constexpr uint32_t CB_OUT = 16;

#ifdef TRISC_UNPACK
inline void unpack_chunk_at(uint32_t src, uint32_t chunk, uint32_t col_start) {
    const uint32_t operand_id = get_operand_id(CB_IN);
    const uint32_t src_format = unpack_src_format[operand_id];
    const uint32_t base = (src >> 4) - 1 + (SCALE_DATUM_SIZE(src_format, col_start * TILE_C_DIM) >> 4);
    _llk_unpack_fast_tilize_block_(base, 0, src_format, chunk, get_operand_num_faces(operand_id));
}
#endif

#ifdef TRISC_PACK
// One pack row chunk as `chunk` runs of the 1-tile MOP (W reset once per chunk, as _llk_pack_fast_tilize_row_chunk_).
inline void pack_row_chunk_mop1(uint32_t chunk) {
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    for (uint32_t i = 0; i < chunk; ++i) {
        ckernel::ckernel_template::run();
    }
}

// diag noaddr: the 1-tile fast-tilize MOP without its end_ops address update.
inline void pack_mop1_noaddr_config() {
    ckernel::ckernel_template tmp(
        1,
        1,
        lltt::replay_insn(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN),
        TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011));
    tmp.set_last_inner_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_outer_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_end_ops(TT_OP_NOP, TT_OP_NOP);
    tmp.program();
}

// hoist: reload the address-update replay with the GPR increment after the WRCFG (for the NEXT tile).
inline void pack_hoist_replay() {
    ckernel::load_replay_buf(REPLAY_ADDR_UPDATE_OFFSET, REPLAY_ADDR_UPDATE_LEN, [] {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        TTI_NOP;
        TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    });
}

// 1-tile fast-tilize MOP with a custom end-op replay length (hoist3 / hoist_nostall: 3 entries at [16..18]).
inline void pack_mop1_config_end3() {
    ckernel::ckernel_template tmp(
        1,
        1,
        lltt::replay_insn(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN),
        TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_start_op(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011));
    tmp.set_last_inner_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_outer_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_end_ops(lltt::replay_insn(REPLAY_ADDR_UPDATE_OFFSET, 3), TT_OP_NOP);
    tmp.program();
}

inline void pack_mop1_config_lean() {
    ckernel::ckernel_template tmp(
        1,
        1,
        lltt::replay_insn(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN),
        TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_inner_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_last_outer_loop_instr(TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010));
    tmp.set_end_ops(lltt::replay_insn(REPLAY_ADDR_UPDATE_OFFSET, 3), TT_OP_NOP);
    tmp.program();
}

inline void pack_row_chunk_lean(uint32_t chunk) {
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    for (uint32_t i = 0; i < chunk; ++i) {
        ckernel::ckernel_template::run();
    }
}

inline void pack_hoist3_replay() {
    ckernel::load_replay_buf(REPLAY_ADDR_UPDATE_OFFSET, 3, [] {
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
        TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    });
}

inline void pack_hoist_nostall_replay() {
    ckernel::load_replay_buf(REPLAY_ADDR_UPDATE_OFFSET, 3, [] {
        TTI_WRCFG(p_gpr_pack::OUTPUT_ADDR, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32);
        TTI_NOP;
        TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
    });
}

// hoist: row_begin programs tile `tile`'s address, then moves the GPR to the next tile's.
inline void pack_row_begin_hoist(uint32_t cb_out, uint32_t tile) {
    llk_pack_fast_tilize_row_begin(cb_out, tile);
    TTI_ADDDMAREG(0, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET);
}

// stream: the 16th PACR of a tile (the replay's last entry), Last as given.
template <uint32_t last>
inline void pack_tile_close() {
    TTI_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        ADDR_MOD_1,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        p_pacr::ALL_INTF_ACTIVE,
        0,
        0,
        p_pacr::NO_CTXT_CTRL,
        0,
        last);
}

// stream: one chunk = one packer stream starting at output tile `first_tile` of the reserved block.
inline void pack_row_chunk_stream(uint32_t cb_out, uint32_t first_tile, uint32_t chunk) {
    llk_pack_fast_tilize_row_begin(cb_out, first_tile);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0010);
    for (uint32_t i = 0; i < chunk; ++i) {
        TTI_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
        lltt::replay(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN - 1);
        if (i + 1 == chunk) {
            pack_tile_close<1>();
        } else {
            pack_tile_close<0>();
        }
        TTI_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010);
    }
}
#endif

template <uint32_t split, uint32_t tw>
constexpr uint32_t chunk_at(uint32_t index, uint32_t done) {
    if constexpr (split == 2) {
        constexpr uint32_t n = (tw + 3) / 4;
        return tw / n + (index < tw % n ? 1 : 0);
    } else {
        const uint32_t remaining = tw - done;
        return (remaining > 5) ? 4 : (remaining == 5) ? 2 : remaining;
    }
}

template <uint32_t split, uint32_t pack_mode, uint32_t diag, uint32_t tw, uint32_t blocks, uint32_t nb>
ALWI void tilize_call() {
    reconfig_data_format_srca(CB_IN);
    pack_reconfig_data_format(CB_OUT);
    fast_tilize_init(CB_IN, tw, CB_OUT);
    if constexpr (split == 0) {
        for (uint32_t k = 0; k < blocks; ++k) {
            cb_reserve_back(CB_OUT, tw);
            fast_tilize_block(CB_IN, tw, CB_OUT, (k % nb) * tw);
            cb_push_back(CB_OUT, tw);
        }
    } else {
        constexpr bool mop1 = pack_mode != 0 && pack_mode != 5;
        constexpr bool hoist = pack_mode >= 4;
        if constexpr (pack_mode == 1 || pack_mode == 4) {
            PACK((_llk_pack_fast_tilize_mop_config_(1)));
        } else if constexpr (pack_mode == 3) {
            PACK((pack_mop1_noaddr_config()));
        } else if constexpr (pack_mode == 6 || pack_mode == 7) {
            PACK((pack_mop1_config_end3()));
        } else if constexpr (pack_mode == 8) {
            PACK((pack_mop1_config_lean()));
        }
        if constexpr (pack_mode == 4 || pack_mode == 5) {
            PACK((pack_hoist_replay()));
        } else if constexpr (pack_mode == 6 || pack_mode == 8) {
            PACK((pack_hoist3_replay()));
        } else if constexpr (pack_mode == 7) {
            PACK((pack_hoist_nostall_replay()));
        }
        [[maybe_unused]] uint32_t base_addr = 0;
        UNPACK(base_addr = get_local_cb_interface(get_operand_id(CB_IN)).fifo_rd_ptr << 4);
        uint32_t prev_chunk = 0;
        for (uint32_t k = 0; k < blocks; ++k) {
            [[maybe_unused]] const uint32_t src = base_addr + (k % nb) * (tw * 32 * 32 * 2);
            cb_reserve_back(CB_OUT, tw);
            if constexpr (hoist) {
                PACK((pack_row_begin_hoist(CB_OUT, 0)));
            } else if constexpr (pack_mode != 2) {
                PACK((llk_pack_fast_tilize_row_begin(CB_OUT, 0)));
            }
            uint32_t done = 0;
            for (uint32_t c = 0; done < tw; ++c) {
                const uint32_t chunk = chunk_at<split, tw>(c, done);
                MATH((llk_math_wait_for_dest_available()));
                PACK((llk_packer_wait_for_math_done()));
                if (chunk != prev_chunk) {
                    UNPACK((llk_unpack_fast_tilize_reinit_xdim(chunk)));
                    if constexpr (!mop1) {
                        PACK((llk_pack_fast_tilize_reinit_unit_dim(CB_OUT, chunk)));
                    }
                    prev_chunk = chunk;
                }
                if constexpr (diag != 2) {
                    UNPACK((unpack_chunk_at(src, chunk, done)));
                    MATH((llk_math_fast_tilize_block_<DST_ACCUM_MODE>(0, CB_IN, 4)));
                }
                if constexpr (diag != 1) {
                    if constexpr (pack_mode == 2) {
                        PACK((pack_row_chunk_stream(CB_OUT, done, chunk)));
                    } else if constexpr (pack_mode == 8) {
                        PACK((pack_row_chunk_lean(chunk)));
                    } else if constexpr (mop1) {
                        PACK((pack_row_chunk_mop1(chunk)));
                    } else {
                        PACK((llk_pack_fast_tilize_row_chunk(0, CB_OUT, chunk)));
                    }
                }
                MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
                PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
                done += chunk;
            }
            PACK((llk_pack_fast_tilize_row_end()));
            cb_push_back(CB_OUT, tw);
        }
    }
    fast_tilize_uninit(CB_IN, CB_OUT, tw);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t split = get_compile_time_arg_val(0);
    constexpr uint32_t pack_mode = get_compile_time_arg_val(1);
    constexpr uint32_t diag = get_compile_time_arg_val(2);
    constexpr uint32_t tw = get_compile_time_arg_val(3);
    constexpr uint32_t blocks = get_compile_time_arg_val(4);
    constexpr uint32_t nb = get_compile_time_arg_val(5);
    constexpr uint32_t reps = get_compile_time_arg_val(6);
    static_assert(tw >= 2, "tw 1 takes the per-tile tilize path");

    compute_kernel_hw_startup(CB_IN, CB_OUT);
    for (uint32_t rep = 0; rep < reps; ++rep) {
        tilize_call<split, pack_mode, diag, tw, blocks, nb>();
        cb_wait_front(CB_OUT, blocks * tw);
        cb_pop_front(CB_OUT, blocks * tw);
    }
}
