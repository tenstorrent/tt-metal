// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Wide (>= 2 tiles) untilize of bf8_b / bf16 TILE tile-rows into a bf16 row-major slab, fp32 DEST half-sync, Blackhole.
// RAW LLK: bypasses compute_kernel_lib::untilize<ct> (its BH fast_untilize path, untilize_helpers.inl).
//
// The helper's fast_untilize splits each tile-row into 4-tile chunks (one DEST section each). For ct > 4 every chunk
// is packed row-strided: each of its 32 output rows is its own L1 stream closed by a Last=1 PACR, which drains the
// packer. So a ct = 8 tile-row costs 64 packer drains and 2 section handshakes per 8 tiles, and the pack is the
// bottleneck (isolated: pack ~34 ns/tile vs unpack + math ~20 ns/tile).
//
// Here (same unpack MOP instructions, same ELWADD fp32 copy, same packer formats and PACR shapes: output
// bit-identical):
//   * a DEST section holds 16 faces as "units": one unit = faces (2h, 2h+1) of `cw` consecutive tiles of a tile-row,
//     i.e. slab rows 16h .. 16h+15 over those tiles. The math places each face with the helper's own fp32 face copy
//     (ELWADD + zero SrcB, DEST row immediates), unit u at DEST row 32 * cw * u, tile t of it at + 32 t.
//   * ct 5..8 (half-row mode): one half tile-row per section (unpack visits each tile twice: faces 0-1, then 2-3).
//     A half tile-row is 16 consecutive slab rows, so the section is ONE contiguous L1 stream: the L1 address is
//     programmed once and only its last PACR closes the stream (one packer drain per section, not 32).
//   * ct <= 4 (whole-row mode): 4 / ct whole tile-rows per section (tile-major unpack, 4 faces per tile); the units
//     (2 per tile-row) are consecutive slab rows again: one stream per section. ct = 2: one MOP walks all 4 units
//     (z_stride = 64 DEST rows = one ct-2 unit); ct 3, 4: one MOP run per unit, the last one closing.
//   * ct > 8 (chunk mode): the tile-row is split into balanced chunks of <= 8 tiles, one half chunk per section; each
//     slab row still closes (row-strided like the helper), but with up to 4 PACRs per drain, not 2.
//   * PACR shape per slab row of a unit: ceil(cw / 2) PACRs of 4 DEST interfaces (2 tiles), the last one 2 interfaces
//     for odd cw: the helper's own PACRs.
//
// What the helper cannot express: its unit of work is one tile-row (block_width_tiles) and fast_untilize's DEST section
// is up to 4 WHOLE tiles of that row; there is no half-tile-row section (faces 0-1, then 2-3, of up to 8 tiles), no
// multi-row section, and its CB sync is per tile-row.
//
// Measured (Blackhole p150b, 1 core, bf8_b -> bf16, fp32 DEST, Sync::Hybrid; helper -> this, ns/tile):
//   ct 2: 75.6 -> 44.9   ct 4: 42.1 -> 39.6   ct 5: 56.1 -> 36.8   ct 6: 43.7 -> 28.6   ct 7: 47.4 -> 34.8
//   ct 8: 36.2 -> 25.9   ct 9: 44.3 -> 41.0   ct 10: 44.4 -> 41.9  ct 16: 39.9 -> 33.2  ct 17: 52.2 -> 47.0
// ct 8 now sits near both hardware rates: pack-only 24.6, unpack + math only 25.1 (the helper's unpack + math: 22.5;
// the gap is the second tile visit). In conv2d_3x3: semantic res2.fuse_conv.1 @ LoFi 115.6 -> 113.8 us (untilize zone
// 14.5 -> 12.7 us; M-block 0 stays paced by the reader's fill); aspp.convs.1 flat (untilize fully fill-paced).
//
// Exit state: PAC ch0 X/Y/Z/W = 0 and ch1 Y = 0 (the BH tilize pack does not reset Y at tile start: a stale Y shifts
// the next tilize's face 0, see conv2d_3x3_narrow_untilize.hpp), DEST target offset at the section base, MATH
// ADDR_MOD_4/5 as the helper's fast_untilize init leaves them, pack ADDR_MOD_2 restored by the uninit's pack init
// (ct = 2 also sets pack ADDR_MOD_3, which only the fast tilize reads, and its init programs it).

#pragma once

#include <cstdint>

#include "api/compute/experimental/fast_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

namespace wide_untilize {

enum class Diag : uint32_t { None = 0, NoPack = 1, PackOnly = 2, UnpackOnly = 3 };
// CB sync: per section group; per untilize batch; input per batch + output reserved per batch, pushed per group
// (Hybrid); or Lazy: input waited per group (cumulative inside the batch) and popped per batch, output reserved per
// batch and pushed per group (the first section starts as soon as its rows land, as with Group).
enum class Sync : uint32_t { Group = 0, Batch = 1, Hybrid = 2, Lazy = 3 };

// A fp32 half-sync DEST section holds 4 tiles = 16 faces.
constexpr uint32_t SECTION_TILES = 4;
// Half tile-rows of up to 8 tiles fill one section.
constexpr uint32_t MAX_UNIT_TILES = 2 * SECTION_TILES;

template <uint32_t ct>
struct Plan {
    static_assert(ct >= 2, "1-wide slabs use the narrow path");
    static constexpr bool whole_rows = ct <= SECTION_TILES;
    static constexpr uint32_t rows_per_section = whole_rows ? SECTION_TILES / ct : 1;
    static constexpr uint32_t num_chunks = whole_rows ? 1 : (ct + MAX_UNIT_TILES - 1) / MAX_UNIT_TILES;
    static constexpr bool strided = num_chunks > 1;
    // Chunk widths: whole tile pairs spread evenly (a 4-interface PACR reads 2 tiles; an odd chunk ends each slab row
    // with a 2-interface PACR), the odd tile of an odd ct on the last chunk. All <= 8.
    static constexpr uint32_t pairs = ct / 2;
    static constexpr uint32_t chunk_w(uint32_t c) {
        return 2 * (pairs / num_chunks + (c < pairs % num_chunks ? 1 : 0)) + ((ct & 1) && c + 1 == num_chunks ? 1 : 0);
    }
    // The init's full_ct_dim: > 4 programs the helper's row-strided pack (ADDR_MOD_1 y_dst += 1, ch1 Y stride = one
    // slab row); <= 4 the contiguous one.
    static constexpr uint32_t init_ct = strided ? ct : 4;
    // Faces per tile the unpacker delivers in one tile visit.
    static constexpr uint32_t faces_per_visit = whole_rows ? 4 : 2;
    // Whole-row mode: DEST rows between units. The packer's strided 4-interface read only sees a unit placed at a
    // 64-row-aligned DEST offset (measured: ct = 3 units at 96 rows read the wrong tile pair), so round up.
    static constexpr uint32_t unit_stride = ((32 * ct + 63) / 64) * 64;
};

#ifdef TRISC_UNPACK
// faces real SrcA faces per tile visit, each with a zero SrcB dvalid (the fp32 ELWADD copy is binary).
template <uint32_t faces>
inline void unpack_mop_config() {
    static constexpr uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint32_t unpack_srcb_zero = TT_OP_UNPACR_NOP(
        SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    ckernel_template tmp(faces, 1, unpack_srca, unpack_srcb_zero);
    tmp.program();
}

// n tiles, the first at `address` (16 B units, unpacker -1 convention), `stride_16B` apart, each from face 2h
// (MOP: the configured faces per visit). The per-tile walk of _llk_unpack_fast_untilize_bfp_block_ (dense pages too).
inline void unpack_tiles(uint32_t address, uint32_t stride_16B, uint32_t n, uint32_t h) {
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    wait_for_next_context(2);
    const bool ctx0 = unp_cfg_context == 0;
    cfg[ctx0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    TT_SETDMAREG(0, LOWER_HALFWORD(stride_16B), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(stride_16B), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, SCRATCH_SEC0_val_ADDR32);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    for (uint32_t t = 0; t < n; ++t) {
        if (h == 0) {
            TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        } else {
            TTI_SETADCZW(0b011, 0, 0, 0, 2, 0b1111);
        }
        ckernel::ckernel_template::run();
        if (t + 1 < n) {
            if (ctx0) {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32);
            } else {
                TTI_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
            }
            TTI_NOP;
        }
    }
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}

// Folded variant: the whole per-tile walk is one MOP (outer = tiles; start op SETADCZW z = 2h; inner = the faces;
// end ops CFGSHIFTMASK base += stride, NOP), so the RISC issues one MOP per unit instead of 4 instructions per tile.
// The start op (h) and the end op (the active config context's base register) are patched per unit.
constexpr uint32_t MOP_START_OP_CFG_INDEX = 2;
constexpr uint32_t MOP_END_OP0_CFG_INDEX = 3;

template <uint32_t faces>
inline void unpack_mop_config_folded(uint32_t n_tiles) {
    static constexpr uint32_t unpack_srca =
        TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint32_t unpack_srcb_zero = TT_OP_UNPACR_NOP(
        SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, p_unpacr_nop::WAIT_LIKE_UNPACR, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    ckernel_template tmp(n_tiles, faces, unpack_srca, unpack_srcb_zero);
    tmp.set_start_op(TT_OP_SETADCZW(0b011, 0, 0, 0, 0, 0b1111));
    tmp.set_end_ops(TT_OP_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32), TT_OP_NOP);
    tmp.program();
}

inline void unpack_tiles_folded(uint32_t address, uint32_t stride_16B, uint32_t h) {
    volatile uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    volatile uint32_t* mop_cfg = reinterpret_cast<volatile uint32_t*>(TENSIX_MOP_CFG_BASE);
    wait_for_next_context(2);
    const bool ctx0 = unp_cfg_context == 0;
    cfg[ctx0 ? THCON_SEC0_REG3_Base_address_ADDR32 : THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    TT_SETDMAREG(0, LOWER_HALFWORD(stride_16B), 0, LO_16(p_gpr_unpack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(stride_16B), 0, HI_16(p_gpr_unpack::TMP0));
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_unpack::TMP0, 0, SCRATCH_SEC0_val_ADDR32);
    ckernel::mop_sync();
    mop_cfg[MOP_START_OP_CFG_INDEX] =
        h == 0 ? TT_OP_SETADCZW(0b011, 0, 0, 0, 0, 0b1111) : TT_OP_SETADCZW(0b011, 0, 0, 0, 2, 0b1111);
    mop_cfg[MOP_END_OP0_CFG_INDEX] =
        ctx0 ? TT_OP_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_address_ADDR32)
             : TT_OP_CFGSHIFTMASK(1, 0b011, 32 - 1, 0, 0b11, THCON_SEC0_REG3_Base_cntx1_address_ADDR32);
    semaphore_post(semaphore::UNPACK_SYNC);
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);
    ckernel::ckernel_template::run();
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    switch_config_context(unp_cfg_context);
}
#endif

#ifdef TRISC_MATH
// The helper's fp32 face copy (ELWADD SrcA + zero SrcB, ADDR_MOD_5: srca += 8) at a compile-time DEST row.
template <uint32_t row, Diag diag>
ALWI void face() {
    if constexpr (diag == Diag::UnpackOnly) {
        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
    } else {
        _llk_math_fast_untilize_copy_face_fp32_(row);
    }
}

// Half-row unit of n tiles: faces in arrival order to consecutive 16-row slots.
template <uint32_t f, uint32_t n_faces, Diag diag>
ALWI void unit_faces() {
    if constexpr (f < n_faces) {
        face<16 * f, diag>();
        unit_faces<f + 1, n_faces, diag>();
    }
}

// Whole-row mode, tile k = (row j, col t) of the section, 4 faces in arrival order: F0, F1 to unit (j, 0), F2, F3 to
// unit (j, 1).
template <uint32_t ct, uint32_t k, uint32_t n_tiles, Diag diag>
ALWI void whole_row_tiles() {
    if constexpr (k < n_tiles) {
        constexpr uint32_t j = k / ct;
        constexpr uint32_t t = k % ct;
        constexpr uint32_t top = Plan<ct>::unit_stride * (2 * j) + 32 * t;
        constexpr uint32_t bottom = Plan<ct>::unit_stride * (2 * j + 1) + 32 * t;
        face<top, diag>();
        face<top + 16, diag>();
        face<bottom, diag>();
        face<bottom + 16, diag>();
        whole_row_tiles<ct, k + 1, n_tiles, diag>();
    }
}

template <uint32_t ct, uint32_t g, Diag diag>
ALWI void whole_row_group(uint32_t rows) {
    if constexpr (g > 0) {
        if (rows == g) {
            whole_row_tiles<ct, 0, g * ct, diag>();
        } else {
            whole_row_group<ct, g - 1, diag>(rows);
        }
    }
}

ALWI void math_section_begin() {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}
#endif

#ifdef TRISC_PACK
constexpr uint32_t MOP_LAST_OUTER_CFG_INDEX = 7;

inline uint32_t pacr(uint32_t addr_mod, uint32_t intf, uint32_t concat, uint32_t last) {
    return TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        intf,
        0,
        concat,
        p_pacr::NO_CTXT_CTRL,
        0,
        last);
}

inline uint32_t tail_intf(uint32_t cw) { return (cw & 1) ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE; }

// One unit: 16 slab rows x ceil(cw / 2) PACRs. ADDR_MOD_0: z += 1 (next tile pair); ADDR_MOD_1: y += 1, z = 0
// (+ y_dst += 1 when strided, set by the init). Contiguous: only the last PACR may close (`close`); strided: every
// row closes, its other PACRs concatenate.
template <bool strided>
inline void pack_mop_config(uint32_t cw, bool close) {
    const uint32_t per_row = (cw + 1) / 2;
    ckernel_template tmp(16, per_row, pacr(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, strided ? 1 : 0, 0));
    tmp.set_last_inner_loop_instr(pacr(ADDR_MOD_1, tail_intf(cw), 0, strided ? 1 : 0));
    tmp.set_last_outer_loop_instr(pacr(ADDR_MOD_1, tail_intf(cw), 0, (strided || close) ? 1 : 0));
    tmp.program();
}

// ct = 2, whole rows: one MOP over `units` units of 16 rows, one 4-interface PACR per row. ADDR_MOD_2: y += 1;
// ADDR_MOD_3: y = 0, z += 1 (z_stride = 64 DEST rows = one unit). Only the last PACR closes.
inline void pack_mop_config_ct2(uint32_t units) {
    ckernel_template tmp(units, 16, pacr(ADDR_MOD_2, p_pacr::ALL_INTF_ACTIVE, 0, 0));
    tmp.set_last_inner_loop_instr(pacr(ADDR_MOD_3, p_pacr::ALL_INTF_ACTIVE, 0, 0));
    tmp.set_last_outer_loop_instr(pacr(ADDR_MOD_3, p_pacr::ALL_INTF_ACTIVE, 0, 1));
    tmp.program();
}

inline void pack_addrmod_ct2() {
    addr_mod_pack_t{.y_src = {.incr = 1}}.set(ADDR_MOD_2);
    addr_mod_pack_t{.y_src = {.incr = 0, .clr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_3);
}

inline void pack_patch_close(uint32_t cw, bool close) {
    volatile uint32_t* mop_cfg = reinterpret_cast<volatile uint32_t*>(TENSIX_MOP_CFG_BASE);
    ckernel::mop_sync();
    mop_cfg[MOP_LAST_OUTER_CFG_INDEX] = pacr(ADDR_MOD_1, tail_intf(cw), 0, close ? 1 : 0);
}

inline void pack_reset_src_counters() {
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

inline void pack_dest_offset(uint32_t rows) {
    TT_SETDMAREG(0, rows, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TT_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + rows, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
    select_packer_dest_registers<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>();
}
#endif

template <Diag diag>
ALWI void section_open() {
    MATH((_llk_math_wait_for_dest_available_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>()));
#ifdef TRISC_MATH
    math_section_begin();
#endif
}

template <Diag diag>
ALWI void section_close_math_wait_pack() {
#ifdef TRISC_MATH
    math::clear_dst_reg_addr();
#endif
    MATH((_llk_math_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));
    PACK((llk_packer_wait_for_math_done()));
}

// Whole-row mode: `rows` (1 .. 4 / ct) tile-rows from input tile in_tile0 to output tile out_tile0.
template <uint32_t ct, uint32_t icb, uint32_t ocb, Diag diag>
ALWI void section_whole_rows(uint32_t rows, uint32_t in_tile0, uint32_t out_tile0) {
    constexpr uint32_t unit_rows = Plan<ct>::unit_stride;  // DEST rows per unit
    [[maybe_unused]] const uint32_t units = 2 * rows;
    section_open<diag>();
#ifdef TRISC_UNPACK
    if constexpr (diag != Diag::PackOnly) {
        const uint32_t operand_id = get_operand_id(icb);
        unpack_tiles(
            llk_unpack_fast_untilize_tile_address(operand_id, in_tile0),
            get_local_cb_interface(operand_id).fifo_page_size,
            rows * ct,
            0);
    }
#endif
#ifdef TRISC_MATH
    if constexpr (diag != Diag::PackOnly) {
        whole_row_group<ct, Plan<ct>::rows_per_section, diag>(rows);
    }
#endif
    section_close_math_wait_pack<diag>();
#ifdef TRISC_PACK
    if constexpr (diag != Diag::NoPack && diag != Diag::UnpackOnly) {
        const uint32_t output_id = get_output_id(ocb);
        program_packer_destination(get_output_tile_address<true, PackMode::Default>(output_id, out_tile0));
        if constexpr (ct == 2) {
            pack_reset_src_counters();
            ckernel_template::run();
        } else {
            for (uint32_t k = 0; k < units; ++k) {
                if (k == 0) {
                    pack_patch_close(ct, false);
                } else if (k + 1 == units) {
                    pack_patch_close(ct, true);
                }
                pack_dest_offset(k * unit_rows);
                pack_reset_src_counters();
                ckernel_template::run();
            }
            pack_dest_offset(0);
        }
        pack_reset_src_counters();
    }
#endif
    PACK((_llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));
}

// Half (h) of a cw-tile chunk at column tile c0 of a tile-row: in_row / out_row index it in the input / output CB
// windows (they differ under Sync::Hybrid).
template <uint32_t ct, uint32_t icb, uint32_t ocb, Diag diag, bool strided, uint32_t cw, bool folded = false>
ALWI void section_half_chunk(uint32_t in_row, uint32_t out_row, uint32_t c0, uint32_t h) {
    section_open<diag>();
#ifdef TRISC_UNPACK
    if constexpr (diag != Diag::PackOnly) {
        const uint32_t operand_id = get_operand_id(icb);
        const uint32_t address = llk_unpack_fast_untilize_tile_address(operand_id, in_row * ct + c0);
        const uint32_t stride = get_local_cb_interface(operand_id).fifo_page_size;
        if constexpr (folded) {
            unpack_tiles_folded(address, stride, h);
        } else {
            unpack_tiles(address, stride, cw, h);
        }
    }
#endif
#ifdef TRISC_MATH
    if constexpr (diag != Diag::PackOnly) {
        unit_faces<0, 2 * cw, diag>();
    }
#endif
    section_close_math_wait_pack<diag>();
#ifdef TRISC_PACK
    if constexpr (diag != Diag::NoPack && diag != Diag::UnpackOnly) {
        const uint32_t output_id = get_output_id(ocb);
        // Slab row 16h of this tile-row, column tile c0: one bf16 slab row is 64 * ct B = 4 * ct 16 B units.
        program_packer_destination(
            get_output_tile_address<true, PackMode::Default>(output_id, out_row * ct) + h * 16 * 4 * ct + c0 * 4);
        pack_reset_src_counters();
        if constexpr (strided) {
            TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1000);  // ch1 Y = 0: this half's first slab row
        }
        ckernel_template::run();
    }
#endif
    PACK((_llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, DST_ACCUM_MODE>()));
}

template <uint32_t ct, uint32_t icb, uint32_t ocb, Diag diag, bool strided, uint32_t cw, bool folded = false>
ALWI void chunk_both_halves(uint32_t in_row, uint32_t out_row, uint32_t c0) {
    section_half_chunk<ct, icb, ocb, diag, strided, cw, folded>(in_row, out_row, c0, 0);
    section_half_chunk<ct, icb, ocb, diag, strided, cw, folded>(in_row, out_row, c0, 1);
}

// Chunk mode: chunk c (compile-time width) at column tile c0, both halves; the pack MOP is reprogrammed only where
// the width differs from the chunk packed before it (cyclically: the last chunk of the previous tile-row).
template <uint32_t ct, uint32_t icb, uint32_t ocb, Diag diag, uint32_t c, uint32_t c0>
ALWI void chunks(uint32_t in_row, uint32_t out_row) {
    using P = Plan<ct>;
    if constexpr (c < P::num_chunks) {
        constexpr uint32_t cw = P::chunk_w(c);
        constexpr uint32_t prev_cw = P::chunk_w(c == 0 ? P::num_chunks - 1 : c - 1);
        if constexpr (cw != prev_cw) {
            PACK((pack_mop_config<true>(cw, true)));
        }
        chunk_both_halves<ct, icb, ocb, diag, true, cw>(in_row, out_row, c0);
        chunks<ct, icb, ocb, diag, c + 1, c0 + cw>(in_row, out_row);
    }
}

// Untilizes num_rows ct-tile tile-rows from icb into ocb (ct bf16 pages per tile-row), section groups never
// straddling a batch_rows boundary (CBs whose capacity is whole batches never wrap inside a group). Includes init
// and uninit (formats reconfigured as the helper does).
template <
    uint32_t ct,
    uint32_t icb,
    uint32_t ocb,
    Diag diag = Diag::None,
    Sync sync = Sync::Group,
    bool folded = false,
    typename InDFB,
    typename OutDFB>
ALWI void untilize_wide(InDFB& in, OutDFB& out, uint32_t num_rows, uint32_t batch_rows) {
    using P = Plan<ct>;
    static_assert(DST_ACCUM_MODE, "fp32 DEST layout (4 tiles per half-sync section)");
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
    fast_untilize_init<P::init_ct>(icb, ocb);
    // Folded unpack: half-row mode only (one chunk width).
    constexpr bool fold = folded && !P::whole_rows && !P::strided;
    if constexpr (fold) {
        UNPACK((unpack_mop_config_folded<P::faces_per_visit>(ct)));
    } else {
        UNPACK((unpack_mop_config<P::faces_per_visit>()));
    }
    [[maybe_unused]] uint32_t prev_units = 0;
    if constexpr (P::whole_rows && ct == 2) {
        PACK((pack_addrmod_ct2()));
    } else if constexpr (!P::strided) {
        PACK((pack_mop_config<false>(ct, true)));
    } else {
        // The last chunk's width: chunks() reprograms at every width change, cyclically from it.
        PACK((pack_mop_config<true>(P::chunk_w(P::num_chunks - 1), true)));
    }
    constexpr uint32_t rps = P::rows_per_section;
    for (uint32_t done = 0; done < num_rows; done += batch_rows) {
        const uint32_t n = num_rows - done < batch_rows ? num_rows - done : batch_rows;
        if constexpr (sync == Sync::Batch || sync == Sync::Hybrid) {
            in.wait_front(n * ct);
        }
        if constexpr (sync != Sync::Group) {
            out.reserve_back(n * ct);
        }
        for (uint32_t r = 0; r < n; r += rps) {
            const uint32_t g = n - r < rps ? n - r : rps;
            // Tile-row index of this group inside the CB windows.
            const uint32_t base = sync == Sync::Batch ? r : 0;
            if constexpr (sync == Sync::Group) {
                in.wait_front(g * ct);
                out.reserve_back(g * ct);
            } else if constexpr (sync == Sync::Lazy) {
                in.wait_front((r + g) * ct);
            }
            const uint32_t in_base = sync == Sync::Group ? 0 : r;
            if constexpr (P::whole_rows) {
#ifdef TRISC_PACK
                if constexpr (ct == 2) {
                    if (2 * g != prev_units) {
                        pack_mop_config_ct2(2 * g);
                        prev_units = 2 * g;
                    }
                }
#endif
                section_whole_rows<ct, icb, ocb, diag>(g, in_base * ct, base * ct);
            } else if constexpr (!P::strided) {
                chunk_both_halves<ct, icb, ocb, diag, false, ct, fold>(in_base, base, 0);
            } else {
                chunks<ct, icb, ocb, diag, 0, 0>(in_base, base);
            }
            if constexpr (sync == Sync::Group) {
                in.pop_front(g * ct);
                out.push_back(g * ct);
            } else if constexpr (sync == Sync::Hybrid || sync == Sync::Lazy) {
                out.push_back(g * ct);
            }
        }
        if constexpr (sync == Sync::Batch) {
            in.pop_front(n * ct);
            out.push_back(n * ct);
        } else if constexpr (sync == Sync::Hybrid || sync == Sync::Lazy) {
            in.pop_front(n * ct);
        }
    }
    // Leave ch0 X/Y, ch1 Y and Z/W at 0, as the helper's pack does.
    PACK((TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1011)));
    PACK((TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b1111)));
    fast_untilize_uninit<P::init_ct>(ocb);
}

// ---------------------------------------------------------------------------------------------------------------
// Diagnostic re-implementation of the helper's fast_untilize row loop (the baseline), with stage stubs.
// ---------------------------------------------------------------------------------------------------------------
template <uint32_t ct, uint32_t icb, uint32_t ocb, Diag diag>
ALWI void base_block(uint32_t in_tile, uint32_t out_tile) {
    constexpr bool fp32 = DST_ACCUM_MODE;
    if constexpr (ct <= FAST_UNTILIZE_MAX_UNIT_DIM) {
        MATH((_llk_math_wait_for_dest_available_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>()));
        if constexpr (diag != Diag::PackOnly) {
            UNPACK((llk_unpack_fast_untilize_block(icb, in_tile, ct)));
            MATH((llk_math_fast_untilize_block<fp32>(0, ct)));
        }
        MATH((_llk_math_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, fp32>()));
        PACK((llk_packer_wait_for_math_done()));
        if constexpr (diag != Diag::NoPack) {
            PACK((llk_pack_fast_untilize_block<FAST_UNTILIZE_MAX_UNIT_DIM>(ocb, out_tile, ct)));
        }
        PACK((_llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, fp32>()));
    } else {
        uint32_t tiles_done = 0;
        constexpr uint32_t first_unpack_unit_dim = fast_untilize_next_unit_dim(ct);
        [[maybe_unused]] uint32_t prev_unpack_unit_dim = first_unpack_unit_dim;
        [[maybe_unused]] uint32_t prev_pack_unit_dim = 0;
        while (tiles_done < ct) {
            const uint32_t unit_dim = fast_untilize_next_unit_dim(ct - tiles_done);
            MATH((_llk_math_wait_for_dest_available_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>()));
            if constexpr (diag != Diag::PackOnly) {
                UNPACK(
                    (llk_unpack_fast_untilize_block<fp32>(icb, in_tile + tiles_done, unit_dim, prev_unpack_unit_dim)));
                MATH((llk_math_fast_untilize_block<fp32>(0, unit_dim)));
            }
            MATH((_llk_math_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, fp32>()));
            PACK((llk_packer_wait_for_math_done()));
            if constexpr (diag != Diag::NoPack) {
                PACK((llk_pack_fast_untilize_block_strided<FAST_UNTILIZE_MAX_UNIT_DIM, ct>(
                    ocb, out_tile, tiles_done, unit_dim, prev_pack_unit_dim)));
            }
            PACK((_llk_pack_dest_section_done_<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE, fp32>()));
            tiles_done += unit_dim;
        }
        UNPACK((llk_unpack_fast_untilize_restore_unit_dim<fp32>(icb, first_unpack_unit_dim, prev_unpack_unit_dim)));
    }
}

template <uint32_t ct, uint32_t icb, uint32_t ocb, Diag diag, typename InDFB, typename OutDFB>
ALWI void untilize_base(InDFB& in, OutDFB& out, uint32_t num_rows) {
    reconfig_data_format_srca(icb);
    pack_reconfig_data_format(ocb);
    fast_untilize_init<ct>(icb, ocb);
    for (uint32_t r = 0; r < num_rows; ++r) {
        in.wait_front(ct);
        out.reserve_back(ct);
        base_block<ct, icb, ocb, diag>(0, 0);
        in.pop_front(ct);
        out.push_back(ct);
    }
    fast_untilize_uninit<ct>(ocb);
}

}  // namespace wide_untilize
