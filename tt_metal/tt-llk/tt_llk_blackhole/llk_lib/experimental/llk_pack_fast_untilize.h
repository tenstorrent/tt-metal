// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BH Fast-Untilize Pack (tt-metal#42048 + #42049).
//
// Read math DEST layout (4 tiles, ct=4; 16-bit or native fp32 DEST) emitted by
// `_llk_math_fast_untilize_*`:
//   Dst rows   0..63:  t0.F0 | t0.F1 | t1.F0 | t1.F1
//   Dst rows  64..127: t2.F0 | t2.F1 | t3.F0 | t3.F1
//   Dst rows 128..191: t0.F2 | t0.F3 | t1.F2 | t1.F3
//   Dst rows 192..255: t2.F2 | t2.F3 | t3.F2 | t3.F3
//
// Full-width PACRs use ALL_INTF_ACTIVE + STRIDED_MODE to read 4 face-rows at
// Dst rows R, R+16, R+32, R+48 (the 16-row interface stride), concatenating
// them into 64 contiguous L1 datums. For unit_dim=3, the tail PACR uses
// TWO_INTFS_ACTIVE and emits the remaining tile's 32 datums. Regular
// pack_untilize uses two interfaces for the common 4-face layout, so this wider
// DEST layout is the pack-side bandwidth win for fast-untilize.
//
// Contiguous PACR sequence per block call:
//   unit_dim=2: 16 rows x 1 PACR x 2 phases = 32 PACRs.
//   unit_dim=3/4: 16 rows x 2 PACRs x 2 phases = 64 PACRs.
// For unit_dim=4, regular pack_untilize needs 16 rows x 4 tile slots x
// 2 phases = 128 two-interface PACRs. The fast DEST layout reduces that to
// 16 rows x 2 block groups x 2 phases = 64 four-interface PACRs.
//   Phase 1 emits top strip rows with Last=0.
//   Phase 2 emits bottom strip rows with Last=1, closing one contiguous L1 stream.
//
// L1 writes are contiguous via PACR's internal datum counter; no per-PACR
// L1_Dest_addr cfg update needed. Single L1 cfg slot (SEC0_REG1). This is only
// valid when the chunk spans the full output row.
//
// Wider rows use a row-strided stream: each chunk row closes with Last=1. The
// MOP path advances L1_Dest_addr by incrementing the destination Y counter
// programmed with one full output-row stride.
//
// Source ADC strides (bytes; x-stride is not used by this PACR shape):
//   y_stride = FACE_C_DIM * bpe     (1 face-row = 16 datums per y+=1)
//   z_stride = FAST_UNTILIZE_BLOCK_STRIDE_ROWS * y_stride
//   w_stride = FAST_UNTILIZE_PHASE_PAIR_STRIDE_ROWS * y_stride
//
// AddrMod scheme:
//   ADDR_MOD_0: z_src.incr=1                       (post-PACR_pair_0: go to second block)
//   ADDR_MOD_1: y_src.incr=1, z_src.clr=1          (post-PACR_pair_1: advance row, reset block)
//   ADDR_MOD_1: y_dst.incr=1 when using the ch1 output-counter row advance
//
// The contiguous path uses two MOP runs per call (phase 1 and phase 2).
// L1_Dest_addr is programmed once at the base address; phase 1 intentionally
// leaves the pack stream open so phase 2 naturally continues at row 16. The
// phase source half is selected by reprogramming the active pack DEST target
// offset: active_half + FAST_UNTILIZE_PACK_TOP_STRIP_DEST_TARGET_OFFSET for
// top rows, then active_half + FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET
// for bottom rows. These are BH remapped DEST-target offsets, not literal row
// numbers from the layout table above.
//
// DOMAIN: unit_dim=2/3/4, num_faces=4, private SyncHalf DEST buffering, and
// Float16_b or Float32 output from supported fast-untilize math layouts. Callers should
// route ct=1, non-4-face shapes, and unsupported formats to standard untilize.

#pragma once

#include <cstdint>

#include "experimental/llk_fast_untilize_common.h"
#include "llk_pack.h"

namespace ckernel
{

constexpr std::uint32_t FAST_UNTILIZE_MOP_LAST_OUTER_CFG_INDEX = 7;

template <bool is_strided_row>
inline void _llk_pack_fast_untilize_configure_addrmod_()
{
    // ADDR_MOD_0: after PACR reading block A, advance z to read block A+1.
    addr_mod_pack_t {.z_src = {.incr = 1}}.set(ADDR_MOD_0);

    // ADDR_MOD_1: after PACR reading block A+1, advance to next row in block A
    // (y+=1) and reset z to 0 (back to block A).
    if constexpr (is_strided_row)
    {
        addr_mod_pack_t {.y_src = {.incr = 1}, .y_dst = {.incr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_1);
    }
    else
    {
        addr_mod_pack_t {.y_src = {.incr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_1);
    }
}

// MOP body: 16 outer iterations x 2 inner PACRs.
// Inner PACR pair: PACR(AM0) writes 64 datums from block A, then PACR(AM1)
// writes 64 datums from block A+1.
//
// MOP runs once per phase (top/bottom). For unit_dim=4 this is exactly
// 2 phases x 16 rows x 2 PACRs = 64 PACRs.
inline std::uint32_t _llk_pack_fast_untilize_row_pacr_(
    const std::uint32_t addr_mod, const std::uint32_t read_intf_sel, const std::uint32_t last, const std::uint32_t concat)
{
    return TT_OP_PACR(
        p_pacr::CFG_CTXT_0,
        p_pacr::NO_ROW_PAD_ZERO,
        p_pacr::DST_ACCESS_STRIDED_MODE,
        addr_mod,
        p_pacr::ADDR_CNT_CTXT_0,
        0,
        read_intf_sel,
        0,
        concat,
        p_pacr::NO_CTXT_CTRL,
        0,
        last);
}

inline std::uint32_t _llk_pack_fast_untilize_tail_read_intf_(const std::uint32_t unit_dim)
{
    return unit_dim == 3 ? p_pacr::TWO_INTFS_ACTIVE : p_pacr::ALL_INTF_ACTIVE;
}

inline std::uint32_t _llk_pack_fast_untilize_last_outer_pacr_(const std::uint32_t unit_dim, const bool last)
{
    return _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, _llk_pack_fast_untilize_tail_read_intf_(unit_dim), last ? 1 : 0, 0);
}

inline void _llk_pack_fast_untilize_mop_config_(const std::uint32_t unit_dim, const bool last)
{
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize pack supports unit_dim 2, 3, or 4");

    constexpr std::uint32_t MOP_OUTER_LOOP = FAST_UNTILIZE_PHASE_ROWS;
    constexpr std::uint32_t MOP_INNER_LOOP = 1; // one PACR pair per strip row

    if (unit_dim == 2)
    {
        ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, _llk_pack_fast_untilize_last_outer_pacr_(unit_dim, false));
        tmp.set_last_outer_loop_instr(_llk_pack_fast_untilize_last_outer_pacr_(unit_dim, last));
        tmp.program();
    }
    else
    {
        const std::uint32_t tail_intf = _llk_pack_fast_untilize_tail_read_intf_(unit_dim);
        // unit_dim=4 has no tail: both PACRs read all four interfaces. unit_dim=3
        // keeps the first full-width PACR and narrows only the second/tail PACR.
        ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0, 0),
            _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, tail_intf, 0, 0));
        tmp.set_last_outer_loop_instr(_llk_pack_fast_untilize_last_outer_pacr_(unit_dim, last));
        tmp.program();
    }
}

inline void _llk_pack_fast_untilize_mop_patch_last_(const std::uint32_t unit_dim, const bool last)
{
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize pack supports unit_dim 2, 3, or 4");

    const std::uint32_t last_outer_pacr = _llk_pack_fast_untilize_last_outer_pacr_(unit_dim, last);

    // MOP config slot 7 is loop0_last_instr: the instruction used for the last
    // inner iteration of the last outer iteration. The MOP body is otherwise
    // unchanged between phase 1 and phase 2, so patch only the final PACR's
    // Last bit instead of rebuilding the whole template.
    volatile std::uint32_t* mop_cfg = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
    ckernel::mop_sync();
    mop_cfg[FAST_UNTILIZE_MOP_LAST_OUTER_CFG_INDEX] = last_outer_pacr;
}

inline void _llk_pack_fast_untilize_reset_src_counters_()
{
    // Source counters select the row within a block group and the block group
    // itself. Reset before each phase so phase selection is controlled only by
    // DEST target offset, not leftover PAC address-generator state.
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b0011);
    TTI_SETADCZW(p_setadc::PAC, 0, 0, 0, 0, 0b0101);
}

inline void _llk_pack_fast_untilize_reset_output_row_counter_()
{
    TTI_SETADCXY(p_setadc::PAC, 0, 0, 0, 0, 0b1000);
}

inline void _llk_pack_fast_untilize_strided_mop_config_(const std::uint32_t unit_dim, const std::uint32_t rows_per_run)
{
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize strided pack supports unit_dim 2, 3, or 4");

    // One MOP run emits rows_per_run output rows. A phase is split into
    // FAST_UNTILIZE_PHASE_ROWS / rows_per_run runs so the carried output-Y
    // offset stays inside the packer window (see _llk_pack_fast_untilize_block_strided_).
    const std::uint32_t MOP_OUTER_LOOP     = rows_per_run;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    if (unit_dim == 2)
    {
        ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, p_pacr::ALL_INTF_ACTIVE, 1, 0));
        tmp.program();
    }
    else
    {
        const std::uint32_t tail_intf = _llk_pack_fast_untilize_tail_read_intf_(unit_dim);
        ckernel_template tmp(
            MOP_OUTER_LOOP,
            MOP_INNER_LOOP,
            _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0, 1),
            _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, tail_intf, 1, 0));
        tmp.program();
    }
}

inline void _llk_pack_fast_untilize_program_output_row_stride_(const std::uint32_t output_row_stride_bytes)
{
    // Wider rows close the pack stream after each chunk row. Program the PAC
    // output Y stride to one full row of the row-major tensor so the ch1 output
    // counter lands at the same column in the next output row.
    // This relies on BH packer channel-1 output-address generation through
    // PCK0_ADDR_CTRL_XY_REG_1 plus AddrMod y_dst.incr. Current public BH ISA
    // docs do not pin down that packer channel-1 stride/address behavior.
    TT_SETDMAREG(0, LOWER_HALFWORD(output_row_stride_bytes << PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(output_row_stride_bytes << PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_pack::TMP1));
    // TMP0/TMP1 are produced by SETDMAREG and consumed as WRCFG input GPRs.
    // Keep the WRCFGs behind the same STALL_CFG/THCON barrier used by existing
    // LLK SETDMAREG->WRCFG sequences so the config write cannot race the GPR
    // setup. ISA documents WRCFG's GPR source and STALLWAIT's CFG block mask.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_BASE_REG_1_Base_ADDR32);
    // Make the helper self-contained: callers may immediately issue PACRs/MOPs
    // that consume this config, and WRCFG.md requires a one-instruction bubble
    // before a following consumer.
    TTI_NOP;
}

inline void _llk_pack_fast_untilize_clear_output_row_stride_()
{
    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_pack::TMP0));
    // Same SETDMAREG->WRCFG dependency as row-stride programming above: clear
    // TMP0 first, then keep the WRCFGs behind the established LLK barrier.
    // ISA documents WRCFG's GPR source and STALLWAIT's CFG block mask.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_BASE_REG_1_Base_ADDR32);
    // Same self-contained WRCFG scheduling bubble as above before later pack
    // address generation observes the cleared config.
    TTI_NOP;
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void _llk_pack_fast_untilize_init_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format)
{
    static_assert(block_ct_dim >= 2 && block_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM, "BH fast untilize supports block_ct_dim 2, 3, or 4");

    TTI_SETDMAREG(0, 0x000, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));
    select_packer_dest_registers<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>();

    TTI_SETADCXX(p_setadc::PAC, FACE_C_DIM - 1, 0x0);

    // Strides for our row/block/phase advance scheme.
    const std::uint32_t bytes_per_datum = (pack_src_format & 0x3) == ckernel::to_underlying(DataFormat::Float32)   ? 4
                                          : (pack_src_format & 0x3) == ckernel::to_underlying(DataFormat::Float16) ? 2
                                                                                                                   : 1;
    // y_stride: 1 face-row of 16 datums per y+=1
    const std::uint32_t y_stride = FACE_C_DIM * bytes_per_datum;
    // z_stride: one block of four face-tile-groups per z+=1.
    const std::uint32_t z_stride = FAST_UNTILIZE_BLOCK_STRIDE_ROWS * y_stride;
    // w_stride: retained for consistency with the pack address generator setup.
    const std::uint32_t w_stride = FAST_UNTILIZE_PHASE_PAIR_STRIDE_ROWS * y_stride;

    TT_SETDMAREG(0, LOWER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT), 0, HI_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, LOWER_HALFWORD(z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT), 0, LO_16(p_gpr_pack::TMP1));
    TT_SETDMAREG(0, UPPER_HALFWORD(w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT), 0, HI_16(p_gpr_pack::TMP1));
    // Serialize the TMP GPR setup before WRCFG copies those stride values into
    // packer address-generator config. This is the same LLK SETDMAREG->WRCFG
    // barrier pattern used elsewhere; ISA documents WRCFG's GPR source and
    // STALLWAIT's Wait Gate blocking.
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TTI_WRCFG(p_gpr_pack::TMP1, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);

    constexpr bool is_strided_row = full_ct_dim > block_ct_dim;
    _llk_pack_fast_untilize_configure_addrmod_<is_strided_row>();
    if constexpr (full_ct_dim > block_ct_dim)
    {
        const std::uint32_t output_row_stride = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * TILE_C_DIM);
        _llk_pack_fast_untilize_reset_src_counters_();
        _llk_pack_fast_untilize_program_output_row_stride_(output_row_stride);
        _llk_pack_fast_untilize_reset_output_row_counter_();
    }
}

template <std::uint32_t phase_offset>
inline void _llk_pack_fast_untilize_select_phase_()
{
    // Phase selection is implemented by moving the active DEST target window,
    // not by changing the PAC source strides. The phase_offset values are named
    // in llk_fast_untilize_common.h because they are part of the shared
    // math/pack layout contract.
    TTI_SETDMAREG(0, phase_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_LO + 0));
    TTI_SETDMAREG(0, DEST_REGISTER_HALF_SIZE + phase_offset, 0, LO_16(p_gpr_pack::DEST_OFFSET_HI + 0));

    select_packer_dest_registers<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>();
}

template <bool reset_output_row_counter>
inline void _llk_pack_fast_untilize_restore_pack_counters_()
{
    // Leave the standard pack counter postcondition for the next LLK in fused
    // kernels. The regular untilize path also restores PAC Z/W after packing.
    _llk_pack_fast_untilize_reset_src_counters_();
    if constexpr (reset_output_row_counter)
    {
        _llk_pack_fast_untilize_reset_output_row_counter_();
    }
    set_dst_write_addr(0);
}

// One call processes one 2/3/4-tile chunk.
// Output: one chunk's worth of RM strip starting at `address` (in 16B units).
template <std::uint32_t block_ct_dim>
inline void _llk_pack_fast_untilize_block_(const std::uint32_t address, const std::uint32_t unit_dim, std::uint32_t& prev_unit_dim)
{
    static_assert(block_ct_dim >= 2 && block_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM, "BH fast untilize supports block_ct_dim 2, 3, or 4");
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= block_ct_dim, "fast_untilize pack unit_dim must be in [2, block_ct_dim]");

    program_packer_destination(address);

    // Phase 1 emits top strip rows and keeps the pack stream open.
    if (unit_dim != prev_unit_dim)
    {
        _llk_pack_fast_untilize_mop_config_(unit_dim, false);
        prev_unit_dim = unit_dim;
    }
    else
    {
        _llk_pack_fast_untilize_mop_patch_last_(unit_dim, false);
    }
    _llk_pack_fast_untilize_select_phase_<FAST_UNTILIZE_PACK_TOP_STRIP_DEST_TARGET_OFFSET>();
    _llk_pack_fast_untilize_reset_src_counters_();
    ckernel_template::run();

    // Phase 2 emits bottom strip rows and closes the stream. Phase 1 already
    // programmed the MOP body; only the final PACR's Last bit changes.
    _llk_pack_fast_untilize_mop_patch_last_(unit_dim, true);
    _llk_pack_fast_untilize_select_phase_<FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET>();
    _llk_pack_fast_untilize_reset_src_counters_();
    ckernel_template::run();
    _llk_pack_fast_untilize_restore_pack_counters_<false>();
}

// The packer carries the destination Y offset (ADC.Y * Ystride) relative to the
// last-programmed L1 base. On BH silicon that carried offset is only reliably
// preserved inside a ~256 KiB window; beyond it the high address bits are
// dropped and rows overwrite each other (verified on silicon: BF16 full_ct_dim
// 133/256 corrupt without a rebase, FP32 corrupts from ~ct 137). The window is
// not pinned down in the public BH ISA docs (see
// _llk_pack_fast_untilize_program_output_row_stride_); 256 KiB is the
// silicon-characterized bound used here.
constexpr std::uint32_t PACKER_CARRIED_OUTPUT_Y_OFFSET_WINDOW_16B = 256 * 1024 / 16;

// Emit one phase (16 output rows) as `runs_per_phase` runs of `rows_per_run`
// rows. Each run reprograms the L1 base and resets the destination Y counter so
// the carried offset within a run never leaves the window, while the source
// counters advance continuously across runs (only the dst Y counter is reset).
template <std::uint32_t phase_offset>
inline void _llk_pack_fast_untilize_emit_phase_(
    const std::uint32_t address,
    const std::uint32_t phase_base_row,
    const std::uint32_t runs_per_phase,
    const std::uint32_t rows_per_run,
    const std::uint32_t output_row_stride_16B)
{
    _llk_pack_fast_untilize_select_phase_<phase_offset>();
    _llk_pack_fast_untilize_reset_src_counters_();
    for (std::uint32_t run = 0; run < runs_per_phase; run++)
    {
        const std::uint32_t out_row = phase_base_row + run * rows_per_run;
        _llk_pack_fast_untilize_reset_output_row_counter_();
        program_packer_destination(address + out_row * output_row_stride_16B);
        ckernel_template::run();
    }
}

// One call processes one 2/3/4-tile chunk inside a wider row.
// Output address points at this chunk's row-0 column in the row-major tensor.
template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void _llk_pack_fast_untilize_block_strided_(
    const std::uint32_t address, const std::uint32_t unit_dim, std::uint32_t& prev_unit_dim, const std::uint32_t output_row_stride_16B = 0)
{
    static_assert(block_ct_dim >= 2 && block_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM, "BH fast untilize strided path supports block_ct_dim 2, 3, or 4");
    static_assert(full_ct_dim > block_ct_dim, "Use the contiguous fast_untilize block when the chunk is the full row");
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= block_ct_dim, "fast_untilize pack unit_dim must be in [2, block_ct_dim]");

    constexpr std::uint32_t MAX_CARRIED_OUTPUT_Y_ROW = 2 * FAST_UNTILIZE_PHASE_ROWS - 1;

    // Fast path: the packer can carry y_dst across all 32 rows of the chunk from
    // a single base without leaving the window. output_row_stride_16B == 0 means
    // a legacy caller that did not supply the stride; keep the old carry behavior.
    const bool carry_full_chunk = output_row_stride_16B == 0 || MAX_CARRIED_OUTPUT_Y_ROW * output_row_stride_16B < PACKER_CARRIED_OUTPUT_Y_OFFSET_WINDOW_16B;

    // Otherwise rebase every rows_per_run rows. rows_per_run is the largest
    // power-of-two divisor of FAST_UNTILIZE_PHASE_ROWS whose top row stays inside
    // the window: (rows_per_run - 1) * stride < window. It bottoms out at 1 (one
    // base reprogram per row), so arbitrarily wide rows / FP32 are handled.
    std::uint32_t rows_per_run = FAST_UNTILIZE_PHASE_ROWS;
    if (!carry_full_chunk)
    {
        while (rows_per_run > 1 && (rows_per_run - 1) * output_row_stride_16B >= PACKER_CARRIED_OUTPUT_Y_OFFSET_WINDOW_16B)
        {
            rows_per_run >>= 1;
        }
    }

    if (unit_dim != prev_unit_dim)
    {
        _llk_pack_fast_untilize_strided_mop_config_(unit_dim, carry_full_chunk ? FAST_UNTILIZE_PHASE_ROWS : rows_per_run);
        prev_unit_dim = unit_dim;
    }

    if (carry_full_chunk)
    {
        // Single base; phase 1 leaves the stream open and phase 2 continues the
        // y_dst carry into output rows 16..31. Matches the original strided path.
        program_packer_destination(address);
        _llk_pack_fast_untilize_select_phase_<FAST_UNTILIZE_PACK_TOP_STRIP_DEST_TARGET_OFFSET>();
        ckernel_template::run();
        _llk_pack_fast_untilize_select_phase_<FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET>();
        _llk_pack_fast_untilize_reset_src_counters_();
        ckernel_template::run();
        _llk_pack_fast_untilize_restore_pack_counters_<true>();
        return;
    }

    const std::uint32_t runs_per_phase = FAST_UNTILIZE_PHASE_ROWS / rows_per_run;
    _llk_pack_fast_untilize_emit_phase_<FAST_UNTILIZE_PACK_TOP_STRIP_DEST_TARGET_OFFSET>(address, 0, runs_per_phase, rows_per_run, output_row_stride_16B);
    _llk_pack_fast_untilize_emit_phase_<FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET>(
        address, FAST_UNTILIZE_PHASE_ROWS, runs_per_phase, rows_per_run, output_row_stride_16B);
    _llk_pack_fast_untilize_restore_pack_counters_<true>();
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void _llk_pack_fast_untilize_uninit_(const std::uint32_t pack_src_format)
{
    if constexpr (full_ct_dim > block_ct_dim)
    {
        _llk_pack_fast_untilize_clear_output_row_stride_();
    }
    set_packer_strides<PackMode::Default>(pack_src_format, TILE_C_DIM);
    // init owns the X counter and sets it itself; strides are restored just above, so skip them in init.
    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        pack_src_format, FACE_R_DIM, TILE_C_DIM, FAST_UNTILIZE_NUM_FACES, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
}

} // namespace ckernel
