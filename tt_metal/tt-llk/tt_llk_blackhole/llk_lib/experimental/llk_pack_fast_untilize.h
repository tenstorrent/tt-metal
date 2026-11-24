// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_pack_fast_untilize.h. Same Tensix instruction sequence and
// access kinds as production, expressed through the HAL, except the HAL-GAP marked below.
//
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
#include <type_traits>

#include "experimental/llk_fast_untilize_common.h"
#include "hal/cfg.h"
#include "hal/counters/adc.h"
#include "hal/mop.h"
#include "hal/nop.h"
#include "hal/pack.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

namespace hal_port
{

#ifndef HAL_PORT_ADC_SET_MASKED_DEFINED_
#define HAL_PORT_ADC_SET_MASKED_DEFINED_

// Issue a compile-time Set/AdvanceAndReset of one X/Y or Z/W group as the masked SETADCXY/SETADCZW (or ADDRCR*)
// production uses. adc::set plans a lone selected counter as SETADC instead, a different instruction.
// HAL-GAP: no public encoder forces the masked packed form for a single counter; uses hal::counters::adc::detail::packed.
template <hal::counters::adc::AddrCounter D>
inline __attribute__((always_inline)) void adc_set_masked()
{
    namespace adc = hal::counters::adc;
    constexpr adc::detail::Group xy {{D.channel0.x, D.channel0.y, D.channel1.x, D.channel1.y}};
    constexpr adc::detail::Group zw {{D.channel0.z, D.channel0.w, D.channel1.z, D.channel1.w}};
    static_assert(adc::detail::is_valid(D), "address-counter descriptor requires valid clients, an action, and at least one field");
    static_assert((xy.mask() == 0) != (zw.mask() == 0), "adc_set_masked takes counters of one group only");
    constexpr bool is_zw         = zw.mask() != 0;
    constexpr std::uint32_t word = adc::detail::packed(D.clients, is_zw, D.action, is_zw ? zw : xy);
    TTI_INSN(word);
}
#endif

namespace fast_untilize_pack
{

namespace cfg = hal::cfg;

// One packer ADDR_MOD slot S (SETC16 of the whole addr_mod_pack_t word): every field of the 16-bit word supplied.
template <cfg::Sec S, std::uint32_t y_src_incr, std::uint32_t y_dst_incr, std::uint32_t z_src_incr, std::uint32_t z_src_clr>
inline void set_addr_mod_pack()
{
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Incr, S, y_src_incr>(),
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Y].Clear, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Y].Incr, S, y_dst_incr>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Y].CR, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Y].Clear, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Z].Incr, S, z_src_incr>(),
        cfg::set<cfg::AddrMod[cfg::Src][cfg::Z].Clear, S, z_src_clr>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Z].Incr, S, 0>(),
        cfg::set<cfg::AddrMod[cfg::Dest][cfg::Z].Clear, S, 0>());
}

inline constexpr auto Ch0 = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0];
inline constexpr auto Ch1 = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg1];
static_assert(Ch0.Xstride.addr32(cfg::Sec::S0) == PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32, "packer ch0 X/Y-stride word");
static_assert(Ch0.Zstride.addr32(cfg::Sec::S0) == PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32, "packer ch0 Z/W-stride word");
static_assert(Ch1.Xstride.addr32(cfg::Sec::S0) == PCK0_ADDR_CTRL_XY_REG_1_Xstride_ADDR32, "packer ch1 X/Y-stride word");
static_assert(Ch0.Ystride.shamt(cfg::Sec::S0) == PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, "Y-stride shift");
static_assert(Ch0.Zstride.shamt(cfg::Sec::S0) == PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT, "Z-stride shift");
static_assert(Ch0.Wstride.shamt(cfg::Sec::S0) == PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT, "W-stride shift");
static_assert(Ch1.Ystride.shamt(cfg::Sec::S0) == PCK0_ADDR_CTRL_XY_REG_1_Ystride_SHAMT, "ch1 Y-stride shift");

} // namespace fast_untilize_pack
} // namespace hal_port

namespace ckernel
{

constexpr std::uint32_t FAST_UNTILIZE_MOP_LAST_OUTER_CFG_INDEX = 7;
static_assert(
    std::is_same_v<decltype(hal::mop::last_outer), const hal::mop::Field<FAST_UNTILIZE_MOP_LAST_OUTER_CFG_INDEX>>, "MopCfg word 7 is hal::mop::last_outer");

template <bool is_strided_row>
inline void _llk_pack_fast_untilize_configure_addrmod_()
{
    namespace fup = hal_port::fast_untilize_pack;
    using hal::cfg::Sec;

    // ADDR_MOD_0: after PACR reading block A, advance z to read block A+1.
    fup::set_addr_mod_pack<Sec::S0, 0 /*y_src_incr*/, 0 /*y_dst_incr*/, 1 /*z_src_incr*/, 0 /*z_src_clr*/>();

    // ADDR_MOD_1: after PACR reading block A+1, advance to next row in block A
    // (y+=1) and reset z to 0 (back to block A).
    if constexpr (is_strided_row)
    {
        fup::set_addr_mod_pack<Sec::S1, 1 /*y_src_incr*/, 1 /*y_dst_incr*/, 0 /*z_src_incr*/, 1 /*z_src_clr*/>();
    }
    else
    {
        fup::set_addr_mod_pack<Sec::S1, 1 /*y_src_incr*/, 0 /*y_dst_incr*/, 0 /*z_src_incr*/, 1 /*z_src_clr*/>();
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
    // TT_OP_PACR(CFG_CTXT_0, NO_ROW_PAD_ZERO, DST_ACCESS_STRIDED_MODE, addr_mod, ADDR_CNT_CTXT_0, 0, read_intf_sel, 0,
    //            concat, NO_CTXT_CTRL, 0, last)
    return hal::pack::DataTransfer {
        .address_modifier = static_cast<std::uint8_t>(addr_mod),
        .interfaces       = static_cast<std::uint8_t>(read_intf_sel),
        .dest_access      = hal::pack::DestAccess::Strided,
        .concatenation    = concat ? hal::pack::Concatenation::Append : hal::pack::Concatenation::NewRow,
        .boundary         = last ? hal::pack::TileBoundary::Last : hal::pack::TileBoundary::NotLast,
    }
        .get_operation();
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
    namespace mop = hal::mop;

    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize pack supports unit_dim 2, 3, or 4");

    constexpr std::uint32_t MOP_OUTER_LOOP = FAST_UNTILIZE_PHASE_ROWS;
    constexpr std::uint32_t MOP_INNER_LOOP = 1; // one PACR pair per strip row

    // Explicit NOP end ops keep production's MOP_CFG stores (end_op0/end_op1 = NOP) unchanged. The last-inner op
    // defaults to the inner loop's last op, as production's ckernel_template constructor sets it.
    if (unit_dim == 2)
    {
        mop::program(mop::loop(MOP_OUTER_LOOP, mop::loop(MOP_INNER_LOOP, _llk_pack_fast_untilize_last_outer_pacr_(unit_dim, false)), TT_OP_NOP, TT_OP_NOP)
                         .patch(mop::last_outer, _llk_pack_fast_untilize_last_outer_pacr_(unit_dim, last)));
    }
    else
    {
        const std::uint32_t tail_intf = _llk_pack_fast_untilize_tail_read_intf_(unit_dim);
        // unit_dim=4 has no tail: both PACRs read all four interfaces. unit_dim=3
        // keeps the first full-width PACR and narrows only the second/tail PACR.
        mop::program(mop::loop(
                         MOP_OUTER_LOOP,
                         mop::loop(
                             MOP_INNER_LOOP,
                             _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0, 0),
                             _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, tail_intf, 0, 0)),
                         TT_OP_NOP,
                         TT_OP_NOP)
                         .patch(mop::last_outer, _llk_pack_fast_untilize_last_outer_pacr_(unit_dim, last)));
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
    hal::mop::update(hal::mop::last_outer, last_outer_pacr); // mop_sync(), then MopCfg[7] = last_outer_pacr
}

inline void _llk_pack_fast_untilize_reset_src_counters_()
{
    namespace adc = hal::counters::adc;
    // Source counters select the row within a block group and the block group
    // itself. Reset before each phase so phase selection is controlled only by
    // DEST target offset, not leftover PAC address-generator state.
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}}.for_clients<adc::Client::Packers>()>();               // SETADCXY(PAC, mask 0b0011)
    adc::set<adc::AddrCounter {.channel0 = {.z = 0}, .channel1 = {.z = 0}}.for_clients<adc::Client::Packers>()>(); // SETADCZW(PAC, mask 0b0101)
}

inline void _llk_pack_fast_untilize_reset_output_row_counter_()
{
    namespace adc = hal::counters::adc;
    hal_port::adc_set_masked<adc::AddrCounter {.channel1 = {.y = 0}}.for_clients<adc::Client::Packers>()>(); // SETADCXY(PAC, mask 0b1000)
}

inline void _llk_pack_fast_untilize_strided_mop_config_(const std::uint32_t unit_dim, const std::uint32_t rows_per_run)
{
    namespace mop = hal::mop;

    LLK_ASSERT(unit_dim >= 2 && unit_dim <= 4, "fast_untilize strided pack supports unit_dim 2, 3, or 4");

    // One MOP run emits rows_per_run output rows. A phase is split into
    // FAST_UNTILIZE_PHASE_ROWS / rows_per_run runs so the carried output-Y
    // offset stays inside the packer window (see _llk_pack_fast_untilize_block_strided_).
    const std::uint32_t MOP_OUTER_LOOP     = rows_per_run;
    constexpr std::uint32_t MOP_INNER_LOOP = 1;

    // Explicit NOP end ops keep production's MOP_CFG stores (end_op0/end_op1 = NOP) unchanged.
    if (unit_dim == 2)
    {
        mop::program(mop::loop(
            MOP_OUTER_LOOP, mop::loop(MOP_INNER_LOOP, _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, p_pacr::ALL_INTF_ACTIVE, 1, 0)), TT_OP_NOP, TT_OP_NOP));
    }
    else
    {
        const std::uint32_t tail_intf = _llk_pack_fast_untilize_tail_read_intf_(unit_dim);
        mop::program(mop::loop(
            MOP_OUTER_LOOP,
            mop::loop(
                MOP_INNER_LOOP,
                _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_0, p_pacr::ALL_INTF_ACTIVE, 0, 1),
                _llk_pack_fast_untilize_row_pacr_(ADDR_MOD_1, tail_intf, 1, 0)),
            TT_OP_NOP,
            TT_OP_NOP));
    }
}

inline void _llk_pack_fast_untilize_program_output_row_stride_(const std::uint32_t output_row_stride_bytes)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace fup  = hal_port::fast_untilize_pack;

    // Wider rows close the pack stream after each chunk row. Program the PAC
    // output Y stride to one full row of the row-major tensor so the ch1 output
    // counter lands at the same column in the next output row. set_packer_strides
    // owns the zero baseline, so hw_configure and the uninit below clear it.
    hal::scalar::set_low(hal::gpr<p_gpr_pack::TMP0>(), LOWER_HALFWORD(output_row_stride_bytes << fup::Ch1.Ystride.shamt(cfg::Sec::S0)));
    hal::scalar::set_high(hal::gpr<p_gpr_pack::TMP0>(), UPPER_HALFWORD(output_row_stride_bytes << fup::Ch1.Ystride.shamt(cfg::Sec::S0)));
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, fup::Ch1.Xstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());
    hal::nop::thread(); // WRCFG needs a one-instruction bubble before a consuming PACR
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim>
inline void _llk_pack_fast_untilize_init_(const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;
    namespace fup  = hal_port::fast_untilize_pack;

    static_assert(block_ct_dim >= 2 && block_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM, "BH fast untilize supports block_ct_dim 2, 3, or 4");

    hal::scalar::set_low<0x000>(hal::gpr<p_gpr_pack::DEST_OFFSET_LO + 0>());
    hal::scalar::set_low<DEST_REGISTER_HALF_SIZE>(hal::gpr<p_gpr_pack::DEST_OFFSET_HI + 0>());
    hal_port::select_packer_dest_registers<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>();

    // TTI_SETADCXX(PAC, FACE_C_DIM - 1, 0x0)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = FACE_C_DIM - 1}}.for_clients<adc::Client::Packers>()>();

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

    hal::scalar::set_low(hal::gpr<p_gpr_pack::TMP0>(), LOWER_HALFWORD(y_stride << fup::Ch0.Ystride.shamt(cfg::Sec::S0)));
    hal::scalar::set_high(hal::gpr<p_gpr_pack::TMP0>(), UPPER_HALFWORD(y_stride << fup::Ch0.Ystride.shamt(cfg::Sec::S0)));
    hal::scalar::set_low(hal::gpr<p_gpr_pack::TMP1>(), LOWER_HALFWORD(z_stride << fup::Ch0.Zstride.shamt(cfg::Sec::S0)));
    hal::scalar::set_high(hal::gpr<p_gpr_pack::TMP1>(), UPPER_HALFWORD(w_stride << fup::Ch0.Wstride.shamt(cfg::Sec::S0)));
    // Serialize the TMP GPR setup before WRCFG copies those stride values into
    // packer address-generator config. This is the same LLK SETDMAREG->WRCFG
    // barrier pattern used elsewhere; ISA documents WRCFG's GPR source and
    // STALLWAIT's Wait Gate blocking.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, fup::Ch0.Xstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());
    cfg::write<cfg::Access::TensixCfgUnit, fup::Ch0.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP1>());

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
    hal::scalar::set_low<phase_offset>(hal::gpr<p_gpr_pack::DEST_OFFSET_LO + 0>());
    hal::scalar::set_low<DEST_REGISTER_HALF_SIZE + phase_offset>(hal::gpr<p_gpr_pack::DEST_OFFSET_HI + 0>());

    hal_port::select_packer_dest_registers<FAST_UNTILIZE_INTERNAL_DST_SYNC_MODE>();
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
    ::set_dst_write_addr<false>(0);
}

// One call processes one 2/3/4-tile chunk.
// Output: one chunk's worth of RM strip starting at `address` (in 16B units).
template <std::uint32_t block_ct_dim>
inline void _llk_pack_fast_untilize_block_(const std::uint32_t address, const std::uint32_t unit_dim, std::uint32_t& prev_unit_dim)
{
    static_assert(block_ct_dim >= 2 && block_ct_dim <= FAST_UNTILIZE_MAX_UNIT_DIM, "BH fast untilize supports block_ct_dim 2, 3, or 4");
    LLK_ASSERT(unit_dim >= 2 && unit_dim <= block_ct_dim, "fast_untilize pack unit_dim must be in [2, block_ct_dim]");

    hal_port::program_packer_destination(address);

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
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();

    // Phase 2 emits bottom strip rows and closes the stream. Phase 1 already
    // programmed the MOP body; only the final PACR's Last bit changes.
    _llk_pack_fast_untilize_mop_patch_last_(unit_dim, true);
    _llk_pack_fast_untilize_select_phase_<FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET>();
    _llk_pack_fast_untilize_reset_src_counters_();
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
    _llk_pack_fast_untilize_restore_pack_counters_<false>();
}

// The packer carries the destination Y offset (ADC.Y * Ystride) relative to the
// last-programmed L1 base. On BH silicon that carried offset is only reliably
// preserved inside a ~256 KiB window; beyond it the high address bits are
// dropped and rows overwrite each other (verified on silicon: BF16 full_ct_dim
// 133/256 corrupt without a rebase, FP32 corrupts from ~ct 137). The window is
// not in the public BH ISA docs; 256 KiB is the silicon-characterized bound
// used here.
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
        hal_port::program_packer_destination(address + out_row * output_row_stride_16B);
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
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
        hal_port::program_packer_destination(address);
        _llk_pack_fast_untilize_select_phase_<FAST_UNTILIZE_PACK_TOP_STRIP_DEST_TARGET_OFFSET>();
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        _llk_pack_fast_untilize_select_phase_<FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET>();
        _llk_pack_fast_untilize_reset_src_counters_();
        hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
        _llk_pack_fast_untilize_restore_pack_counters_<true>();
        return;
    }

    const std::uint32_t runs_per_phase = FAST_UNTILIZE_PHASE_ROWS / rows_per_run;
    _llk_pack_fast_untilize_emit_phase_<FAST_UNTILIZE_PACK_TOP_STRIP_DEST_TARGET_OFFSET>(address, 0, runs_per_phase, rows_per_run, output_row_stride_16B);
    _llk_pack_fast_untilize_emit_phase_<FAST_UNTILIZE_PACK_BOTTOM_STRIP_DEST_TARGET_OFFSET>(
        address, FAST_UNTILIZE_PHASE_ROWS, runs_per_phase, rows_per_run, output_row_stride_16B);
    _llk_pack_fast_untilize_restore_pack_counters_<true>();
}

inline void _llk_pack_fast_untilize_uninit_(const std::uint32_t pack_src_format)
{
    // set_packer_strides also zeroes the channel-1 Y stride the strided path programs.
    hal_port::set_packer_strides<PackMode::Default>(pack_src_format, TILE_C_DIM);
    // init owns the X counter and sets it itself; strides are restored just above, so skip them in init.
    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        pack_src_format, FACE_R_DIM, TILE_C_DIM, FAST_UNTILIZE_NUM_FACES, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
}

} // namespace ckernel
