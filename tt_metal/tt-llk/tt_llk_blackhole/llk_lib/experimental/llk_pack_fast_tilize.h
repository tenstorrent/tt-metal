// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/experimental/llk_pack_fast_tilize.h. Same Tensix instruction sequence and
// access kinds as production, expressed through the HAL, except the HAL-GAP marked below.
//
// BH Fast-Tilize Pack — Multi-tile MOP with per-tile Last=1, zero L1 overflow.
//
// Replay buffer layout (BH = 32 entries total):
//   [0..15]:  Tile replay (16 PACRs, Last=1 on final PACR)
//   [16..19]: Address-update replay (ADDDMAREG + STALLWAIT + WRCFG + NOP)
//
// MOP: outerloop=unit_dim, innerloop=1.
//   Each outer iteration packs one tile via tile replay, then advances L1
//   address via address-update replay in end_ops. No per-tile RISC-V overhead.

#pragma once

#include <cstdint>

#include "hal/cfg.h"
#include "hal/counters/adc.h"
#include "hal/mop.h"
#include "hal/nop.h"
#include "hal/pack.h"
#include "hal/replay.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

constexpr std::uint32_t REPLAY_TILE_OFFSET = 0;
constexpr std::uint32_t REPLAY_TILE_LEN    = 16;

constexpr std::uint32_t REPLAY_ADDR_UPDATE_OFFSET = ckernel::packer::replay_buf_offset;
constexpr std::uint32_t REPLAY_ADDR_UPDATE_LEN    = 4;

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

namespace fast_tilize_pack
{

namespace adc = hal::counters::adc;

// One fast-tilize face-row PACR: TT_OP_PACR(CFG_CTXT_0, NO_ROW_PAD_ZERO, DST_ACCESS_STRIDED_MODE, addr_mod,
// ADDR_CNT_CTXT_0, 0, ALL_INTF_ACTIVE, 0, 0, NO_CTXT_CTRL, 0, last).
constexpr hal::pack::DataTransfer face_row(const std::uint8_t addr_mod, const bool last)
{
    return {
        .address_modifier = addr_mod,
        .interfaces       = p_pacr::ALL_INTF_ACTIVE,
        .dest_access      = hal::pack::DestAccess::Strided,
        .boundary         = last ? hal::pack::TileBoundary::Last : hal::pack::TileBoundary::NotLast,
    };
}

static_assert(
    face_row(ADDR_MOD_1, true).get_operation() == TT_OP_PACR(
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
                                                      1),
    "fast-tilize PACR encoding");

// One face: three PACRs with ADDR_MOD_0, then the boundary PACR (production's EMIT_FACE_PACRS).
template <std::uint8_t boundary_am, bool last>
inline __attribute__((always_inline)) void emit_face_pacrs()
{
    hal::pack::run<face_row(ADDR_MOD_0, false)>();
    hal::pack::run<face_row(ADDR_MOD_0, false)>();
    hal::pack::run<face_row(ADDR_MOD_0, false)>();
    hal::pack::run<face_row(boundary_am, last)>();
}

inline constexpr hal::replay::BufferRange TILE_REPLAY        = {REPLAY_TILE_OFFSET, REPLAY_TILE_LEN};
inline constexpr hal::replay::BufferRange ADDR_UPDATE_REPLAY = {REPLAY_ADDR_UPDATE_OFFSET, REPLAY_ADDR_UPDATE_LEN};
static_assert(hal::replay::get_operation(TILE_REPLAY) == lltt::replay_insn(REPLAY_TILE_OFFSET, REPLAY_TILE_LEN), "tile replay encoding");
static_assert(
    hal::replay::get_operation(ADDR_UPDATE_REPLAY) == lltt::replay_insn(REPLAY_ADDR_UPDATE_OFFSET, REPLAY_ADDR_UPDATE_LEN), "address-update replay encoding");

// TT_OP_ADDRCRZW(PAC, 0, 0, 1, 0, 0b0010): W_Cr += 1, W = W_Cr
inline constexpr std::uint32_t ADVANCE_W =
    adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = {.w = 1}}.for_clients<adc::Client::Packers>().operation();
// TT_OP_ADDRCRZW(PAC, 0, 0, 0, 0, 0b0011): Z = Z_Cr, W = W_Cr
inline constexpr std::uint32_t RESTORE_ZW =
    adc::AddrCounter {.action = adc::Action::AdvanceAndReset, .channel0 = {.z = 0, .w = 0}}.for_clients<adc::Client::Packers>().operation();
static_assert(ADVANCE_W == TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 1, 0, 0b0010), "ADDRCRZW W encoding");
static_assert(RESTORE_ZW == TT_OP_ADDRCRZW(p_setadc::PAC, 0, 0, 0, 0, 0b0011), "ADDRCRZW Z/W encoding");

} // namespace fast_tilize_pack
} // namespace hal_port

__attribute__((noinline)) void _llk_pack_fast_tilize_configure_addrmod_()
{
    using hal::cfg::Sec;
    // addr_mod_pack_t {.y_src = {.incr = 1}}.set(ADDR_MOD_0)
    hal_port::set_addr_mod_pack<Sec::S0, 1 /*y_src_incr*/, 0 /*y_src_clr*/, 0 /*y_src_cr*/, 0 /*z_src_incr*/, 0 /*z_src_clr*/>();
    // addr_mod_pack_t {.y_src = {.clr = 1, .cr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_1)
    hal_port::set_addr_mod_pack<Sec::S1, 0 /*y_src_incr*/, 1 /*y_src_clr*/, 1 /*y_src_cr*/, 0 /*z_src_incr*/, 1 /*z_src_clr*/>();
    // addr_mod_pack_t {.y_src = {.incr = 4, .cr = 1}, .z_src = {.clr = 1}}.set(ADDR_MOD_2)
    hal_port::set_addr_mod_pack<Sec::S2, 4 /*y_src_incr*/, 0 /*y_src_clr*/, 1 /*y_src_cr*/, 0 /*z_src_incr*/, 1 /*z_src_clr*/>();
    // addr_mod_pack_t {.y_src = {.cr = 1}, .z_src = {.incr = 1}}.set(ADDR_MOD_3)
    hal_port::set_addr_mod_pack<Sec::S3, 0 /*y_src_incr*/, 0 /*y_src_clr*/, 1 /*y_src_cr*/, 1 /*z_src_incr*/, 0 /*z_src_clr*/>();
}

__attribute__((noinline)) void _llk_pack_fast_tilize_load_replay_()
{
    namespace ftp  = hal_port::fast_tilize_pack;
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // Tile replay [0..15]: 16 PACRs, Last=1 on final.
    hal::replay::record<ftp::TILE_REPLAY>(
        []
        {
            ftp::emit_face_pacrs<ADDR_MOD_3, false>();
            ftp::emit_face_pacrs<ADDR_MOD_2, false>();
            ftp::emit_face_pacrs<ADDR_MOD_3, false>();
            ftp::emit_face_pacrs<ADDR_MOD_1, true>(); // face3: Last=1
        });

    // Address-update replay [16..19]: advance L1_Dest_addr for next tile.
    // Same pattern as BH pack-untilize (llk_pack_untilize.h:94-108).
    hal::replay::record<ftp::ADDR_UPDATE_REPLAY>(
        []
        {
            // TTI_ADDDMAREG(0, OUTPUT_ADDR, OUTPUT_ADDR, OUTPUT_ADDR_OFFSET): OUTPUT_ADDR += OUTPUT_ADDR_OFFSET
            hal::scalar::add(hal::gpr<p_gpr_pack::OUTPUT_ADDR>(), hal::gpr<p_gpr_pack::OUTPUT_ADDR_OFFSET>(), hal::gpr<p_gpr_pack::OUTPUT_ADDR>());
            sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
            cfg::write<
                cfg::Access::TensixCfgUnit,
                cfg::Thcon[cfg::Reg1].L1_Dest_addr,
                cfg::Sec::S0,
                cfg::GprTransferSize::Bits32,
                cfg::WrcfgCompletion::Deferred>(hal::gpr<p_gpr_pack::OUTPUT_ADDR>());
            hal::nop::thread();
        });
}

// Multi-tile MOP: outerloop=unit_dim tiles, each with tile replay + addr update.
// end_ops fire per outer iteration = per tile.
inline void _llk_pack_fast_tilize_mop_config_(const std::uint32_t unit_dim)
{
    namespace mop = hal::mop;
    namespace ftp = hal_port::fast_tilize_pack;

    // outerloop: one iteration per tile; innerloop: single tile body (tile replay, then advance W).
    // start_op: restore Z/W from CR shadows. W_Cr accumulates across tiles.
    // With innerloop=1: every inner iter is "last inner".
    // Tiles 0..N-2: last_inner fires (advance W).
    // Tile N-1: last_outer fires (advance W — same op, no PACR_FLUSH).
    // Both last-iteration ops default to the inner loop's last op, ADVANCE_W, which is what production sets them to.
    // end_ops: L1 address advance via replay (runs per outer iteration = per tile), then NOP.
    mop::program(mop::loop(
        unit_dim,
        ftp::RESTORE_ZW,
        mop::loop(1, hal::replay::get_operation(ftp::TILE_REPLAY), ftp::ADVANCE_W),
        hal::replay::get_operation(ftp::ADDR_UPDATE_REPLAY),
        TT_OP_NOP));
}

template <DstSync Dst, bool is_fp32_dest_acc_en = false>
__attribute__((noinline)) void _llk_pack_fast_tilize_init_(
    [[maybe_unused]] const std::uint32_t use_32bit_dest,
    const std::uint32_t pack_dst_format,
    [[maybe_unused]] const std::uint32_t unit_dim,
    [[maybe_unused]] const std::uint32_t num_faces = 4,
    const std::uint32_t pack_src_format            = static_cast<std::uint32_t>(DataFormat::Float16_b))
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;
    namespace adc  = hal::counters::adc;

    // DEST remap (remap_addrs + swizzle_32b) is set by _llk_math_fast_tilize_init_
    // on the math thread (mirrors pack_untilize_dest_init; tracked by
    // tt-metal#17132 / tt-llk#989). No action needed here, no uninit.

    if constexpr (is_fp32_dest_acc_en)
    {
        // Packer borrow for fast-tilize: the caller's pack_src_format may be fp32
        // (e.g. after mm_init for fp32 matmul output, or infer_pack_in picking fp32
        // for Bfp4_b+dest_acc). Fast-tilize needs bf16-stride stepping through DEST
        // (Read_32b=0), but per ISA fp32 src requires Read_32b=1. Reconfigure the
        // packer to a bf16-compat src — this coherently updates in_data_format
        // across SEC0/SEC1 REG1/REG8, exp_section_size for BFP dst, TILE_HEADER,
        // and dest_rd_ctrl. Then force Read_32b=0 (reconfig sets it to 1 when
        // is_fp32_dest_acc_en). Uninit mirrors by restoring caller's pack_src.
        constexpr std::uint32_t compat_src = ckernel::to_underlying(DataFormat::Float16_b);
        const std::uint32_t tile_size      = SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM * TILE_R_DIM);
        hal_port::reconfig_packer_data_format<is_fp32_dest_acc_en>(compat_src, pack_dst_format, tile_size, TILE_C_DIM, num_faces, /*partial_face=*/false);
        sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0, 0>();
    }

    // Per-tile L1 advance (used by addr-update replay's ADDDMAREG).
    const std::uint32_t tile_l1_size = GET_L1_HEADERLESS_TILE_SIZE(pack_dst_format);
    hal::scalar::set_low(hal::gpr<p_gpr_pack::OUTPUT_ADDR_OFFSET>(), LOWER_HALFWORD(tile_l1_size));

    hal::scalar::set_low<0x000>(hal::gpr<p_gpr_pack::DEST_OFFSET_LO + 0>());
    hal::scalar::set_low<DEST_REGISTER_HALF_SIZE>(hal::gpr<p_gpr_pack::DEST_OFFSET_HI + 0>());
    hal_port::select_packer_dest_registers<Dst>();

    // TTI_SETADCXX(PAC, FACE_C_DIM - 1, 0x0)
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = FACE_C_DIM - 1}}.for_clients<adc::Client::Packers>()>();

    // Stride registers are in bytes, divided by pack_src datum size at PACR execution
    // (ttsim: src_addr = byte_offset / src_element_align).
    // x_stride = datum_bytes(effective pack_src). When is_fp32_dest_acc_en we
    // overrode pack_src cfg to Float16_b above, so use bf16 datum size (2) here
    // regardless of the caller's original pack_src_format param.
    const std::uint32_t effective_src = is_fp32_dest_acc_en ? ckernel::to_underlying(DataFormat::Float16_b) : pack_src_format;
    const std::uint32_t x_stride      = (effective_src & 0x3) == ckernel::to_underlying(DataFormat::Float32)   ? 4
                                        : (effective_src & 0x3) == ckernel::to_underlying(DataFormat::Float16) ? 2
                                                                                                               : 1;
    std::uint32_t y_stride            = 64 * FACE_C_DIM * x_stride;
    std::uint32_t z_stride            = FACE_C_DIM * x_stride;
    std::uint32_t w_stride            = 2 * FACE_C_DIM * x_stride;

    constexpr auto Ch0 = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0];
    static_assert(Ch0.Xstride.addr32(cfg::Sec::S0) == PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32, "packer ch0 X/Y-stride word");
    static_assert(Ch0.Zstride.addr32(cfg::Sec::S0) == PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32, "packer ch0 Z/W-stride word");
    static_assert(Ch0.Ystride.shamt(cfg::Sec::S0) == PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, "Y-stride shift");
    static_assert(Ch0.Zstride.shamt(cfg::Sec::S0) == PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT, "Z-stride shift");
    static_assert(Ch0.Wstride.shamt(cfg::Sec::S0) == PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT, "W-stride shift");
    hal::scalar::set_low(hal::gpr<p_gpr_pack::TMP0>(), LOWER_HALFWORD(y_stride << Ch0.Ystride.shamt(cfg::Sec::S0)));
    hal::scalar::set_high(hal::gpr<p_gpr_pack::TMP0>(), UPPER_HALFWORD(y_stride << Ch0.Ystride.shamt(cfg::Sec::S0)));
    hal::scalar::set_low(hal::gpr<p_gpr_pack::TMP1>(), LOWER_HALFWORD(z_stride << Ch0.Zstride.shamt(cfg::Sec::S0)));
    hal::scalar::set_high(hal::gpr<p_gpr_pack::TMP1>(), UPPER_HALFWORD(w_stride << Ch0.Wstride.shamt(cfg::Sec::S0)));
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, Ch0.Xstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());
    cfg::write<cfg::Access::TensixCfgUnit, Ch0.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP1>());

    _llk_pack_fast_tilize_configure_addrmod_();
    _llk_pack_fast_tilize_load_replay_();
    _llk_pack_fast_tilize_mop_config_(unit_dim);
}

// ===========================================================================
// Row-scoped pack helpers.
//
// Hoist program_packer_destination() and the bulk of counter resets out of
// the per-chunk loop so they are paid once per row instead of once per chunk.
//
// The fast-tilize replay already advances OUTPUT_ADDR per tile via end_ops
// (ADDDMAREG + STALLWAIT + WRCFG + NOP), so after each chunk MOP OUTPUT_ADDR
// is already at the correct position for the next chunk.
//
// Counter state after every complete tile (confirmed via ISA docs + ttsim):
//   X: wraps to 0 (SETADCXX end=FACE_C_DIM-1, 16 PACRs per tile)
//   Y: 0 (ADDR_MOD_1 last PACR: y_src={clr:1,cr:1})
//   Z: 0 (ADDR_MOD_1 last PACR: z_src={clr:1})
//   W: accumulates (last_inner/last_outer ADDRCRZW W+=1 per tile)
//
// row_begin: reset X/Y, program destination — once per row.
// row_chunk: reset W only (X/Y/Z are naturally 0 at chunk boundaries) + MOP.
// row_end:   no-op (placeholder for future cleanup hooks).
// ===========================================================================

inline void _llk_pack_fast_tilize_row_begin_(const std::uint32_t address)
{
    namespace adc = hal::counters::adc;
    // Reset X and Y once for the whole row. After each tile replay:
    //   X wraps to 0 (SETADCXX end=FACE_C_DIM-1, 16 PACRs per tile → X=0).
    //   Y=0 (ADDR_MOD_1 fires last PACR of every tile: y_src={clr:1,cr:1}).
    // So these counters are naturally 0 at every chunk boundary after row_begin.
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}}.for_clients<adc::Client::Packers>()>(); // SETADCXY(PAC, mask 0b0011)
    hal_port::program_packer_destination(address);
}

inline void _llk_pack_fast_tilize_row_chunk_(
    [[maybe_unused]] const std::uint32_t tile_index, [[maybe_unused]] const std::uint32_t unit_dim, [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    namespace adc = hal::counters::adc;
    // Only W needs resetting per chunk — it accumulates via W_Cr across tiles
    // (last_inner/last_outer each fire ADDRCRZW W+=1, and start_op restores W=W_Cr).
    // Z=0 naturally after every tile (ADDR_MOD_1: z_src={clr:1}); no reset needed.
    // X=0 and Y=0 naturally at chunk boundaries (see row_begin comment).
    hal_port::adc_set_masked<adc::AddrCounter {.channel0 = {.w = 0}}.for_clients<adc::Client::Packers>()>(); // reset W only: SETADCZW(PAC, mask 0b0010)
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}

inline void _llk_pack_fast_tilize_row_end_()
{
    // No-op.
}

// Reprogram MOP outerloop for a different unit_dim.
inline void _llk_pack_fast_tilize_reinit_unit_dim_([[maybe_unused]] const std::uint32_t pack_dst_format, const std::uint32_t new_unit_dim)
{
    _llk_pack_fast_tilize_mop_config_(new_unit_dim);
}

// One call = one row-chunk (one MOP run). num_units loop removed; block
// height is always 1 and chunk iteration is in the caller.
inline void _llk_pack_fast_tilize_block_(
    [[maybe_unused]] const std::uint32_t tile_index,
    const std::uint32_t address,
    [[maybe_unused]] const std::uint32_t unit_dim,
    [[maybe_unused]] const std::uint32_t num_faces = 4)
{
    namespace adc = hal::counters::adc;
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0}}.for_clients<adc::Client::Packers>()>(); // SETADCXY(PAC, mask 0b0011)
    hal_port::program_packer_destination(address);
    // Reset Z/W counters. MOP start_op restores from CR shadows.
    // W_Cr accumulates across tiles within the unit via ADDRCRZW in loop_op1.
    adc::set<adc::AddrCounter {.channel0 = {.z = 0, .w = 0}}.for_clients<adc::Client::Packers>()>(); // SETADCZW(PAC, mask 0b0011)
    hal::mop::Runner<hal::mop::MopTemplate::Template1>::run();
}

template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_fast_tilize_uninit_(
    const std::uint32_t pack_dst_format,
    [[maybe_unused]] const std::uint32_t face_r_dim,
    [[maybe_unused]] const std::uint32_t num_faces,
    const std::uint32_t pack_src_format = static_cast<std::uint32_t>(DataFormat::Float16_b))
{
    if constexpr (is_fp32_dest_acc_en)
    {
        // Mirror of init: restore caller's pack_src_format via reconfig, which also
        // sets Read_32b correctly (=1 for fp32/dest_acc per cpack_common logic).
        const std::uint32_t tile_size = SCALE_DATUM_SIZE(pack_dst_format, TILE_C_DIM * TILE_R_DIM);
        hal_port::reconfig_packer_data_format<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, TILE_C_DIM, num_faces, /*partial_face=*/false);
    }
    // DEST remap is NOT cleared here — set/owned by the math thread (see init comment).
    // BH-specific: restore strides modified by fast-tilize init (WH doesn't modify them).
    // Note: set_packer_strides's first param is semantically pack_src_format.
    hal_port::set_packer_strides<PackMode::Default>(pack_src_format, TILE_C_DIM);
    // Restore X counter, addr_mods, and MOP via _llk_pack_init_ (aligned with WH approach); init owns
    // the X counter and sets it itself. Strides are restored just above, so skip them in init.
    _llk_pack_init_<PackMode::Default, false /* zero_output */, false /* skip_addrmod_config */, true /* skip_packer_strides */>(
        pack_src_format, FACE_R_DIM, TILE_C_DIM, 4 /* num_faces */, 1 /* num_tiles */, false /* skip_bh_tilize_workaround */);
}
