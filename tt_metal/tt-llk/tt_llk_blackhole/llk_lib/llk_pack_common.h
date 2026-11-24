// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// HAL port of tt_llk_blackhole/llk_lib/llk_pack_common.h and the pack helpers of common/inc/cpack_common.h.
// Same Tensix instruction sequence and access kinds as production; every instruction and register access goes
// through the HAL except the listed HAL-GAPs (tensix_sync, RISC regfile[] stores).
// The ported cpack_common.h helpers live in namespace hal_port under their production names; the production
// ckernel::packer versions stay visible for the production-copy headers in tests/hal_llk that call them.

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_mutex_guard.h" // not used here: kept for production-copy headers that include llk_pack_common.h
#include "ckernel_ops.h"
#include "cpack_common.h" // types, constants and pure helpers only; no production instruction path is called
#include "hal/cfg.h"
#include "hal/counters/adc.h"
#include "hal/dst.h"
#include "hal/nop.h"
#include "hal/scalar.h"
#include "hal/sync.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "tensor_shape.h"

using namespace ckernel;
using namespace ckernel::packer;

namespace hal_port
{

namespace cfg  = hal::cfg;
namespace sync = hal::sync;
namespace adc  = hal::counters::adc;

// Place a value into field F of its CFG word, like the (v << SHAMT) & MASK of the raw defines.
template <const cfg::Field& F, cfg::Sec S = cfg::Sec::S0>
constexpr std::uint32_t place(const std::uint32_t value)
{
    return (value << F.shamt(S)) & F.mask(S);
}

// SETADC* sites of a thread that borrows the pack ADCs are serialized with mutex::THREAD2_ADC (M3).
template <bool enable = true>
class [[nodiscard]] AdcMutexGuard final
{
public:
    static_assert(ckernel::mutex::THREAD2_ADC == 3, "THREAD2_ADC must map to hal::sync::Mutex::M3");

    AdcMutexGuard() noexcept
    {
        if constexpr (enable)
        {
            sync::mutex::acquire<sync::Mutex::M3>();
        }
    }

    ~AdcMutexGuard()
    {
        if constexpr (enable)
        {
            sync::mutex::release<sync::Mutex::M3>();
        }
    }

    AdcMutexGuard(const AdcMutexGuard&)            = delete;
    AdcMutexGuard& operator=(const AdcMutexGuard&) = delete;
    AdcMutexGuard(AdcMutexGuard&&)                 = delete;
    AdcMutexGuard& operator=(AdcMutexGuard&&)      = delete;
};

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
// SETADCXY(PAC, mask 0b1011) + SETADCZW(PAC, mask 0b1111)
inline void packer_addr_counter_init()
{
    adc::set<adc::AddrCounter {.channel0 = {.x = 0, .y = 0, .z = 0, .w = 0}, .channel1 = {.y = 0, .z = 0, .w = 0}}.for_clients<adc::Client::Packers>()>();
}

template <PackMode pack_mode = PackMode::Default>
inline void set_packer_strides(const std::uint32_t pack_src_format, const std::uint32_t tile_c_dim)
{
    constexpr auto Ch0 = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg0];
    constexpr auto Ch1 = cfg::Packer[0].AddrCtrl[cfg::PackerReg::Reg1];

    std::uint32_t x_stride = datum_size_in_bytes(pack_src_format);
    std::uint32_t y_stride = FACE_C_DIM * x_stride;
    std::uint32_t w_stride = TILE_NUM_FACES * FACE_C_DIM * FACE_R_DIM * x_stride;

    // Untilize mode has 2 packer interfaces active, so z counter needs to jump by 2
    // faces, since z counter is only 1 bit (can't be programmed to inc by 2)
    const std::uint32_t z_stride = ((pack_mode != PackMode::Default) && (tile_c_dim == TILE_C_DIM)) ? 2 * FACE_R_DIM * y_stride : FACE_R_DIM * y_stride;

    // x-stride not used! The low half of the Y-stride word is a compile-time zero (compile-time form of the SETDMAREG).
    static_assert(Ch0.Ystride.shamt(cfg::Sec::S0) == 16, "Y stride must occupy the high half of its CFG word");
    hal::scalar::set_low<0 /*Value*/>(hal::gpr<p_gpr_pack::TMP0>());
    hal::scalar::set_high(hal::gpr<p_gpr_pack::TMP0>(), place<Ch0.Ystride>(y_stride) >> 16);
    hal::scalar::set(hal::gpr<p_gpr_pack::TMP1>(), place<Ch0.Zstride>(z_stride) | place<Ch0.Wstride>(w_stride));
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, Ch0.Xstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());
    cfg::write<cfg::Access::TensixCfgUnit, Ch0.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP1>());
    // Channel-1 (L1 output) strides are zero for every pack mode: output placement comes from
    // L1_Dest_addr and the packer's write pointer. Only the fast-untilize strided pack sets a
    // channel-1 (Y) stride, in its own init.
    cfg::write<cfg::Access::TensixCfgUnit, Ch1.Xstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(hal::gpr<p_gpr::ZERO>());
    cfg::write<cfg::Access::TensixCfgUnit, Ch1.Zstride, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(hal::gpr<p_gpr::ZERO>());
    hal::nop::thread();
    hal::nop::thread();
}

// Compile-time form of production's reconfigure_packer_l1_acc(pack_l1_acc); the only caller passes 0.
template <std::uint32_t pack_l1_acc>
inline void reconfigure_packer_l1_acc()
{
    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    // Stall to avoid clobbering current packer configuration, then enable/disable L1 accumulation
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, R1.Pack_L1_Acc, cfg::Sec::S0, pack_l1_acc>();
    // Accumulating writes are bank atomics; with the fast tile end drain the lockstep packer ports can settle into a slow rate.
#if !defined(TT_METAL_TTSIM) // ttsim does not model PACK_GLOBAL_CFG_CTL and aborts on any access to it
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, cfg::Sec::S0, pack_l1_acc ? 1 : 0>();
#endif
}

// Production reconfigure_packer_l1_acc(pack_l1_acc) with a runtime value (_llk_pack_reconfig_l1_acc_).
inline void reconfigure_packer_l1_acc(const std::uint32_t pack_l1_acc)
{
    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    // Stall to avoid clobbering current packer configuration, then enable/disable L1 accumulation
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, R1.Pack_L1_Acc, cfg::Sec::S0>(pack_l1_acc);
    // Accumulating writes are bank atomics; with the fast tile end drain the lockstep packer ports can settle into a slow rate.
#if !defined(TT_METAL_TTSIM) // ttsim does not model PACK_GLOBAL_CFG_CTL and aborts on any access to it
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PackGlobalCfgCtl::pack_disable_fast_tile_end_drain, cfg::Sec::S0>(pack_l1_acc ? 1 : 0);
#endif
}

/**
 * @brief Configure packer exponent thresholding for the current destination and output formats.
 *
 * We use packer exponent thresholding to zero out datums that are too small to be repesentable
 * This is specifically done in FP32 (Dest) -> BFPxA (L1) case.
 *
 * This implies that exponent thresholding should be reconfigured whenever packer formats change.
 *
 * @tparam is_fp32_dest_acc_en True when Dest register is FP32.
 * @param pack_dst_format Pack output data format.
 */
template <bool is_fp32_dest_acc_en>
inline void reconfigure_exp_threshold(const std::uint32_t pack_dst_format)
{
    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    if constexpr (is_fp32_dest_acc_en)
    {
        bool enable             = false;
        std::uint32_t threshold = 0;

        // Workaround for HW bug: tenstorrent/budabackend#1394
        if (IS_BFP_A_FORMAT(pack_dst_format))
        {
            // After early format conversion the exponent is EXP_B; packing to BFP-A uses EXP_A.
            // Zero out values too small to represent in EXP_A (EXP_B < 113).
            enable    = true;
            threshold = 113;
        }

        // Exp_threshold_en and Exp_threshold share one CFG word: one grouped RMW.
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<R1.Exp_threshold_en, cfg::Sec::S0>(enable ? 1 : 0), cfg::set<R1.Exp_threshold, cfg::Sec::S0>(threshold));
    }
    else
    {
        // Without an FP32 Dest the enable and the threshold are compile-time zero: immediate form.
        cfg::write<cfg::Access::TensixCfgUnit>(cfg::set<R1.Exp_threshold_en, cfg::Sec::S0, 0>(), cfg::set<R1.Exp_threshold, cfg::Sec::S0, 0>());
    }
}

template <bool is_fp32_dest_acc_en>
inline void set_packer_config(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4, const bool partial_face = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);
    const std::uint32_t pack_output_dst_format = masked_data_format(pack_dst_format);
    // Gasket converts Float16_b -> Float16 before the packer, so hardware in_data_format must be Float16 for Fp8 output.
    const std::uint32_t pack_hw_src_format =
        ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)) ? to_underlying(DataFormat::Float16) : pack_output_src_format;

    // Packer config words 0 and 2 (word 1 is L1_Dest_addr, set up on every pack)
    const std::uint32_t exp_section_size =
        ((pack_output_dst_format == to_underlying(DataFormat::Lf8)) || (pack_output_dst_format == to_underlying(DataFormat::Int8)))
            ? 0
            : (partial_face ? 1 : num_faces); // set to num_faces as exp section size is not used for non-bfp formats except for lf8/int8

    const std::uint32_t config_word0 = place<R1.Exp_section_size>(exp_section_size);
    const std::uint32_t config_word2 =
        place<R1.Disable_zero_compress>(1 /*uncompress*/) | place<R1.Out_data_format>(pack_output_dst_format) | place<R1.In_data_format>(pack_hw_src_format);

    reconfigure_exp_threshold<is_fp32_dest_acc_en>(pack_output_dst_format);

    cfg::write<cfg::Access::MMIO, R1.Row_start_section_size, cfg::Sec::S0, 1 /*Count*/>(std::array<std::uint32_t, 1> {config_word0});
    cfg::write<cfg::Access::MMIO, R1.Disable_zero_compress, cfg::Sec::S0, 1 /*Count*/>(std::array<std::uint32_t, 1> {config_word2});

    // Reset L1 accumulation flag
    reconfigure_packer_l1_acc<0>();

    const bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                               pack_src_format == to_underlying(DataFormat::Float32);
    const bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);

    const bool read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    const bool read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;
    const bool read_unsigned = pack_dst_format == to_underlying(DataFormat::UInt8);

    // Round to 10 bit mantissa from fp32 dest
    const bool round_10b_mant = (is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16))) ||
                                (!IS_A_FORMAT(pack_src_format) && (pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3));

    const std::uint32_t dest_rd_ctrl_word = place<cfg::PckDestRdCtrl::Read_32b_data>(read_32b_data) | place<cfg::PckDestRdCtrl::Read_unsigned>(read_unsigned) |
                                            place<cfg::PckDestRdCtrl::Read_int8>(read_int8) | place<cfg::PckDestRdCtrl::Round_10b_mant>(round_10b_mant);
    cfg::write<cfg::Access::MMIO, cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0, 1 /*Count*/>(std::array<std::uint32_t, 1> {dest_rd_ctrl_word});

    // Save to GPR for quick data format reconfig
    // HAL-GAP: RISC MMIO regfile[] store (and the sync read-back) has no HAL API.
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP] = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    sync_regfile_write(p_gpr_pack::EXP0_SEC_SIZE_BFP);
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void configure_pack(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    [[maybe_unused]] const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t tile_c_dim                  = TILE_C_DIM,
    const std::uint32_t num_faces                   = 4,
    const bool partial_face                         = false,
    const std::uint32_t relu_config                 = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(static_cast<DataFormat>(pack_src_format & 0xF), static_cast<DataFormat>(pack_dst_format & 0xF)),
        "Unsupported packer to L1 conversion.");

    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    // Drain the packer before any of the configuration below is rewritten, whatever the formats. The raw
    // cfg[] stores in this function and in set_packer_config are RISC MMIO, which no STALLWAIT can hold
    // back: the STALLWAIT only keeps new packer instructions out, and tensix_sync() blocks this RISC until
    // the packer has gone idle, which is what orders those stores behind a pack still in flight. Without
    // it the pack samples the new format/width mid-tile.
    sync::wait::stall<sync::StallTarget::Pack, sync::StallCondition::PackerIdle>();
    ckernel::tensix_sync(); // HAL-GAP: blocking RISC store to pc_buf has no HAL API (HAL's own atomic.h calls ckernel::tensix_sync directly)

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);

    set_packer_strides<pack_mode>(pack_src_format, tile_c_dim);

    // NOTE: the packer X (datum) counter (SETADCXX PAC) is intentionally NOT programmed here. It is
    // owned by _llk_pack_init_, which runs after hw-configure and sets its own value per the
    // "inits own SETADCXX" contract.

    sync::mutex::acquire<sync::Mutex::M0>(); // mutex::REG_RMW

    // Set Fp8 E4M3 mode for packer
    const bool is_fp8_e4m3 = (pack_dst_format & 0x1F) == static_cast<std::uint32_t>(DataFormat::Fp8_e4m3);
    cfg::write<cfg::Access::TensixCfgUnit, R1.Pac_LF8_4b_exp, cfg::Sec::S0>(is_fp8_e4m3 ? 1 : 0);

    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg2::Dstacc, cfg::Sec::S0>(
        is_fp8_e4m3 ? to_underlying(DataFormat::Float16) : pack_output_src_format);

    // Establish the no-override baseline for the Dstacc ALU format-select fields (ADDR32=0), consumed
    // by the packer: clear Dstacc_val/override so the base REG2_Dstacc format programmed above is
    // used. The SrcA/SrcB override fields in the same word are owned by the math thread
    // (_llk_math_hw_configure_); the two writers touch disjoint bits and rely on per-byte RMWCIB
    // atomicity, so this write does not depend on the surrounding REG_RMW mutex for cross-thread safety.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::AluFormatSpecReg::Dstacc_val, cfg::Sec::S0, 0>(), cfg::set<cfg::AluFormatSpecReg::Dstacc_override, cfg::Sec::S0, 0>());

    // Config RELU
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(relu_config & 0xf),
        cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>((relu_config >> 16) & 0xffff));

    sync::mutex::release<sync::Mutex::M0>(); // mutex::REG_RMW

    set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);

    // Number of reads per face used for resetting tile position generator for edge masks.
    // pack_counters word: only pack_reads_per_xy_plane = 1 (disable auto last generation)
    for (std::uint32_t i = 0; i < NUM_PACKERS; i++)
    {
        static_assert(NUM_PACKERS == 1, "single packer: one PACK_COUNTERS word");
        cfg::write<cfg::Access::MMIO, cfg::PackCounters::pack_per_xy_plane, cfg::Sec::S0, 1 /*Count*/>(
            std::array<std::uint32_t, 1> {place<cfg::PackCounters::pack_reads_per_xy_plane>(1)});
    }

    cfg::write<cfg::Access::MMIO, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, 1 /*Count*/>(
        std::array<std::uint32_t, 1> {place<cfg::PckEdgeOffsetSec0::mask>(0xffff)});
    cfg::write<cfg::Access::MMIO, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, 1 /*Count*/>(
        std::array<std::uint32_t, 1> {0}); // All packers use row set mapping 0, edge offset 0 mask
    cfg::write<cfg::Access::MMIO, cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 1 /*Count*/>(std::array<std::uint32_t, 1> {0});

    // HAL-GAP: RISC MMIO regfile[] stores (and the sync read-back) have no HAL API.
    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
    sync_regfile_write(p_gpr_pack::TILE_HEADER + 3);
}

/**
 * Validates that the packer pack_reads_per_xy_plane counter matches the expected value (assert-only readback).
 * Intended to be called from reconfig paths to assert the invariant: between non-reduce operations the counter is 1;
 * the reduce layer owns transitions to FACE_R_DIM via `_llk_pack_reduce_mask_config_` and back to 1 via
 * `_llk_pack_reduce_mask_clear_`.
 *
 * @param expected   Expected value of pack_reads_per_xy_plane
 * @param nop_count  Number of nop operations to ensure configuration writes complete (default 10)
 * @return true if the packer counter matches expected, false otherwise.
 */
inline __attribute__((noinline)) bool is_pack_reads_per_xy_plane(const std::uint32_t expected, const std::uint32_t nop_count = 10)
{
    // Ensure configuration writes complete before reading back
    ckernel::tensix_sync(); // HAL-GAP: blocking RISC store to pc_buf has no HAL API
    for (std::uint32_t i = 0; i < nop_count; i++)
    {
        asm volatile("nop");
    }

    static_assert(NUM_PACKERS == 1, "NUM_PACKERS must be 1");
    return cfg::read<cfg::Access::MMIO, cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0>() == expected;
}

template <bool is_fp32_dest_acc_en>
__attribute__((noinline)) inline void reconfig_packer_data_format(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t tile_c_dim,
    const std::uint32_t num_faces,
    const bool partial_face)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    constexpr auto R1 = cfg::Thcon[cfg::Reg1];

    const std::uint32_t pack_output_src_format = masked_data_format(pack_src_format);
    const std::uint32_t pack_output_dst_format = masked_data_format(pack_dst_format);
    // Gasket converts Float16_b -> Float16 before the packer, so hardware in_data_format must be Float16 for Fp8 output.
    const std::uint32_t pack_hw_src_format =
        ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3)) ? to_underlying(DataFormat::Float16) : pack_output_src_format;

    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(static_cast<DataFormat>(pack_output_src_format), static_cast<DataFormat>(pack_output_dst_format)),
        "Unsupported packer to L1 conversion.");

    // Configure packers: only word[2][15:0] (uncompress, out/in data format) changes
    const std::uint32_t config_word2 =
        place<R1.Disable_zero_compress>(1 /*uncompress*/) | place<R1.Out_data_format>(pack_output_dst_format) | place<R1.In_data_format>(pack_hw_src_format);
    hal::scalar::set_low(hal::gpr<p_gpr_pack::TMP_LO>(), LOWER_HALFWORD(config_word2));
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle | sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, R1.Disable_zero_compress, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP_LO>());

    LLK_ASSERT(
        is_pack_reads_per_xy_plane(1),
        "reconfig_packer_data_format: pack_reads_per_xy_plane counter must be 1 before packer reconfig "
        "(invariant violated; reduce mask should have been cleared via _llk_pack_reduce_mask_clear_). Please uncomment "
        "DEVICE_PRINT #2111 for debugging.");

    const bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                               pack_src_format == to_underlying(DataFormat::Float32);
    const bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);

    const bool read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    const bool read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;
    const bool read_unsigned = pack_dst_format == to_underlying(DataFormat::UInt8);
    // Round to 10 bit mantissa from fp32 dest
    const bool round_10b_mant = (is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16))) ||
                                (!IS_A_FORMAT(pack_src_format) && (pack_dst_format & 0x1F) == to_underlying(DataFormat::Fp8_e4m3));

    // Read_int8 is what makes an Int8/UInt8 pack read 16-bit Dest datums as 8-bit integers, so a
    // reconfig onto or away from Int8/UInt8 must write it like the full configure does: the RMW
    // covers every dest_rd_ctrl field (one grouped RMWCIB, as production's combined mask).
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::PckDestRdCtrl::Read_32b_data, cfg::Sec::S0>(read_32b_data),
        cfg::set<cfg::PckDestRdCtrl::Read_unsigned, cfg::Sec::S0>(read_unsigned),
        cfg::set<cfg::PckDestRdCtrl::Read_int8, cfg::Sec::S0>(read_int8),
        cfg::set<cfg::PckDestRdCtrl::Round_10b_mant, cfg::Sec::S0>(round_10b_mant));

    if (IS_BFP_FORMAT(pack_output_dst_format))
    {
        cfg::write<cfg::Access::TensixCfgUnit, R1.Exp_section_size, cfg::Sec::S0>(partial_face ? 1 : num_faces);
    }
    else if ((pack_output_dst_format == to_underlying(DataFormat::Lf8)) || (pack_output_dst_format == to_underlying(DataFormat::Int8)))
    {
        cfg::write<cfg::Access::TensixCfgUnit, R1.Row_start_section_size, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr<p_gpr::ZERO>());
    }

    const bool is_fp8_e4m3 = (pack_dst_format & 0x1F) == static_cast<DataFormatType>(DataFormat::Fp8_e4m3);
    cfg::write<cfg::Access::TensixCfgUnit, R1.Pac_LF8_4b_exp, cfg::Sec::S0>(is_fp8_e4m3 ? 1 : 0);

    hal::scalar::set_low(hal::gpr<p_gpr_pack::TILE_HEADER>(), LOWER_HALFWORD(tile_size));

    reconfigure_exp_threshold<is_fp32_dest_acc_en>(pack_output_dst_format);

    cfg::write<cfg::Access::TensixCfgUnit, cfg::AluFormatSpecReg2::Dstacc, cfg::Sec::S0>(
        is_fp8_e4m3 ? to_underlying(DataFormat::Float16) : pack_output_src_format);

    // Set packer strides
    set_packer_strides<PackMode::Default>(pack_output_src_format, tile_c_dim);

    // Program the packer X (datum) counter. _llk_pack_init_ owns this counter ("inits own SETADCXX"),
    // but reconfig also programs it here: a reconfig is not always followed by an init, and on Blackhole
    // the value is a mode-independent constant (single row, FACE_C_DIM - 1), so it is safe to set here.
    adc::set<adc::AddrCounter {.channel0 = {.x = 0}, .channel1 = {.x = FACE_C_DIM - 1}}.for_clients<adc::Client::Packers>()>(); // SETADCXX(PAC, FACE_C_DIM - 1,
                                                                                                                                // 0)
}

inline std::uint8_t get_packer_dest_offset_index()
{
    return (dest_offset_id ? p_gpr_pack::DEST_OFFSET_HI : p_gpr_pack::DEST_OFFSET_LO);
}

inline std::uint32_t get_packer_dest_offset()
{
    return (dest_offset_id ? DEST_REGISTER_HALF_SIZE : 0x0);
}

inline void flip_packer_dest_offset_id()
{
    dest_offset_id = 1 - dest_offset_id;
}

inline void reset_dest_offset_id()
{
    dest_offset_id = 0;
}

// Flip packer dest register offset to 0 or DEST_REGISTER_HALF_SIZE
// flip-flopping between two halves
template <DstSync Dst>
inline void select_packer_dest_registers()
{
    if constexpr (Dst == DstSync::SyncFull)
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr<p_gpr_pack::DEST_OFFSET_LO>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::DestTargetRegCfgPack::Offset, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr(get_packer_dest_offset_index()));
    }
    hal::nop::scalar();
    hal::nop::scalar();
}

// Program packer destination addresses from GPRs
inline void program_packer_destination(const std::uint32_t addr)
{
    LLK_ASSERT(is_valid_L1_address(addr), "L1 address must be in valid L1 memory region");
    // No packer-drain STALLWAIT is needed before reprogramming the destination: the pack thread issues its
    // instruction stream in order, so each tile's PACR has already started -- and latched L1_Dest_addr,
    // sampled at PACR start -- before the next call's WRCFG reprograms it, and the Last=1 PACR that ends each
    // pack MOP drains the packer and forces the next pack to re-sample L1_Dest_addr. The STALLWAIT below is
    // only the GPR-producer fence: it ensures the SETDMAREG write to OUTPUT_ADDR retires before WRCFG reads it.
    // The address halves are computed once: bit 31 is set in the high half for the WRCFG, then cleared again.
    constexpr auto output_addr  = hal::gpr<p_gpr_pack::OUTPUT_ADDR>();
    const std::uint32_t addr_hi = UPPER_HALFWORD(addr);
    hal::scalar::set_low(output_addr, LOWER_HALFWORD(addr));
    hal::scalar::set_high(output_addr, addr_hi | (1u << 15));

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::Thcon[cfg::Reg1].L1_Dest_addr, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        output_addr);

    hal::scalar::set_high(output_addr, addr_hi);
    hal::nop::scalar(); // One NOP should be enough for WRCFG due to SETDMAREG above.
}

} // namespace hal_port

/**
 * @brief Select which destination-register tile the packer reads from.
 *
 * Sets the packer CH0 W counter to tile_index, which addresses the tile within the destination
 * register that subsequent PACR instructions pack out.
 *
 * @tparam mutex_ADC: When true, serialize the SETADC issue against mutex::THREAD2_ADC. Needed only when
 *         another thread borrows the pack thread's ADCs; forwarded from @ref _llk_pack_.
 * @param tile_index: Index of the source tile in the destination register.
 */
template <bool mutex_ADC = false>
inline void set_dst_write_addr(const std::uint32_t tile_index)
{
    namespace adc                  = hal::counters::adc;
    constexpr std::uint8_t packers = adc::AddrCounter {}.for_clients<adc::Client::Packers>().clients;

    hal_port::AdcMutexGuard<mutex_ADC> guard;
    // SETADC(PAC, CH_0, SET_W, tile_index), encoded with the SETADC encoder adc::set(AddrCounter) itself selects for a
    // runtime singleton. The public runtime adc::set() plans over a four-field group, and inside a caller's unrolled
    // tile loop (custom_mm CT_DIM == 7) that stopped GCC from hoisting the per-tile SETDMAREG words of
    // program_packer_destination: it re-derived and spilled them every tile. Same instruction either way.
    // HAL-GAP: no public runtime encoder for a single address counter; uses hal::counters::adc::detail::single.
    TT_INSN(adc::detail::single(packers, true /*zw*/, 1 /*channel 0, W*/, tile_index));
}

/**
 * @brief Stall the packer until the math thread has produced data to pack.
 *
 * Waits on the MATH_PACK semaphore so the packer does not run ahead of the math result.
 */
// wait until math is done and has produced something to pack
inline void _llk_packer_wait_for_math_done_()
{
    // The mask must cover STALL_SYNC as well as STALL_TDMA: the Wait Gate only blocks classes named in it, so with a
    // TDMA-only mask the Sync-class ATGETM of a mutexed _llk_pack_ slips past this unmet wait and takes
    // mutex::THREAD2_ADC, the SETADC behind it blocks head-of-line, and the ATRELM is stranded behind that. The packer
    // then waits on MATH_PACK holding the mutex the unpack thread needs every iteration: deadlock. Naming the Sync
    // class costs nothing when unmutexed.
    hal::sync::wait::semaphore<
        hal::sync::StallTarget::Tdma | hal::sync::StallTarget::Sync,
        hal::sync::SemaphoreCondition::WhileZero,
        static_cast<hal::sync::Semaphore>(semaphore::MATH_PACK)>();
}

/**
 * @brief Signal the MATH_PACK semaphore (decrement via SEMGET) to release the math thread.
 *
 * Tells math it may overwrite the destination register again now that the packer has consumed it.
 *
 * @tparam WaitRes: p_stall resource mask to stall on before signalling (e.g. p_stall::NONE, p_stall::PACK); default p_stall::NONE issues no stall.
 */
// Tell math that it can write again
template <std::uint32_t WaitRes = p_stall::NONE>
inline void _llk_packer_set_math_semaphore_()
{
    if constexpr (WaitRes != p_stall::NONE)
    {
        hal::sync::wait::stall<hal::sync::StallTarget::Sync, static_cast<hal::sync::StallCondition>(WaitRes)>();
    }
    hal::sync::semaphore::get<static_cast<hal::sync::Semaphore>(semaphore::MATH_PACK)>(); // Indicate that packer is done and header is written into L1
}

/**
 * @brief Finish a destination-register section: wait for pack, optionally clear dest, and release math.
 *
 * Waits until the pack completes, then signals the MATH_PACK semaphore. If clear_dest is true,
 * first zeroes the just-packed dest region (all of dest for SyncFull, the active half for SyncHalf).
 * For SyncHalf it also flips the dest-offset id and re-selects the packer dest registers so the
 * next half can be packed.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam clear_dest: Clear on release (default). If false, MATH must clear after acquiring each section.
 */
// Wait for all writes to complete in L1 (header + data)
// Tell math it can write again
// Clear dest only when clear_dest is true
template <DstSync Dst, bool is_fp32_dest_acc_en, bool clear_dest = true>
inline void _llk_pack_dest_section_done_()
{
    namespace dst  = hal::dst;
    namespace sync = hal::sync;

    constexpr dst::DestWidth width = is_fp32_dest_acc_en ? dst::DestWidth::Bits32 : dst::DestWidth::Bits16;

    if constexpr (clear_dest)
    {
        sync::wait::stall<sync::StallTarget::Math, sync::StallCondition::PackerIdle>(); // wait for pack to finish

        if constexpr (Dst == DstSync::SyncFull)
        {
            // ADDR_MOD_1: the hardware applies ZEROACC's ADDR_MOD in every mode
            dst::run<dst::Zero {.scope = dst::ZeroScope::All, .width = width, .address_mode = ADDR_MOD_1}>();
        }
        else
        {
            static_assert(Dst == DstSync::SyncHalf);
            dst::run(dst::Zero {.scope = dst::ZeroScope::Half, .index = dest_offset_id % 2, .width = width, .address_mode = ADDR_MOD_1});
        }

        // Tell math that it can write again
        _llk_packer_set_math_semaphore_<p_stall::NONE>();
    }
    else
    {
        // Without ZEROACC, block SEMGET itself until PACK has finished reading DST.
        _llk_packer_set_math_semaphore_<p_stall::PACK>();
    }

    if constexpr (Dst == DstSync::SyncHalf)
    {
        hal_port::flip_packer_dest_offset_id();
        hal_port::select_packer_dest_registers<Dst>();
    }
}

/**
 * @brief Initialize the packer destination-offset GPRs and select the dest registers.
 *
 * Programs the low/high dest-offset GPRs for row-major order and selects the packer destination
 * registers for the chosen sync mode.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 */
template <DstSync Dst>
inline void _llk_init_packer_dest_offset_registers_()
{
    namespace sync = hal::sync;

    sync::wait::stall<sync::StallTarget::Tdma | sync::StallTarget::Thcon, sync::StallCondition::PackerIdle>(); // wait for pack to finish

    // RowMajor order
    hal::scalar::set_low<0 /*Value*/>(hal::gpr<p_gpr_pack::DEST_OFFSET_LO>());
    hal::scalar::set_low<DEST_REGISTER_HALF_SIZE>(hal::gpr<p_gpr_pack::DEST_OFFSET_HI>());

    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::ScalarIdle>();
    hal_port::select_packer_dest_registers<Dst>();
}

/**
 * @brief Initialize packer destination state, at kernel start or as a per-block re-init.
 *
 * Resets the dest-offset id, programs the packer dest-offset registers, initializes the packer address
 * counter, and resets the tile dest pointer. Everything it issues is a Tensix instruction ordered behind
 * the packer by the in-stream wait in @ref _llk_init_packer_dest_offset_registers_.
 *
 * @note Does not drain this RISC (no tensix_sync). A RISC MMIO packer-config write that must follow a
 *       pack still in flight needs its own drain; @ref _llk_pack_hw_configure_ provides one.
 * @note Pair with @ref _llk_math_pack_sync_init_ on the math thread, which re-seeds the MATH_PACK handshake.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @ref _llk_init_packer_dest_offset_registers_ performs the dest-offset register setup.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en>
inline void _llk_pack_dest_init_()
{
    hal_port::reset_dest_offset_id();
    _llk_init_packer_dest_offset_registers_<Dst>();
    hal_port::packer_addr_counter_init();
    pack_sync_tile_dst_ptr = 0;
}

namespace hal_port
{
// Whole-word descriptor for TILE_FACE_SET_MAPPING_0 (production writes all 32 bits with one cfg_reg_rmw_tensix).
// HAL-GAP: hal::cfg has no whole-word (Raw) field for the face-set mapping words; built from the public Field type.
inline constexpr cfg::Field TileFaceSetMapping0Word {
    cfg::RegisterScope::State,
    32,
    cfg::TileFaceSetMapping[0][0].base,
    cfg::TileFaceSetMapping[0][0].word,
    0 /*shamt0*/,
    32 /*width*/,
    1 /*count*/,
    0 /*sec_bits*/};
static_assert(TileFaceSetMapping0Word.addr32(cfg::Sec::S0) == TILE_FACE_SET_MAPPING_0_face_set_mapping_0_ADDR32, "face-set mapping word address");
} // namespace hal_port

/**
 * @brief Configure the packer relu mode and threshold.
 *
 * Writes the relu mode and threshold carried by the config to the STACC_RELU config register.
 *
 * @param relu_config: Relu configuration supplying the mode and threshold to apply.
 */
TT_ALWAYS_INLINE void _llk_pack_relu_config_(const ckernel::ReluConfig& relu_config)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    const std::uint32_t mode = static_cast<std::uint32_t>(relu_config.get_mode());

    // STACC_RELU shares this cfg word with ALU_ACC_CTRL_Zero_Flag_disabled_src/dst (bits 0-1, owned by the
    // MATH/UNPACK threads). A whole-word WRCFG_32b would clobber those bits, so use a masked RMW under
    // mutex::REG_RMW -- matching how configure_pack programs relu.
    // Only the packer needs draining: RMWCIB takes the data as an immediate (no GPR), so there is no
    // SETDMAREG->WRCFG producer to fence -- the THCON wait the old whole-word path used is no longer needed.
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();
    sync::mutex::acquire<sync::Mutex::M0>(); // mutex::REG_RMW
    // ApplyRelu and ReluThreshold share one CFG word: one grouped RMW, as production's combined mask.
    cfg::write<cfg::Access::TensixCfgUnit>(
        cfg::set<cfg::StaccRelu::ApplyRelu, cfg::Sec::S0>(mode), cfg::set<cfg::StaccRelu::ReluThreshold, cfg::Sec::S0>(relu_config.get_threshold()));
    sync::mutex::release<sync::Mutex::M0>(); // mutex::REG_RMW
}

/**
 * @brief Enable or disable packer L1 accumulation.
 *
 * @param enable: Non-zero to accumulate packed output into existing L1 data, zero to overwrite.
 */
inline void _llk_pack_reconfig_l1_acc_(const std::uint32_t enable)
{
    hal_port::reconfigure_packer_l1_acc(enable);
}

/**
 * @brief Configure the packer face, row, and edge masks for a reduce output.
 *
 * Programs PCK_EDGE_OFFSET_SEC0/SEC1 masks and TILE_ROW_SET_MAPPING_1 so that only the reduced
 * datums survive: for row reduce a single column per row, for col reduce only the first row, and for
 * scalar reduce a single datum. Default tiled packing selects the row mask per face through
 * TILE_FACE_SET_MAPPING_0; Blackhole has one packer, so the packer selectors cannot select faces.
 *
 * @tparam dim: Reduction dimension, values = <REDUCE_ROW/REDUCE_COL/REDUCE_SCALAR>
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @param tensor_shape: Output face dimensions and face grid.
 * @note Untilize retains its existing row-mask configuration; face selection below is for Default only.
 * @note Pairs with @ref _llk_math_reduce_ on the math thread, whose reduced output these masks gate.
 * @note Call @ref _llk_pack_reduce_mask_clear_ to restore the default pass-through masks.
 */
template <ReduceDim dim, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_reduce_mask_config_(const TensorShape& tensor_shape = DEFAULT_TENSOR_SHAPE)
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // PCK_EDGE_OFFSET_SEC0: mask clears out all the datums in the row; the direct selector (used when face
    // lookup is disabled, Untilize) selects TILE_ROW_SET_MAPPING_1 for packer 0 (tile_row_set_select_pack0 = 1).
    constexpr std::uint32_t pack_edge_offset = hal_port::place<cfg::PckEdgeOffsetSec0::mask>(0x0) | hal_port::place<cfg::PckEdgeTileRowSetSelect::select>(1);

    constexpr std::uint32_t edge_offset_sec1_mask = []
    {
        if constexpr (dim == ReduceDim::REDUCE_COL)
        {
            // PCK_EDGE_OFFSET_SEC1 mask will pass through all the datums in the row as they are
            return 0xffffu;
        }
        else if constexpr (dim == ReduceDim::REDUCE_ROW || dim == ReduceDim::REDUCE_SCALAR)
        {
            // PCK_EDGE_OFFSET_SEC1 mask will clear out all the datums in the row except the first one
            return 0x0001u;
        }
        else
        {
            return 0u;
        }
    }();

    constexpr std::uint32_t row_set_mapping_1 = []
    {
        if constexpr (dim == ReduceDim::REDUCE_ROW)
        {
            // Untilize: each packer packs 1x32 row; Default: all rows use PCK_EDGE_OFFSET_SEC1 mask (1x16 rows)
            return pack_mode == PackMode::Untilize ? 0x11111111u : 0x55555555u;
        }
        else if constexpr (dim == ReduceDim::REDUCE_COL)
        {
            // Untilize: each packer packs 1x32 row; Default: only the first row uses PCK_EDGE_OFFSET_SEC1 mask
            return pack_mode == PackMode::Untilize ? 0x00000005u : 0x00000001u;
        }
        else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
        {
            // TILE_ROW_SET_MAPPING_1 configuration sets only first row to use PCK_EDGE_OFFSET_SEC1 mask
            return 0x00000001u;
        }
        else
        {
            return 0u;
        }
    }();

    // Initialize TMP registers with values we need to write in CFG registers
    hal::scalar::set_low<LOWER_HALFWORD(pack_edge_offset)>(hal::gpr<p_gpr_pack::TMP0>());
    hal::scalar::set_high<UPPER_HALFWORD(pack_edge_offset)>(hal::gpr<p_gpr_pack::TMP0>());
    hal::scalar::set_low<LOWER_HALFWORD(edge_offset_sec1_mask)>(hal::gpr<p_gpr_pack::TMP_LO>());
    hal::scalar::set_low<LOWER_HALFWORD(row_set_mapping_1)>(hal::gpr<p_gpr_pack::TMP1>());
    hal::scalar::set_high<UPPER_HALFWORD(row_set_mapping_1)>(hal::gpr<p_gpr_pack::TMP1>());

    // Wait for packer to finish to avoid breaking its current configuration
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();

    cfg::write<cfg::Access::TensixCfgUnit, cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0>(tensor_shape.face_r_dim);

    // Configure packer
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeOffsetSec1::mask, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP_LO>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[1][0], cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP1>());

    if constexpr (pack_mode == PackMode::Default)
    {
        // The constants repeat 2-bit row-table selectors across all 16 face-table entries:
        // 0x55555555 = [1], 0x11111111 = [1,0], 0x05050505 = [1,1,0,0], 0x01010101 = [1,0,0,0].
        // Row-table 1 applies the reduction mask; row-table 0 masks the entire face.
        const std::uint32_t face_set_mapping = [&tensor_shape]
        {
            static_assert(dim == ReduceDim::REDUCE_ROW || dim == ReduceDim::REDUCE_COL || dim == ReduceDim::REDUCE_SCALAR, "Invalid reduction dimension");
            if (tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 1)
            {
                return 0x55555555;
            }
            else if (tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 2)
            {
                return dim == ReduceDim::REDUCE_COL ? 0x55555555 : 0x11111111;
            }
            else if (tensor_shape.num_faces_r_dim == 2 && tensor_shape.num_faces_c_dim == 1)
            {
                return dim == ReduceDim::REDUCE_ROW ? 0x55555555 : 0x11111111;
            }
            else
            {
                LLK_ASSERT(tensor_shape.num_faces_r_dim == 2 && tensor_shape.num_faces_c_dim == 2, "Invalid tensor shape face grid");
                if constexpr (dim == ReduceDim::REDUCE_ROW)
                {
                    return 0x11111111;
                }
                else if constexpr (dim == ReduceDim::REDUCE_COL)
                {
                    return 0x05050505;
                }
                else
                {
                    return 0x01010101;
                }
            }
        }();
        cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
            hal::gpr<p_gpr::ZERO>());
        cfg::write<cfg::Access::TensixCfgUnit, hal_port::TileFaceSetMapping0Word, cfg::Sec::S0>(face_set_mapping);
        // Select face table 0 for packer 0 and enable the face -> row -> column-mask lookup.
        cfg::write<cfg::Access::TensixCfgUnit>(
            cfg::set<cfg::PckEdgeTileFaceSetSelect::select, cfg::Sec::S0, 0>(), cfg::set<cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 1>());
    }
    else
    {
        cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 0>();
    }

    hal::nop::thread();
    hal::nop::thread();
}

/**
 * @brief Restore the default packer edge masks and tile-row-set mapping after a reduce.
 *
 * Disables face selection, resets the edge-offset masks to pass-through, and points the active
 * tile-row-set mappings back to PCK_EDGE_OFFSET_SEC0, undoing @ref _llk_pack_reduce_mask_config_.
 *
 * @note Pairs with @ref _llk_pack_reduce_mask_config_.
 */
inline void _llk_pack_reduce_mask_clear_()
{
    namespace cfg  = hal::cfg;
    namespace sync = hal::sync;

    // By default, all packers are set to use TILE_ROW_SET_MAPPING_0 and
    // mask is configured to pass through all the datums
    constexpr std::uint32_t pack_edge_offset = hal_port::place<cfg::PckEdgeOffsetSec0::mask>(0xffff);

    // Initialize TMP registers with values we need to write in CFG registers
    hal::scalar::set_low<LOWER_HALFWORD(pack_edge_offset)>(hal::gpr<p_gpr_pack::TMP0>());
    hal::scalar::set_high<UPPER_HALFWORD(pack_edge_offset)>(hal::gpr<p_gpr_pack::TMP0>());

    // Wait for packer to finish to avoid breaking its current configuration
    sync::wait::stall<sync::StallTarget::Config, sync::StallCondition::PackerIdle>();

    cfg::write<cfg::Access::TensixCfgUnit, cfg::PackCounters::pack_reads_per_xy_plane, cfg::Sec::S0, 1>();
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeTileFaceSetSelect::enable, cfg::Sec::S0, 0>();

    // Clear out packer configuration for reduce
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeOffsetSec0::mask, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::PckEdgeOffsetSec1::mask, cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr_pack::TMP0>());

    // All mappings point to PCK_EDGE_OFFSET_SEC0_mask_ADDR32
    cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[0][0], cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr::ZERO>());
    cfg::write<cfg::Access::TensixCfgUnit, cfg::TileRowSetMapping[1][0], cfg::Sec::S0, cfg::GprTransferSize::Bits32, cfg::WrcfgCompletion::Deferred>(
        hal::gpr<p_gpr::ZERO>());

    hal::nop::thread();
    hal::nop::thread();
}
