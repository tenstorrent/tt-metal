// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "llk_defs.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

// Bitmask helper: maps a DataFormat enum value to its corresponding bit in a 64-bit set
inline constexpr std::uint64_t df_bit(DataFormat df)
{
    return 1ULL << to_underlying(df);
}

// Set of Source register DataFormats compatible with Float32 Dest register format
inline constexpr std::uint64_t kFp32DestCompatibleMask = df_bit(DataFormat::Float16_b) | df_bit(DataFormat::Float16) | df_bit(DataFormat::Tf32) |
                                                         df_bit(DataFormat::Float32) | df_bit(DataFormat::MxFp4_2x_A) | df_bit(DataFormat::MxFp4_2x_B);

// Set of Source register DataFormats compatible with Int32 Dest register format
// Int16 is absent from the list because, for the FPU, it is supported only for MOV ops and is
// not compatible with 32bit Dest register mode.
inline constexpr std::uint64_t kInt32DestCompatibleMask =
    df_bit(DataFormat::Int8) | df_bit(DataFormat::UInt8) | df_bit(DataFormat::Int32) | df_bit(DataFormat::Int8_2x) | df_bit(DataFormat::UInt8_2x);

/**
 * @brief Determines whether the source register format and Float32 destination register format are a supported combination
 *
 * @param src_reg_fmt: Source register format
 * @return true if the bit corresponding to src_reg_fmt is set in kFp32DestCompatibleMask.
 *         The mask is right-shifted by the enum value so the relevant bit lands at position 0,
 *         then AND-ed with 1 to isolate it.
 */
inline bool _is_src_fmt_fp32_dest_compatible_(const DataFormat src_reg_fmt)
{
    return (kFp32DestCompatibleMask >> to_underlying(src_reg_fmt)) & 1;
}

/**
 * @brief Determines whether the source register format and Int32 destination register format are a supported combination
 *
 * @param src_reg_fmt: Source register format
 * @return true if the bit corresponding to src_reg_fmt is set in kInt32DestCompatibleMask.
 *         The mask is right-shifted by the enum value so the relevant bit lands at position 0,
 *         then AND-ed with 1 to isolate it.
 */
inline bool _is_src_fmt_int32_dest_compatible_(const DataFormat src_reg_fmt)
{
    return (kInt32DestCompatibleMask >> to_underlying(src_reg_fmt)) & 1;
}

/// @brief What this thread has programmed into the ALU format registers.
#ifdef ENV_LLK_INFRA
static AluCfgTracker alu_cfg {};
#else
extern thread_local AluCfgTracker alu_cfg;
#endif

/**
 * @brief The Dest format an FPU or MOVx2D write produces from a given srcA format.
 *
 * Programming the result into Dstacc_val and asserting Dstacc_override lets Dest readers take the
 * format from config instead of the implied Dest format.
 *
 * @tparam EN_32BIT_DEST: Set to true to use Dest register in 32bit mode
 * @param srcA_format: Input srcA format the Dest format is derived from
 * @param en_int32_dest_format: Set to true to use Dest register in Int32 format
 */
template <bool EN_32BIT_DEST>
constexpr DataFormat _fpu_dest_write_format_(const DataFormat srcA_format, const bool en_int32_dest_format)
{
    if (en_int32_dest_format)
    {
        return DataFormat::Int32;
    }
    if constexpr (EN_32BIT_DEST)
    {
        return DataFormat::Float32;
    }
    else
    {
        switch (srcA_format)
        {
            // Dest needs FP16A for MXFP4_2x_A and FP16B for MXFP4_2x_B
            case DataFormat::MxFp4_2x_A:
                return DataFormat::Float16;
            case DataFormat::MxFp4_2x_B:
                return DataFormat::Float16_b;
            default:
                return srcA_format;
        }
    }
}

/**
 * @brief Derives an @ref AluCfgState into ALU_FORMAT_SPEC words 0..2. Does not program them, @ref _configure_alu_cfg_ does.
 *
 * The one place the bit layout and the per-type quirks live, so equal states build equal words.
 *
 * @tparam EN_32BIT_DEST: Set to true to use Dest register in 32bit mode (Float32/Int32)
 * @param state: The config type and the SrcA/SrcB formats to derive from
 * @param words: Receives the target words
 */
template <bool EN_32BIT_DEST>
inline void _build_alu_words_(const AluCfgState& state, alu_config_u& words)
{
    const bool mov_ops = (state.type == AluCfgType::MOV_OPS_EXPLICIT_FORMAT);

    // MOV ops, 32bit dest: unpacking Int32 or Fp32 to dest decodes the src registers as Tf32.
    const DataFormat srca_fmt = (mov_ops && EN_32BIT_DEST) ? DataFormat::Tf32 : state.srca_format;
    const DataFormat srcb_fmt = (mov_ops && EN_32BIT_DEST) ? DataFormat::Tf32 : state.srcb_format;

    // ALU_ACC_CTRL_INT8_math_enabled does not work with MOVD2A/B, so the MOV path never asserts it.
    const bool int8_math =
        !mov_ops && EN_32BIT_DEST && _is_src_fmt_int32_dest_compatible_(state.srca_format) && _is_src_fmt_int32_dest_compatible_(state.srcb_format);

    const std::uint8_t SRCA_FORMAT_MASKED = masked_data_format(to_underlying(srca_fmt));
    const std::uint8_t SRCB_FORMAT_MASKED = masked_data_format(to_underlying(srcb_fmt));

    words.val[0] = 0;
    words.val[1] = 0;
    words.val[2] = 0;

    words.f.ALU_FORMAT_SPEC_REG_SrcA_val      = SRCA_FORMAT_MASKED;
    words.f.ALU_FORMAT_SPEC_REG_SrcA_override = 1;
    words.f.ALU_FORMAT_SPEC_REG_SrcB_val      = SRCB_FORMAT_MASKED;
    words.f.ALU_FORMAT_SPEC_REG_SrcB_override = 1;
    words.f.ALU_FORMAT_SPEC_REG0_SrcA         = SRCA_FORMAT_MASKED;
    words.f.ALU_FORMAT_SPEC_REG1_SrcB         = SRCB_FORMAT_MASKED;

    words.f.ALU_ACC_CTRL_Fp32_enabled      = EN_32BIT_DEST;
    words.f.ALU_ACC_CTRL_SFPU_Fp32_enabled = EN_32BIT_DEST;
    words.f.ALU_ACC_CTRL_INT8_math_enabled = int8_math;

    // Dest keeps its implied (saved per-bank) format; only the 32-bit MOV path pins it, because
    // MOVD2B/MOVB2D would otherwise pick the hi16 datum mux from a stale saved format.
    if (mov_ops && EN_32BIT_DEST)
    {
        // Takes srca_format, not the Tf32 substitution above
        const DataFormat dstacc_fmt             = _fpu_dest_write_format_<EN_32BIT_DEST>(state.srca_format, int8_math);
        const std::uint8_t DSTACC_FORMAT_MASKED = masked_data_format(to_underlying(dstacc_fmt));

        words.f.ALU_FORMAT_SPEC_REG_Dstacc_val      = DSTACC_FORMAT_MASKED;
        words.f.ALU_FORMAT_SPEC_REG_Dstacc_override = 1;
        words.f.ALU_FORMAT_SPEC_REG2_Dstacc         = DSTACC_FORMAT_MASKED;
    }
}

/**
 * @brief Brings the ALU format registers to the requested state. The entry point for all format config.
 *
 * Returns immediately when the requested state is already programmed; otherwise builds the target words
 * and writes only the ones that differ from what hardware holds.
 *
 * Whole words are written rather via MMIO stores to shadowed registers.
 *
 * @tparam TYPE: Which ALU format config type to program
 * @tparam EN_32BIT_DEST: Set to true to use math dest in 32bit mode (Float32/Int32)
 * @param srca: Input srcA format, used to set ALU configs
 * @param srcb: Input srcB format, used to set ALU configs
 */
template <AluCfgType TYPE, bool EN_32BIT_DEST>
inline void _configure_alu_cfg_(const DataFormat srca, const DataFormat srcb)
{
    static_assert(TYPE != AluCfgType::UNCONFIGURED, "UNCONFIGURED is the empty-tracker marker, not a config to program");

    const AluCfgState wanted {TYPE, srca, srcb};
    if (alu_cfg.programmed == wanted)
    {
        return;
    }

    // alu_cfg is zero-initialised, so before the first write its cached words mean nothing: a target
    // word of 0 would compare equal to one that was never programmed.
    const bool words_known = (alu_cfg.programmed.type != AluCfgType::UNCONFIGURED);

    alu_config_u target;
    _build_alu_words_<EN_32BIT_DEST>(wanted, target);

    for (std::uint32_t i = 0; i < NUM_WORDS_ALU_FORMAT; i++)
    {
        if (words_known && (alu_cfg.words.val[i] == target.val[i]))
        {
            continue;
        }
        cfg[ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32 + i] = target.val[i];
        alu_cfg.words.val[i]                         = target.val[i];
    }

    alu_cfg.programmed = wanted;
}

/**
 * @brief Configures math hardware by setting ALU formats (@ref AluCfgType::DEFAULT) and disabling implied SrcA/SrcB formats
 *
 * @tparam EN_32BIT_DEST: Set to true to use math dest in 32bit mode (Float32/Int32)
 * @param srcA_format: Input srcA format, used to set ALU configs
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 * @param srcB_format: Input srcB format, used to set ALU configs
 * values = Dataformat enum, ex: <Float16/Float16_b/Tf32/Int8/Int16/UInt8>
 * @note Implied SrcA/SrcB formats are always disabled: the ALU reads ALU_FORMAT_SPEC_REG0/1. Implied Dest
 *       format is always enabled - Dest follows SrcA, and Dstacc_override is set only by the MOV-ops path.
 */
template <bool EN_32BIT_DEST>
inline void _llk_math_srcAB_hw_configure_(DataFormat srcA_format, DataFormat srcB_format)
{
    alu_cfg.programmed.type = AluCfgType::UNCONFIGURED;

    // Turn on automatic Tensix-TRISC synchronization
    // RT: This is turned on by default by HW, this should be removed
    set_ttsync_enables<TRACK_ALL>(TRISC_ID);

    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::WAIT_SFPU, p_stall::MATH);

    cfg[DISABLE_IMPLIED_SRCA_FMT_SEC0_Base_ADDR32 + TRISC_ID] = 1;
    cfg[DISABLE_IMPLIED_SRCB_FMT_SEC0_Base_ADDR32 + TRISC_ID] = 1;

    _configure_alu_cfg_<AluCfgType::DEFAULT, EN_32BIT_DEST>(srcA_format, srcB_format);
}

/**
 * @brief Sets the dest dvalid for a FPU/SFPU.
 *
 * @tparam SET_DEST_DVALID: which client to set data valid for, values = <p_cleardvalid::FPU/SFPU>
 * @tparam DST: Destination synchronization mode, values = <SyncFull/SyncHalf>
 */
template <std::uint8_t SET_DEST_DVALID, DstSync DST>
inline void _llk_math_set_dvalid_()
{
    static_assert(SET_DEST_DVALID == p_cleardvalid::FPU || SET_DEST_DVALID == p_cleardvalid::SFPU, "Can only set dest dvalid for FPU and SFPU");

    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::WAIT_SFPU);
    TTI_CLEARDVALID(0, 0, 0, 0, SET_DEST_DVALID, 0);
    if constexpr (DST == DstSync::SyncFull)
    {
        // For DstSync::SyncFull issue a CLEARDVALID instruction for dest bank1 as well in order to use full dest register
        // Reset dest bank id to 0 for the given dest client to ensure SyncFull starts from bank0
        TTI_CLEARDVALID(0, 0, 0, SET_DEST_DVALID, SET_DEST_DVALID, 0);
    }
}

// All the following functions are added to enable Math <-> Pack synchronization
// on destination register due to dest dvalid issue.
// The following functions should be removed once the above issue is resolved.

/**
 * @brief Initialize the math/pack synchronization semaphore and reset the destination section base.
 *
 * Waits for outstanding packs to drain before claiming the destination register, then seeds the
 * MATH_PACK semaphore (one count for SyncFull, two for SyncHalf double-buffering).
 *
 * @tparam DST: Destination synchronization mode, values = <SyncFull/SyncHalf>
 */
template <DstSync DST>
inline void _llk_math_pack_sync_init_()
{
    static_assert(DST == DstSync::SyncFull || DST == DstSync::SyncHalf, "Only Dest Sync Half and Full are supported");

    // Wait for previous packs to finish before claiming all dest
    while (semaphore_read(semaphore::MATH_PACK) > 0)
    {
    }

    _reset_dest_register_offset_();
    _set_dest_section_base_<TRISC_ID>(_get_dest_buffer_base_());

    constexpr std::uint32_t num_sem = (DST == DstSync::SyncFull) ? 1 : 2;
    TTI_SEMINIT(num_sem, 0, 0, p_stall::SEMAPHORE_1);
}

/**
 * @brief Block the math thread until the destination register is available for writing.
 *
 * @note @ref _llk_math_pack_sync_init_ must have seeded the MATH_PACK semaphore before this function.
 */
inline void _llk_math_wait_for_dest_available_()
{
    TTI_SEMWAIT(p_stall::STALL_MATH | p_stall::STALL_SFPU | p_stall::STALL_SYNC, p_stall::STALL_ON_MAX, 0, semaphore::t6_sem(semaphore::MATH_PACK));
}

/**
 * @brief Signal completion of a destination section, posting the MATH_PACK semaphore and flipping banks in half-sync mode.
 *
 * @tparam DST: Destination synchronization mode, values = <SyncFull/SyncHalf>
 * @tparam EN_32BIT_DEST: Set to true if the destination register is in 32-bit (Float32/Int32) mode
 */
template <DstSync DST, bool EN_32BIT_DEST>
inline void _llk_math_dest_section_done_()
{
    t6_semaphore_post<p_stall::MATH, p_stall::WAIT_SFPU>(semaphore::MATH_PACK);
    if constexpr (DST == DstSync::SyncHalf)
    {
        _update_dest_register_offset_<EN_32BIT_DEST>();
        std::uint32_t base_addr = _get_dest_buffer_base_();
        TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::MATH, p_stall::WAIT_SFPU);
        _set_dest_section_base_<TRISC_ID>(base_addr);
    }
}
