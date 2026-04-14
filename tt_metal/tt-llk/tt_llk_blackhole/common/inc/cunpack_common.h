// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "llk_assert.h"

namespace ckernel::unpacker
{
constexpr std::uint32_t TILE_DESC_SIZE = 2; // Unpacker descriptor size in dwords
constexpr std::uint32_t CONFIG_SIZE    = 2; // Unpacker configuration size in dwords
constexpr std::uint32_t NUM_UNPACKERS  = 2; // Number of unpackers

// Unpack tile descriptor
typedef struct
{
    // word 0
    std::uint32_t in_data_format : DATA_FORMAT_BIT_COUNT;
    std::uint32_t uncompressed       : 1;
    std::uint32_t reserved_0         : 3;
    std::uint32_t blobs_per_xy_plane : 4;
    std::uint32_t reserved_1         : 4;
    std::uint32_t x_dim              : 16;
    // word 1
    std::uint32_t y_dim : 16;
    std::uint32_t z_dim : 16;
    // word 2
    std::uint32_t w_dim            : 16;
    std::uint32_t blobs_y_start_lo : 16;
    // word 3
    std::uint32_t blobs_y_start_hi : 16;
    std::uint32_t digest_type      : 8; // Not used
    std::uint32_t digest_size      : 8; // Not used
} unpack_tile_descriptor_t;             // Unpack configuration

static_assert(sizeof(unpack_tile_descriptor_t) == (sizeof(std::uint32_t) * 4));

typedef union
{
    std::uint32_t val[4];
    unpack_tile_descriptor_t f;
} unpack_tile_descriptor_u;

// Unpack config
typedef struct
{
    // word 0
    std::uint32_t out_data_format : DATA_FORMAT_BIT_COUNT;
    std::uint32_t throttle_mode             : 2;
    std::uint32_t context_count             : 2;
    std::uint32_t haloize_mode              : 1; // this controls xy transpose on unpacker
    std::uint32_t tileize_mode              : 1;
    std::uint32_t unpack_src_reg_set_update : 1;
    std::uint32_t unpack_if_sel             : 1;
    std::uint32_t upsample_rate             : 2;
    std::uint32_t reserved_1                : 1;
    std::uint32_t upsamle_and_interlave     : 1;
    std::uint32_t shift_amount              : 16;
    // word 1
    std::uint32_t uncompress_cntx0_3    : 4;
    std::uint32_t unpack_if_sel_cntx0_3 : 4;
    std::uint32_t force_shared_exp      : 1;
    std::uint32_t reserved_2            : 7;
    std::uint32_t uncompress_cntx4_7    : 4;
    std::uint32_t unpack_if_sel_cntx4_7 : 4;
    std::uint32_t reserved_3            : 8;
    // word 2
    std::uint32_t limit_addr : 17;
    std::uint32_t reserved_4 : 15;
    // word 3
    std::uint32_t fifo_size  : 17;
    std::uint32_t reserved_5 : 15;
} unpack_config_t;

static_assert(sizeof(unpack_config_t) == (sizeof(std::uint32_t) * 4));

typedef union
{
    std::uint32_t val[4];
    unpack_config_t f;
} unpack_config_u;

// ALU config
typedef struct
{
    std::uint32_t ALU_ROUNDING_MODE_Fpu_srnd_en     : 1;
    std::uint32_t ALU_ROUNDING_MODE_Gasket_srnd_en  : 1;
    std::uint32_t ALU_ROUNDING_MODE_Packer_srnd_en  : 1;
    std::uint32_t ALU_ROUNDING_MODE_Padding         : 10;
    std::uint32_t ALU_ROUNDING_MODE_GS_LF           : 1;
    std::uint32_t ALU_ROUNDING_MODE_Bfp8_HF         : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG0_SrcAUnsigned : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG0_SrcBUnsigned : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG0_SrcA         : 4;
    std::uint32_t ALU_FORMAT_SPEC_REG1_SrcB         : 4;
    std::uint32_t ALU_FORMAT_SPEC_REG2_Dstacc       : 4;
    std::uint32_t ALU_ACC_CTRL_Fp32_enabled         : 1;
    std::uint32_t ALU_ACC_CTRL_SFPU_Fp32_enabled    : 1;
    std::uint32_t ALU_ACC_CTRL_INT8_math_enabled    : 1;
} alu_config_t;

static_assert(sizeof(alu_config_t) == sizeof(std::uint32_t));

typedef union
{
    std::uint32_t val;
    alu_config_t f;
} alu_config_u;

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
inline void unpacker_addr_counter_init()
{
    TTI_SETADCXY(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1011);
    TTI_SETADCZW(p_setadc::UNP_A | p_setadc::UNP_B, 0, 0, 0, 0, 0b1111);
}

inline void unpacker_iteration_cleanup(std::uint32_t &context)
{
    // Indicate that unpacker is done, and we can program the next one
    t6_semaphore_get(semaphore::UNPACK_SYNC);
    context = 1 - context;
    if (context == 1)
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0104);
    }
    else
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    }
}

inline void unpacker_wrapup()
{
    // Clear unpacker0 tile offset address
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_address_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC0_REG7_Offset_cntx1_address_ADDR32);

    // Clear unpacker1 tile offset address
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC1_REG7_Offset_address_ADDR32);
    TTI_WRCFG(p_gpr::ZERO, p_cfg::WRCFG_32b, THCON_SEC1_REG7_Offset_cntx1_address_ADDR32);

    // Clear context offset and counter
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x1010);
}

inline std::uint32_t unpack_16B_address(const std::uint32_t addr)
{
    return (addr << FIFO_BASE_ADDRESS_ALIGN_BITS) >> 4;
}

inline void flush_xsearch_cache(const std::uint32_t unpacker)
{
    TTI_UNPACR(unpacker, 0, 0, 0, 0, 0, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 1, 0);
}

// Wait for threshold of busy contexts to fall below total available contexts
inline void wait_for_next_context(const std::uint32_t num_contexts)
{
    while (semaphore_read(semaphore::UNPACK_SYNC) >= num_contexts)
    {
    }
}

inline void switch_config_context(std::uint32_t &unp_cfg_context)
{
    // Switch config context
    unp_cfg_context = 1 - unp_cfg_context;
    if (unp_cfg_context == 0)
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
    }
    else
    {
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0101);
    }
}

inline void reset_config_context()
{
    // Switch config context
    unp_cfg_context = 0;
    TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, 0x0000);
}

// Sync on unpacker idle via waiting busy contexts counter 0
inline void wait_for_idle()
{
    while (semaphore_read(semaphore::UNPACK_SYNC) > 0)
    {
    }
}

inline void enable_int8_fpu_math()
{
    alu_config_u alu_payload                     = {.val = 0};
    alu_payload.f.ALU_ACC_CTRL_INT8_math_enabled = 1;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, ALU_ACC_CTRL_INT8_math_enabled_MASK>(alu_payload.val);
}

/**
 * \brief Returns true if unpacker I/O uses 32-bit formats (Int32 or Float32).
 *
 * Used to determine unpack-to-dest mode and related configuration when both
 * input and output are 32-bit. Masks low nibble of format codes for comparison.
 *
 * \param unpack_src_format Unpacker input (L1) data format.
 * \param unpack_dst_format Unpacker output (register) data format.
 * \return true if both formats are Int32 or Float32; false otherwise.
 */
inline constexpr bool is_32bit_input(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format)
{
    const DataFormat input_df  = static_cast<DataFormat>(masked_data_format(unpack_src_format));
    const DataFormat output_df = static_cast<DataFormat>(masked_data_format(unpack_dst_format));
    return (input_df == DataFormat::Int32 || input_df == DataFormat::Float32) && (output_df == DataFormat::Int32 || output_df == DataFormat::Float32);
}

/**
 * \brief Checks if the unpacker conversion is supported w.r.t. the FP32 dest accumulation mode.
 *
 * The unpacker writes to one of two register destinations (ref: Unpackers/FormatConversion.md):
 *
 *   - SrcA / SrcB registers (unpack_to_dest = false): support TF32, BF16 (Float16_b), FP16
 *     (Float16), or Integer "8/16" register formats depending on the L1 format.
 *   - Dst register (unpack_to_dest = true): supports FP32 (Float32), BF16 (Float16_b), FP16
 *     (Float16), Integer "32/16/8" register formats depending on the L1 format.
 *
 * The `is_fp32_dest_acc_en` flag is used as a policy gate for TF32 availability:
 *   - true:  TF32 register output is enabled on the SrcA/SrcB path.
 *   - false: TF32 register output is disabled in this support check.
 *
 * NOTE: this gating is primarily an LLK policy choice, not necessarily a strict hardware
 * prohibition. Src registers can represent TF32, but when DEST accumulates in 16-bit mode
 * (is_fp32_dest_acc_en == false), results are quantized at DEST write/accumulation points.
 * In that mode, LLK prefers unpacking to Float16/Float16_b to keep behavior aligned with
 * inferred formats and to avoid mixed-precision ambiguity.
 *
 * This is one half of the full unpacker conversion support check; it validates only constraints
 * related to is_fp32_dest_acc_en.  For a complete check, also call
 * is_unpacker_format_conversion_supported_dest().
 *
 * \param unpack_src_format   Data format of tiles in L1 (maps to InDataFormat config field).
 * \param unpack_dst_format   Desired register output format (maps to OutDataFormat config field).
 * \param is_fp32_dest_acc_en True when FP32 dest accumulation is enabled; controls availability
 *                            of TF32 (SrcA/SrcB) and FP32 (Dst) register formats.
 * \return true if the conversion is supported given the FP32 accumulation setting.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_unpacker_format_conversion_supported_fp32_acc(
    const DataFormat unpack_src_format, const DataFormat unpack_dst_format, const bool is_fp32_dest_acc_en)
{
    switch (unpack_src_format)
    {
        // -------------------------------------------------------------------------
        // 1. Float32 (FP32, e8m23) in L1.
        //
        //    ISA conversions (Unpackers/FormatConversion.md):
        //      SrcA/SrcB path:
        //        FP32 → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked below)
        //                                   and !unpack_to_dest (checked in _dest).
        //                                   TF32 is the 19-bit SrcA/SrcB format used to preserve
        //                                   maximum precision when DEST accumulates in FP32 mode.
        //        FP32 → BF16  (Float16_b): always valid.
        //        FP32 → FP16  (Float16):   always valid.
        //      Dst path:
        //        FP32 → FP32  (Float32):   identity; unpack_to_dest checked in _dest.
        //        FP32 → BF16  (Float16_b): always valid.
        //        FP32 → FP16  (Float16):   always valid.
        case DataFormat::Float32:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                    return is_fp32_dest_acc_en;
                case DataFormat::Float32:
                case DataFormat::Float16:
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // 2. Tf32 (TF32, e8m10) in L1 — stored in a 32-bit FP32 footprint with the
        //    lower 13 mantissa bits zeroed.
        //
        //    ISA conversions:
        //      SrcA/SrcB path:
        //        TF32 → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked below)
        //                                   and !unpack_to_dest (checked in _dest).
        //        TF32 → BF16  (Float16_b): always valid.
        //        TF32 → FP16  (Float16):   always valid.
        //      Dst path:
        //        TF32 → FP32  (Float32):   valid when is_fp32_dest_acc_en (checked below)
        //                                   and unpack_to_dest (checked in _dest).
        //                                   Dst is in FP32 accumulation mode; TF32 bits are
        //                                   preserved as the mantissa zeros make it a valid FP32.
        //        TF32 → BF16  (Float16_b): always valid.
        //        TF32 → FP16  (Float16):   always valid.
        case DataFormat::Tf32:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                case DataFormat::Float32:
                    return is_fp32_dest_acc_en;
                case DataFormat::Float16:
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // 3. Float16 (FP16, e5m10), Lf8 (FP8, e5m2), and Fp8_e4m3 (FP8, e4m3) in L1.
        //
        //    ISA conversions (same rule for all; A-format exponent family):
        //      SrcA/SrcB and Dst: FP16/FP8 → FP16 (Float16) in the register.
        //    Config: InDataFormat and OutDataFormat can each be any of these code points;
        //    the hardware always produces FP16 data in the register.
        //    Fp8_e4m3 is a Blackhole-only variant distinguished by the Unp_LF8_4b_exp bit.
        case DataFormat::Float16:
        case DataFormat::Lf8:
        case DataFormat::Fp8_e4m3:
            return unpack_dst_format == DataFormat::Float16 || unpack_src_format == unpack_dst_format;

        // -------------------------------------------------------------------------
        // 4. Float16_b (BF16, e8m7) in L1.
        //
        //    ISA conversions:
        //      SrcA/SrcB path:
        //        BF16 → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked below)
        //                                   and !unpack_to_dest (checked in _dest).
        //                                   TF32 is the 19-bit SrcA/SrcB format used to preserve
        //                                   maximum precision when DEST accumulates in FP32 mode.
        //        BF16 → BF16  (Float16_b): always valid (identity).
        //      Dst path:
        //        BF16 → BF16  (Float16_b): always valid (identity).
        //    Note: FP16 is NOT a valid output for BF16 input — no cross-exponent-width conversion
        //    from 8-bit exponent BF16 to 5-bit exponent FP16 is supported by the unpacker.
        case DataFormat::Float16_b:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                    return is_fp32_dest_acc_en;
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // 5. Bfp8 / Bfp4 / Bfp2 (A-side block float: BFP8a/BFP4a/BFP2a, 5-bit exponent) in L1.
        //
        //    ISA conversions: BFP8a/BFP4a/BFP2a → FP16 (Float16) only (SrcA/SrcB and Dst).
        //    The hardware reads the shared 5-bit exponent and per-datum mantissa bits,
        //    reconstructing FP16-format data in the register.
        //
        //    Config: InDataFormat and OutDataFormat are set to the same BFP code point
        //    (identity config). The actual register content is always FP16.
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
            switch (unpack_dst_format)
            {
                case DataFormat::Float16:
                    return true;
                default:
                    return unpack_src_format == unpack_dst_format;
            }

        // -------------------------------------------------------------------------
        // 6. Bfp8_b / Bfp4_b / Bfp2_b (B-side block float: BFP8/BFP4/BFP2, 8-bit exponent) in L1.
        //
        //    ISA conversions (FormatConversion.md):
        //      SrcA/SrcB path:
        //        → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked below)
        //                              and !unpack_to_dest (checked in _dest).
        //        → BF16  (Float16_b): always valid.
        //      Dst path:
        //        → BF16  (Float16_b): always valid.
        //
        //    ISA config: identity (InDataFormat == OutDataFormat). The actual register
        //    content is TF32 (when fp32_dest_acc_en, SrcA/SrcB) or BF16 (otherwise).
        //
        //    Sub-byte formats (Bfp4_b, Bfp2_b) additionally support expansion to
        //    Bfp8_b via InDataFormat=BFP4/BFP2, OutDataFormat=BFP8 (used by
        //    data_format_inference.py).
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                    return is_fp32_dest_acc_en;
                case DataFormat::Float16_b:
                case DataFormat::Bfp8_b:
                    return true;
                default:
                    return unpack_src_format == unpack_dst_format;
            }

        // -------------------------------------------------------------------------
        // 7. Int32 (INT32, sign-magnitude 32-bit) in L1.
        //
        //    ISA conversions:
        //      SrcA/SrcB path: NOT possible (ISA doc explicitly states "Not possible").
        //      Dst path:       INT32 → Integer "32" (Int32): valid (32b data movement to Dst).
        //    Hence, only valid when unpack_to_dest = true (checked in _dest).
        case DataFormat::Int32:
            return unpack_dst_format == DataFormat::Int32;

        // -------------------------------------------------------------------------
        // 8. UInt32 (opaque 32-bit) in L1.
        //
        //    Not explicitly listed in the ISA doc. Treated as opaque 32-bit data analogous to
        //    Int32: only valid when targeting the Dst register (unpack_to_dest = true, checked in _dest).
        case DataFormat::UInt32:
            return unpack_dst_format == DataFormat::UInt32;

        // -------------------------------------------------------------------------
        // 9. UInt16 (INT16, opaque 16-bit data) in L1.
        //
        //    ISA conversions: INT16 → Integer "16" (UInt16) [SrcA/SrcB and Dst].
        //    Unlike INT32 (which is "Not possible" for SrcA/SrcB), INT16 is valid for
        //    both SrcA/SrcB and Dst — no unpack_to_dest restriction applies.
        case DataFormat::UInt16:
            return unpack_dst_format == DataFormat::UInt16;

        // -------------------------------------------------------------------------
        // 10. UInt8 (UINT8, unsigned 8-bit integer) in L1.
        //
        //     ISA conversions: UINT8 → Integer "8" [SrcA/SrcB and Dst].
        //     Both INT8 and UINT8 L1 formats land in Int8 register format (Integer "8");
        //     signed vs unsigned is distinguished at ALU time via ALU_FORMAT_SPEC_REG0_SrcA/BUnsigned.
        case DataFormat::UInt8:
            return unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8;

        // -------------------------------------------------------------------------
        // 11. Int8 (INT8, sign-magnitude 8-bit) in L1.
        //
        //     ISA conversions:
        //       SrcA/SrcB path:
        //         INT8 → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked below)
        //                                    and !unpack_to_dest (checked in _dest).
        //                                    TF32 is the 19-bit SrcA/SrcB format used to preserve
        //                                    maximum precision when DEST accumulates in FP32 mode.
        //                                    Uses InDataFormat=BFP8 with REG2_Force_shared_exp set
        //                                    and a fixed exponent supplied via FORCED_SHARED_EXP.
        //         INT8 → BF16  (Float16_b): always valid (same BFP8+force_shared_exp mechanism).
        //         INT8 → Integer "8" (Int8): always valid.
        //       Dst path:
        //         INT8 → BF16  (Float16_b): always valid (BFP8+force_shared_exp).
        //         INT8 → Integer "8" (Int8): always valid.
        case DataFormat::Int8:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                    return is_fp32_dest_acc_en;
                case DataFormat::Float16_b:
                case DataFormat::Int8:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // 12. Unknown or not-yet-handled formats.
        default:
            return false;
    }
}

/**
 * \brief Checks if the unpacker conversion is supported w.r.t. the target register destination.
 *
 * This is one half of the full unpacker conversion support check; it validates only constraints
 * related to unpack_to_dest.  For a complete check, also call
 * is_unpacker_format_conversion_supported_fp32_acc().
 *
 * \param unpack_src_format Data format of tiles in L1 (maps to InDataFormat config field).
 * \param unpack_dst_format Desired register output format (maps to OutDataFormat config field).
 * \param unpack_to_dest    True when targeting the Dst register (32b path); false when
 *                          targeting SrcA/SrcB registers (Tf32/16b/8b path).
 * \return true if the conversion is supported given the register destination.
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_unpacker_format_conversion_supported_dest(
    const DataFormat unpack_src_format, const DataFormat unpack_dst_format, const bool unpack_to_dest)
{
    switch (unpack_src_format)
    {
        // -------------------------------------------------------------------------
        // 1. Float32 (FP32, e8m23) in L1.
        //
        //    ISA conversions (Unpackers/FormatConversion.md):
        //      SrcA/SrcB path:
        //        FP32 → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked in _fp32_acc)
        //                                   and !unpack_to_dest (checked below).
        //                                   TF32 is the 19-bit SrcA/SrcB format used to preserve
        //                                   maximum precision when DEST accumulates in FP32 mode.
        //        FP32 → BF16  (Float16_b): always valid.
        //        FP32 → FP16  (Float16):   always valid.
        //      Dst path:
        //        FP32 → FP32  (Float32):   identity; unpack_to_dest checked below.
        //        FP32 → BF16  (Float16_b): always valid.
        //        FP32 → FP16  (Float16):   always valid.
        case DataFormat::Float32:
            switch (unpack_dst_format)
            {
                case DataFormat::Float32:
                    return unpack_to_dest;
                case DataFormat::Tf32:
                    // TODO: Uncomment this line when unpack_to_dest gets handled in compute API.
                    // return !unpack_to_dest;
                case DataFormat::Float16:
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // 2. Tf32 (TF32, e8m10) in L1 — stored in a 32-bit FP32 footprint with the
        //    lower 13 mantissa bits zeroed.
        //
        //    ISA conversions:
        //      SrcA/SrcB path:
        //        TF32 → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked in _fp32_acc)
        //                                   and !unpack_to_dest (checked below).
        //        TF32 → BF16  (Float16_b): always valid.
        //        TF32 → FP16  (Float16):   always valid.
        //      Dst path:
        //        TF32 → FP32  (Float32):   valid when is_fp32_dest_acc_en (checked in _fp32_acc)
        //                                   and unpack_to_dest (checked below).
        //                                   Dst is in FP32 accumulation mode; TF32 bits are
        //                                   preserved as the mantissa zeros make it a valid FP32.
        //        TF32 → BF16  (Float16_b): always valid.
        //        TF32 → FP16  (Float16):   always valid.
        case DataFormat::Tf32:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                    return !unpack_to_dest;
                case DataFormat::Float32:
                    return unpack_to_dest;
                case DataFormat::Float16:
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // 3. Float16 (FP16, e5m10), Lf8 (FP8, e5m2), and Fp8_e4m3 (FP8, e4m3) in L1.
        //
        //    ISA conversions (same rule for all; A-format exponent family):
        //      SrcA/SrcB and Dst: FP16/FP8 → FP16 (Float16) in the register.
        //    Config: InDataFormat and OutDataFormat can each be any of these code points;
        //    the hardware always produces FP16 data in the register.
        //    Fp8_e4m3 is a Blackhole-only variant distinguished by the Unp_LF8_4b_exp bit.
        case DataFormat::Float16:
        case DataFormat::Lf8:
        case DataFormat::Fp8_e4m3:
            return unpack_dst_format == DataFormat::Float16 || unpack_src_format == unpack_dst_format;

        // -------------------------------------------------------------------------
        // 4. Float16_b (BF16, e8m7) in L1.
        //
        //    ISA conversions:
        //      SrcA/SrcB path:
        //        BF16 → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked in _fp32_acc)
        //                                   and !unpack_to_dest (checked below).
        //                                   TF32 is the 19-bit SrcA/SrcB format used to preserve
        //                                   maximum precision when DEST accumulates in FP32 mode.
        //        BF16 → BF16  (Float16_b): always valid (identity).
        //      Dst path:
        //        BF16 → BF16  (Float16_b): always valid (identity).
        //    Note: FP16 is NOT a valid output for BF16 input — no cross-exponent-width conversion
        //    from 8-bit exponent BF16 to 5-bit exponent FP16 is supported by the unpacker.
        case DataFormat::Float16_b:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                    return !unpack_to_dest;
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // 5. Bfp8 / Bfp4 / Bfp2 (A-side block float: BFP8a/BFP4a/BFP2a, 5-bit exponent) in L1.
        //
        //    ISA conversions: BFP8a/BFP4a/BFP2a → FP16 (Float16) only (SrcA/SrcB and Dst).
        //    The hardware reads the shared 5-bit exponent and per-datum mantissa bits,
        //    reconstructing FP16-format data in the register.
        //
        //    Config: InDataFormat and OutDataFormat are set to the same BFP code point
        //    (identity config). The actual register content is always FP16.
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
            return unpack_dst_format == DataFormat::Float16 || unpack_src_format == unpack_dst_format;

        // -------------------------------------------------------------------------
        // 6. Bfp8_b / Bfp4_b / Bfp2_b (B-side block float: BFP8/BFP4/BFP2, 8-bit exponent) in L1.
        //
        //    ISA conversions (FormatConversion.md):
        //      SrcA/SrcB path:
        //        → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked in _fp32_acc)
        //                              and !unpack_to_dest (checked below).
        //        → BF16  (Float16_b): always valid.
        //      Dst path:
        //        → BF16  (Float16_b): always valid.
        //
        //    ISA config: identity (InDataFormat == OutDataFormat). The actual register
        //    content is TF32 (when fp32_dest_acc_en, SrcA/SrcB) or BF16 (otherwise).
        //
        //    Sub-byte formats (Bfp4_b, Bfp2_b) additionally support expansion to
        //    Bfp8_b via InDataFormat=BFP4/BFP2, OutDataFormat=BFP8 (used by
        //    data_format_inference.py).
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                    return !unpack_to_dest;
                case DataFormat::Float16_b:
                case DataFormat::Bfp8_b:
                    return true;
                default:
                    return unpack_src_format == unpack_dst_format;
            }

        // -------------------------------------------------------------------------
        // 7. Int32 (INT32, sign-magnitude 32-bit) in L1.
        //
        //    ISA conversions:
        //      SrcA/SrcB path: NOT possible (ISA doc explicitly states "Not possible").
        //      Dst path:       INT32 → Integer "32" (Int32): valid (32b data movement to Dst).
        //    Hence, only valid when unpack_to_dest = true.
        case DataFormat::Int32:
            return unpack_dst_format == DataFormat::Int32 && unpack_to_dest;

        // -------------------------------------------------------------------------
        // 8. UInt32 (opaque 32-bit) in L1.
        //
        //    Not explicitly listed in the ISA doc. Treated as opaque 32-bit data analogous to
        //    Int32: only valid when targeting the Dst register (unpack_to_dest = true).
        case DataFormat::UInt32:
            return unpack_dst_format == DataFormat::UInt32 && unpack_to_dest;

        // -------------------------------------------------------------------------
        // 9. UInt16 (INT16, opaque 16-bit data) in L1.
        //
        //    ISA conversions: INT16 → Integer "16" (UInt16) [SrcA/SrcB and Dst].
        //    Unlike INT32 (which is "Not possible" for SrcA/SrcB), INT16 is valid for
        //    both SrcA/SrcB and Dst — no unpack_to_dest restriction applies.
        case DataFormat::UInt16:
            return unpack_dst_format == DataFormat::UInt16;

        // -------------------------------------------------------------------------
        // 10. UInt8 (UINT8, unsigned 8-bit integer) in L1.
        //
        //     ISA conversions: UINT8 → Integer "8" [SrcA/SrcB and Dst].
        //     Both INT8 and UINT8 L1 formats land in Int8 register format (Integer "8");
        //     signed vs unsigned is distinguished at ALU time via ALU_FORMAT_SPEC_REG0_SrcA/BUnsigned.
        case DataFormat::UInt8:
            return unpack_dst_format == DataFormat::Int8 || unpack_dst_format == DataFormat::UInt8;

        // -------------------------------------------------------------------------
        // 11. Int8 (INT8, sign-magnitude 8-bit) in L1.
        //
        //     ISA conversions:
        //       SrcA/SrcB path:
        //         INT8 → TF32  (Tf32):      valid when is_fp32_dest_acc_en (checked in _fp32_acc)
        //                                    and !unpack_to_dest (checked below).
        //                                    TF32 is the 19-bit SrcA/SrcB format used to preserve
        //                                    maximum precision when DEST accumulates in FP32 mode.
        //                                    Uses InDataFormat=BFP8 with REG2_Force_shared_exp set
        //                                    and a fixed exponent supplied via FORCED_SHARED_EXP.
        //         INT8 → BF16  (Float16_b): always valid (same BFP8+force_shared_exp mechanism).
        //         INT8 → Integer "8" (Int8): always valid.
        //       Dst path:
        //         INT8 → BF16  (Float16_b): always valid (BFP8+force_shared_exp).
        //         INT8 → Integer "8" (Int8): always valid.
        case DataFormat::Int8:
            switch (unpack_dst_format)
            {
                case DataFormat::Tf32:
                    return !unpack_to_dest;
                case DataFormat::Float16_b:
                case DataFormat::Int8:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // 12. Unknown or not-yet-handled formats.
        default:
            return false;
    }
}

template <bool is_fp32_dest_acc_en, bool row_pool = false, bool fpu_srnd_en = false, bool pack_srnd_en = false, bool disable_src_zero_flag = false>
inline void configure_unpack_AB(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const bool transpose_xy_srca_en     = false,
    const std::uint32_t unpA_num_faces  = 4,
    const std::uint32_t unpB_num_faces  = 4)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpA_src_format), static_cast<DataFormat>(unpA_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");
    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpB_src_format), static_cast<DataFormat>(unpB_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    // Check that unpacker is done (all contexts freed up) before starting hw configuration
    wait_for_idle();

    // Reset address counters
    unpacker_addr_counter_init();

    const std::uint32_t unpA_src_format_masked = masked_data_format(unpA_src_format);
    const std::uint32_t unpB_src_format_masked = masked_data_format(unpB_src_format);
    const std::uint32_t unpA_dst_format_masked = masked_data_format(unpA_dst_format);
    const std::uint32_t unpB_dst_format_masked = masked_data_format(unpB_dst_format);

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    std::uint32_t unpA_ch1_x_stride = (unpA_dst_format_masked & 0x3) == to_underlying(DataFormat::Float32)   ? 4
                                      : (unpA_dst_format_masked & 0x3) == to_underlying(DataFormat::Float16) ? 2
                                                                                                             : 1;
    std::uint32_t unpB_ch1_x_stride = (unpB_dst_format_masked & 0x3) == to_underlying(DataFormat::Float32)   ? 4
                                      : (unpB_dst_format_masked & 0x3) == to_underlying(DataFormat::Float16) ? 2
                                                                                                             : 1;
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    std::uint32_t unpB_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpB_ch1_x_stride;
    std::uint32_t exp_width         = (static_cast<std::uint32_t>(unpA_dst_format_masked) >> 2) & 0x1; // 0=5-bit, 1=8-bit

    // Strides for incrementing ch1 address to srcA and srcB
    cfg[UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
        (0 << UNP0_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
        (unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT); // Z and W(not used) stride for dest address (ch1)

    cfg[UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32] =
        (0 << UNP1_ADDR_CTRL_ZW_REG_1_Wstride_SHAMT) |
        (unpB_ch1_z_stride << UNP1_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT); // Z and W(not used) stride for dest address (ch1)

    // Math ALU_FORMAT_REG
    t6_mutex_acquire(mutex::REG_RMW);
    std::uint32_t alu_src_format = (0x0 << ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT);

    constexpr std::uint32_t mask0 = (1 << (ALU_FORMAT_SPEC_REG_Dstacc_override_SHAMT + 1)) - 1;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, ALU_FORMAT_SPEC_REG_SrcA_val_SHAMT, mask0>(alu_src_format);

    alu_config_u alu_payload = {.val = 0};

    constexpr std::uint32_t alu_format_mask = ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK | ALU_FORMAT_SPEC_REG0_SrcBUnsigned_MASK;

    if (unpA_src_format == to_underlying(DataFormat::UInt8))
    {
        alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcAUnsigned = 1;
    }
    if (unpB_src_format == to_underlying(DataFormat::UInt8))
    {
        alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcBUnsigned = 1;
    }

    // FP32 accumulation and SFPU to read dest as FP32
    // NOTE: This assumes these config fields are adjacent and in same register!!
    static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_FORMAT_SPEC_REG0_SrcA_ADDR32);
    static_assert(ALU_ACC_CTRL_Fp32_enabled_ADDR32 == ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32);
    constexpr std::uint32_t alu_stoch_rnd_mask =
        ALU_ROUNDING_MODE_Fpu_srnd_en_MASK | ALU_ROUNDING_MODE_Gasket_srnd_en_MASK | ALU_ROUNDING_MODE_Packer_srnd_en_MASK;
    alu_payload.f.ALU_ROUNDING_MODE_Fpu_srnd_en    = fpu_srnd_en;
    alu_payload.f.ALU_ROUNDING_MODE_Gasket_srnd_en = pack_srnd_en;
    alu_payload.f.ALU_ROUNDING_MODE_Packer_srnd_en = pack_srnd_en;

    constexpr std::uint32_t alu_mask = alu_format_mask | alu_stoch_rnd_mask;

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_mask>(alu_payload.val);

    // TODO NC: Find out why we need to disable src zero flags for uint16 dst format #960
    bool disable_src_zero_flag_val = disable_src_zero_flag || (static_cast<std::uint32_t>(unpA_dst_format) == static_cast<std::uint32_t>(DataFormat::UInt16)) ||
                                     (static_cast<std::uint32_t>(unpB_dst_format) == static_cast<std::uint32_t>(DataFormat::UInt16));
    cfg_reg_rmw_tensix<ALU_ACC_CTRL_Zero_Flag_disabled_src_RMW>(disable_src_zero_flag_val ? 1 : 0);

    // Set FP8 E4M3 mode, bit is accessible by unpacker/packer
    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Unp_LF8_4b_exp_RMW>(((unpA_src_format & 0x1F) == (std::uint32_t)DataFormat::Fp8_e4m3) ? 1 : 0);
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Unp_LF8_4b_exp_RMW>(((unpB_src_format & 0x1F) == (std::uint32_t)DataFormat::Fp8_e4m3) ? 1 : 0);

    t6_mutex_release(mutex::REG_RMW);

    // Set tile descriptor
    unpack_tile_descriptor_u tile_descriptor;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        tile_descriptor.val[i] = 0;
    }
    tile_descriptor.f.in_data_format = static_cast<std::uint32_t>(unpA_src_format_masked);
    tile_descriptor.f.uncompressed   = 1; // Input tile is uncompressed
    tile_descriptor.f.x_dim          = 0; // Not used for unpA as value is overridden by per context x_dim set below. Used for unpB
    tile_descriptor.f.y_dim          = 1;
    tile_descriptor.f.z_dim          = unpA_num_faces;
    // tile_descriptor.f.blobs_per_xy_plane = 0;
    // tile_descriptor.f.blobs_y_start = 0;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    }
    tile_descriptor.f.in_data_format = row_pool ? to_underlying(DataFormat::Float32) : unpB_src_format_masked;
    tile_descriptor.f.x_dim          = unpB_face_r_dim * FACE_C_DIM;
    tile_descriptor.f.z_dim          = unpB_num_faces;
    for (std::uint32_t i = 0; i < TILE_DESC_SIZE; i++)
    {
        cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + i] = tile_descriptor.val[i];
    }

    // Set unpacker config
    unpack_config_u config;
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        config.val[i] = 0;
    }
    config.f.out_data_format = unpA_dst_format_masked;
    config.f.throttle_mode   = 2;
    config.f.context_count   = 0;
    config.f.haloize_mode    = transpose_xy_srca_en ? 1 : 0;
    // config.f.upsample_rate   = 0;
    // config.f.upsamle_and_interlave  = 0;
    // config.f.shift_amount = 0;
    config.f.uncompress_cntx0_3 = 0xf;
    config.f.uncompress_cntx4_7 = 0xf;
    // config.f.limit_addr = 0; // Set dynamically
    // config.f.fifo_size = 0; // Set dynamically
    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC0_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    }

    config.f.out_data_format = row_pool ? (to_underlying(DataFormat::Float16) | (exp_width << 2)) : unpB_dst_format_masked;
    config.f.haloize_mode    = 0;

    for (std::uint32_t i = 0; i < CONFIG_SIZE; i++)
    {
        cfg[THCON_SEC1_REG2_Out_data_format_ADDR32 + i] = config.val[i];
    }

    std::uint32_t unpA_x_end = (unpA_face_r_dim == 0) ? 1 : (unpA_face_r_dim << 4) - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, (unpB_face_r_dim << 4) - 1, 0x0);

    // Program base address for all 2 sections (each section address is loaded to corresponding context)
    // Load dummy data to unused location if face height is 0
    const std::uint32_t Dest_cntx0_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    const std::uint32_t Dest_cntx1_address         = unpA_face_r_dim == 0 ? 22 * 16 : 4 * 16;
    cfg[THCON_SEC0_REG5_Dest_cntx0_address_ADDR32] = Dest_cntx0_address | (Dest_cntx1_address << 16);

    // Program unpacker0 per context x_dim (face size in l1)
    // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
    const std::uint32_t face_dim                 = unpA_face_r_dim * FACE_C_DIM;
    cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32] = face_dim | (face_dim << 16);

    constexpr std::uint32_t face_dim_16x16 = FACE_R_DIM * FACE_C_DIM;
    regfile[p_gpr_unpack::FACE_DIM_16x16]  = (face_dim_16x16 / 1) | ((face_dim_16x16 / 1) << 16);
    regfile[p_gpr_unpack::FACE_DIM_8x16]   = (face_dim_16x16 / 2) | ((face_dim_16x16 / 2) << 16);
    regfile[p_gpr_unpack::FACE_DIM_4x16]   = (face_dim_16x16 / 4) | ((face_dim_16x16 / 4) << 16);
    regfile[p_gpr_unpack::FACE_DIM_2x16]   = (face_dim_16x16 / 8) | ((face_dim_16x16 / 8) << 16);
    regfile[p_gpr_unpack::FACE_DIM_1x16]   = (face_dim_16x16 / 16) | ((face_dim_16x16 / 16) << 16);
    sync_regfile_write(p_gpr_unpack::FACE_DIM_1x16);

    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4);

    // Enable address counter for unpacker ch1/dst address
    // final address is calculated as: Dest_cntx0/1_address + address_counter_ch1
    // used for face by face unpacking of entire tile into srcA
    cfg[UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_ADDR32] = 0x1 << UNP0_ADD_DEST_ADDR_CNTR_add_dest_addr_cntr_SHAMT;

    // Clear context ID
    reset_config_context();
}

template <std::uint32_t UNP_SEL = p_setadc::UNP_AB>
inline void config_unpacker_x_end(const std::uint32_t face_r_dim)
{
    static_assert(UNP_SEL == p_setadc::UNP_A || UNP_SEL == p_setadc::UNP_B || UNP_SEL == p_setadc::UNP_AB, "UNP_SEL must be UNP_A, UNP_B, or UNP_AB");
    LLK_ASSERT(
        face_r_dim == 1 || face_r_dim == 2 || face_r_dim == 4 || face_r_dim == 8 || face_r_dim == FACE_R_DIM, "face_r_dim must be 1, 2, 4, 8, or FACE_R_DIM");

    switch (face_r_dim)
    {
        case 1:
            TTI_SETADCXX(UNP_SEL, 1 * FACE_C_DIM - 1, 0x0);
            break;
        case 2:
            TTI_SETADCXX(UNP_SEL, 2 * FACE_C_DIM - 1, 0x0);
            break;
        case 4:
            TTI_SETADCXX(UNP_SEL, 4 * FACE_C_DIM - 1, 0x0);
            break;
        case 8:
            TTI_SETADCXX(UNP_SEL, 8 * FACE_C_DIM - 1, 0x0);
            break;
        default:
            TTI_SETADCXX(UNP_SEL, FACE_R_DIM * FACE_C_DIM - 1, 0x0);
            break;
    }
}

inline void wait_for_dest_available()
{
    t6_semaphore_wait_on_max<p_stall::STALL_UNPACK>(semaphore::UNPACK_TO_DEST);
}

inline void unpack_to_dest_tile_done(std::uint32_t &context_id)
{
    t6_semaphore_post<p_stall::UNPACK0>(semaphore::UNPACK_TO_DEST);
    TTI_WRCFG(p_gpr_unpack::UNPACK_STRIDE, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32); // Restore unpack stride
    // Restore config context
    if (context_id == 0)
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(4 * 16);
    }
    else
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(0);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(4 * 16);
    }
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x4); // re-enable address bit swizzle
}

inline void set_dst_write_addr(const std::uint32_t &context_id, const std::uint32_t &unpack_dst_format)
{
    std::uint32_t dst_byte_addr = 16 * (4 + mailbox_read(ThreadId::MathThreadId));  // Apply fixed offset of 4*16 to dest address
    TTI_SETC16(SRCA_SET_Base_ADDR32, 0x0);                                          // Disable address bit swizzle
    TTI_RDCFG(p_gpr_unpack::UNPACK_STRIDE, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32); // Save current stride
    std::uint32_t unpA_ch1_x_stride = (unpack_dst_format & 0x3) == to_underlying(DataFormat::Float32)   ? 4
                                      : (unpack_dst_format & 0x3) == to_underlying(DataFormat::Float16) ? 2
                                                                                                        : 1;
    std::uint32_t unpA_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpA_ch1_x_stride;
    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_ch1_z_stride << UNP0_ADDR_CTRL_ZW_REG_1_Zstride_SHAMT), 0, LO_16(p_gpr_unpack::TMP_LO));
    TTI_WRCFG(p_gpr_unpack::TMP_LO, p_cfg::WRCFG_32b, UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32); // Set unpack stride
    if (context_id == 0)
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx0_RMW>(1);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx0_address_RMW>(dst_byte_addr);
    }
    else
    {
        cfg_reg_rmw_tensix<THCON_SEC0_REG2_Unpack_if_sel_cntx1_RMW>(1);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx1_address_RMW>(dst_byte_addr);
    }
}

// READERS FOR STRUCTS

inline unpack_tile_descriptor_t read_unpack_tile_descriptor_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr *cfg)
{
    unpack_tile_descriptor_u tile_descriptor = {.val = 0};

    tile_descriptor.val[0] = cfg[reg_addr];
    tile_descriptor.val[1] = cfg[reg_addr + 1];
    tile_descriptor.val[2] = cfg[reg_addr + 2];
    tile_descriptor.val[3] = cfg[reg_addr + 3];

    return tile_descriptor.f;
}

inline std::array<unpack_tile_descriptor_t, NUM_UNPACKERS> read_unpack_tile_descriptor()
{
    std::array<unpack_tile_descriptor_t, NUM_UNPACKERS> tile_descriptor_vec;
    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    tile_descriptor_vec[0] = read_unpack_tile_descriptor_helper(THCON_SEC0_REG0_TileDescriptor_ADDR32, cfg);
    tile_descriptor_vec[1] = read_unpack_tile_descriptor_helper(THCON_SEC1_REG0_TileDescriptor_ADDR32, cfg);

    return tile_descriptor_vec;
}

inline unpack_config_t read_unpack_config_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr *cfg)
{
    unpack_config_u config;

    config.val[0] = cfg[reg_addr];
    config.val[1] = cfg[reg_addr + 1];
    config.val[2] = cfg[reg_addr + 2];
    config.val[3] = cfg[reg_addr + 3];

    return config.f;
}

inline std::array<unpack_config_t, NUM_UNPACKERS> read_unpack_config()
{
    std::array<unpack_config_t, NUM_UNPACKERS> config_vec;
    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    config_vec[0] = read_unpack_config_helper(THCON_SEC0_REG2_Out_data_format_ADDR32, cfg);
    config_vec[1] = read_unpack_config_helper(THCON_SEC1_REG2_Out_data_format_ADDR32, cfg);

    return config_vec;
}

inline alu_config_t read_alu_config()
{
    alu_config_u config;
    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    config.val = cfg[ALU_ROUNDING_MODE_Fpu_srnd_en_ADDR32];

    return config.f;
}

enum class UnpackerProgramType
{
    ProgramByTile,
    ProgramByFace,
};

/**
 * Checks whether unpacker A tile descriptor and config match the expected formats and dimensions.
 *
 * @param unpA_src_format   Expected input data format for unpacker A (context 0)
 * @param unpA_dst_format   Expected output data format for unpacker A (context 0)
 * @param unpA_face_r_dim   Expected face row dimension for unpacker A (default FACE_R_DIM)
 * @param unpA_num_faces    Expected number of faces for unpacker A (default TILE_NUM_FACES)
 * @param nop_count         Number of nop operations to ensure configuration writes complete (default 10)
 * @return true if unpacker A configuration matches all expected values, false otherwise
 */
template <UnpackerProgramType program_type = UnpackerProgramType::ProgramByTile>
__attribute__((noinline)) bool is_unpacker_A_configured_correctly(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces  = TILE_NUM_FACES,
    const std::uint32_t nop_count       = 10)
{
    // Ensure configuration writes complete before subsequent operations
    tensix_sync();
    for (std::uint32_t i = 0; i < nop_count; i++)
    {
        asm volatile("nop");
    }

    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    // tile_descriptor[0] word 0: in_data_format at bits [3:0]
    const std::uint32_t td_word0 = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32];
    // unpack_config[0] word 0: out_data_format at bits [3:0]
    const std::uint32_t cfg_word0 = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32];

    if ((td_word0 & DATA_FORMAT_CONFIG_MASK) != (unpA_src_format & DATA_FORMAT_CONFIG_MASK) ||
        (cfg_word0 & DATA_FORMAT_CONFIG_MASK) != (unpA_dst_format & DATA_FORMAT_CONFIG_MASK))
    {
        return false;
    }

    if constexpr (program_type == UnpackerProgramType::ProgramByTile)
    {
        const std::uint32_t face_dim               = unpA_face_r_dim * FACE_C_DIM;
        const std::uint32_t tile_x_dim_cntx0_value = cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32];
        return tile_x_dim_cntx0_value == (face_dim | (face_dim << 16));
    }
    else
    {
        // tile_descriptor[0] word 1: z_dim at bits [31:16]
        const std::uint32_t td_word1 = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1];
        return (td_word1 >> 16) == unpA_num_faces;
    }
}

/**
 * Checks whether the unpacker tile descriptor and config match the expected formats and dimensions.
 *
 * @param unpA_src_format   Expected input data format for unpacker A (context 0)
 * @param unpA_dst_format   Expected output data format for unpacker A (context 0)
 * @param unpB_src_format   Expected input data format for unpacker B (context 1)
 * @param unpB_dst_format   Expected output data format for unpacker B (context 1)
 * @param unpA_face_r_dim   Expected face row dimension for unpacker A (default FACE_R_DIM)
 * @param unpB_face_r_dim   Expected face row dimension for unpacker B (default FACE_R_DIM)
 * @param unpA_num_faces    Expected number of faces for unpacker A (default TILE_NUM_FACES)
 * @param unpB_num_faces    Expected number of faces for unpacker B (default TILE_NUM_FACES)
 * @param nop_count         Number of nop operations to ensure configuration writes complete (default 80)
 * @return true if the current unpacker configuration matches all expected values, false otherwise
 */
template <UnpackerProgramType program_type = UnpackerProgramType::ProgramByTile>
__attribute__((noinline)) bool are_unpackers_AB_configured_correctly(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpB_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpA_num_faces  = TILE_NUM_FACES,
    const std::uint32_t unpB_num_faces  = TILE_NUM_FACES,
    const std::uint32_t nop_count       = 10)
{
    // Ensure configuration writes complete before subsequent operations
    tensix_sync();
    for (std::uint32_t i = 0; i < nop_count; i++)
    {
        asm volatile("nop");
    }

    volatile std::uint32_t tt_reg_ptr *cfg = get_cfg_pointer();

    // tile_descriptor word 0: in_data_format at bits [3:0], x_dim at bits [31:16]
    const std::uint32_t td0_word0 = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32];
    const std::uint32_t td1_word0 = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32];
    // unpack_config word 0: out_data_format at bits [3:0]
    const std::uint32_t cfg0_word0 = cfg[THCON_SEC0_REG2_Out_data_format_ADDR32];
    const std::uint32_t cfg1_word0 = cfg[THCON_SEC1_REG2_Out_data_format_ADDR32];

    if ((td0_word0 & DATA_FORMAT_CONFIG_MASK) != (unpA_src_format & DATA_FORMAT_CONFIG_MASK) ||
        (cfg0_word0 & DATA_FORMAT_CONFIG_MASK) != (unpA_dst_format & DATA_FORMAT_CONFIG_MASK) ||
        (td1_word0 & DATA_FORMAT_CONFIG_MASK) != (unpB_src_format & DATA_FORMAT_CONFIG_MASK) ||
        (cfg1_word0 & DATA_FORMAT_CONFIG_MASK) != (unpB_dst_format & DATA_FORMAT_CONFIG_MASK))
    {
        return false;
    }

    if constexpr (program_type == UnpackerProgramType::ProgramByTile)
    {
        const std::uint32_t face_dim_a             = unpA_face_r_dim * FACE_C_DIM;
        const std::uint32_t tile_x_dim_cntx0_value = cfg[THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32];
        return tile_x_dim_cntx0_value == (face_dim_a | (face_dim_a << 16)) && (td1_word0 >> 16) == unpB_face_r_dim * FACE_C_DIM;
    }
    else
    {
        // tile_descriptor word 1: z_dim at bits [31:16]
        const std::uint32_t td0_word1 = cfg[THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1];
        const std::uint32_t td1_word1 = cfg[THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1];
        return (td0_word1 >> 16) == unpA_num_faces && (td1_word1 >> 16) == unpB_num_faces;
    }
}

} // namespace ckernel::unpacker
