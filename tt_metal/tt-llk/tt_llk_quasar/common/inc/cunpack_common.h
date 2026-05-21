// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::unpack
{
// Unpack tilize config register struct
// Covers ADDR32 18-20 for UNPACKER0, ADDR32 30-32 for UNPACKER1
constexpr std::uint32_t NUM_WORDS_UNPACK_TILIZE_CFG = 3;

struct unpack_tilize_cfg_t
{
    // word 0 — ADDR32 18 or 30
    std::uint32_t src_z_stride      : 16; // TILIZE_SRC_Z_STRIDE
    std::uint32_t dst_z_stride      : 8;  // TILIZE_DST_Z_STRIDE
    std::uint32_t stride_val_source : 1;  // STRIDE_VAL_SOURCE
    std::uint32_t stride_no_write   : 1;  // STRIDE_NO_WRITE
    std::uint32_t reserved0         : 6;

    // word 1 — ADDR32 19 or 31
    std::uint32_t stride_mask_val : 32; // STRIDE_MASK_VAL

    // word 2 — ADDR32 20 or 32
    std::uint32_t stride_offset_0 : 16; // STRIDE_OFFSET_0
    std::uint32_t stride_offset_1 : 16; // STRIDE_OFFSET_1
};

static_assert(sizeof(unpack_tilize_cfg_t) == NUM_WORDS_UNPACK_TILIZE_CFG * sizeof(std::uint32_t));

union unpack_tilize_cfg_u
{
    std::uint32_t val[NUM_WORDS_UNPACK_TILIZE_CFG];
    unpack_tilize_cfg_t f;
};

// Number of rows for Unpack functions
constexpr static std::uint32_t UNPACR_STRIDE_MAX_ROWS = 8;
constexpr static std::uint32_t TRISC_ID               = 0;

/**
 * Whether reprogramming OUT_DATA_FORMAT from buffer-descriptor \p unpack_src_format (L1) to
 * \p unpack_dst_format is supported on Quasar for the given FP32 dest-accumulation mode and unpack path.
 *
 * \param en_32bit_dest   Enables TF32 / FP32 register paths where the ISA requires it.
 * \param unpack_to_dest        True: Unpack-to-Dest or Unpack-to-SrcS; false: SrcA/SrcB.
 *
 * Rules follow the Quasar Unpacker Format Conversions table (gasket outside TDMA).
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_quasar_unpack_reconfig_pair_supported(
    const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const bool en_32bit_dest, const bool unpack_to_dest)
{
    const DataFormat src = static_cast<DataFormat>(unpack_src_format);
    const DataFormat dst = static_cast<DataFormat>(unpack_dst_format);

    switch (src)
    {
        // -------------------------------------------------------------------------
        // Float32 — FP32→FP32 Dest/SrcS only; FP32→TF32 SrcA/B only + FP32 dest acc
        case DataFormat::Float32:
            switch (dst)
            {
                case DataFormat::Float32:
                    return unpack_to_dest;
                case DataFormat::Tf32:
                    return en_32bit_dest && !unpack_to_dest;
                case DataFormat::Float16:
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // TF32 — identity / FP32 split by path; FP16/BF16 always
        case DataFormat::Tf32:
            switch (dst)
            {
                case DataFormat::Tf32:
                    return en_32bit_dest && !unpack_to_dest;
                case DataFormat::Float32:
                    return en_32bit_dest && unpack_to_dest;
                case DataFormat::Float16:
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // Float16 / FP8R / FP8P — TF32 not for Unpack-to-Dest/SrcS
        case DataFormat::Float16:
        case DataFormat::Fp8R:
        case DataFormat::Fp8P:
            switch (dst)
            {
                case DataFormat::Tf32:
                    return !unpack_to_dest;
                case DataFormat::Float16:
                case DataFormat::Float16_b:
                case DataFormat::Fp8R:
                case DataFormat::Fp8P:
                case DataFormat::MxFp8R:
                case DataFormat::MxFp8P:
                case DataFormat::MxFp6R:
                case DataFormat::MxFp6P:
                case DataFormat::MxInt8:
                case DataFormat::MxInt4:
                case DataFormat::MxInt2:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // Float16_b — BF16→TF32 SrcA/B + FP32 dest acc
        case DataFormat::Float16_b:
            switch (dst)
            {
                case DataFormat::Tf32:
                    return en_32bit_dest && !unpack_to_dest;
                case DataFormat::Float16_b:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // MXFP4 — TF32 / 2× not for Unpack-to-Dest/SrcS
        case DataFormat::MxFp4:
            switch (dst)
            {
                case DataFormat::Float16:
                case DataFormat::Float16_b:
                    return true;
                case DataFormat::Tf32:
                    return en_32bit_dest && !unpack_to_dest;
                case DataFormat::MxFp4_2x_A:
                case DataFormat::MxFp4_2x_B:
                    return !unpack_to_dest;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // INT32 — INT32→INT32 Dest/SrcS only
        case DataFormat::Int32:
            return dst == DataFormat::Int32 && unpack_to_dest;

        // -------------------------------------------------------------------------
        // UINT8 — UINT8→UINT8 (Trinity 2× omitted on Quasar)
        case DataFormat::UInt8:
            return dst == DataFormat::UInt8;

        // -------------------------------------------------------------------------
        // INT8 — TF32 SrcA/B + FP32 dest acc; INT8/BF16 unrestricted by path
        case DataFormat::Int8:
            switch (dst)
            {
                case DataFormat::Tf32:
                    return en_32bit_dest && !unpack_to_dest;
                case DataFormat::Float16_b:
                case DataFormat::Int8:
                    return true;
                default:
                    return false;
            }

        case DataFormat::Int16:
            return dst == DataFormat::Int16;

        case DataFormat::Int4:
            return dst == DataFormat::Int8;

        case DataFormat::UInt4:
            return dst == DataFormat::Int8 || dst == DataFormat::UInt8;

        default:
            return false;
    }
}
} // namespace ckernel::unpack
