// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::pack
{
// Pack untilize stride config register struct
// Covers ADDR32 54-55 (THCON_PACKER0_REG1)
constexpr std::uint32_t NUM_WORDS_PACK_UNTILIZE_STRIDE_CFG = 2;

struct pack_untilize_stride_cfg_t
{
    // word 0 — ADDR32 54
    std::uint32_t edge_mask_mode : 2;  // EDGE_MASK_MODE
    std::uint32_t src_z_stride   : 8;  // PACK_UNTILIZE_SRC_Z_STRIDE
    std::uint32_t dst_z_stride   : 16; // PACK_UNTILIZE_DST_Z_STRIDE
    std::uint32_t reserved0      : 6;

    // word 1 — ADDR32 55
    std::uint32_t stride_offset_0 : 16; // PACK_STRIDE_OFFSET_0
    std::uint32_t stride_offset_1 : 16; // PACK_STRIDE_OFFSET_1
};

static_assert(sizeof(pack_untilize_stride_cfg_t) == NUM_WORDS_PACK_UNTILIZE_STRIDE_CFG * sizeof(std::uint32_t));

union pack_untilize_stride_cfg_u
{
    std::uint32_t val[NUM_WORDS_PACK_UNTILIZE_STRIDE_CFG];
    pack_untilize_stride_cfg_t f;
};

constexpr static std::uint32_t TRISC_ID = 2;
static std::uint32_t clear_dest_bank_id = 0;

inline void _update_clear_dest_bank_id_()
{
    clear_dest_bank_id = 1 - clear_dest_bank_id;
}

// Edge Mask configs for pack reduce
constexpr std::uint32_t EDGE_MASK_MODE_ZERO    = 0x0; // masked datums -> 0
constexpr std::uint32_t EDGE_MASK_MODE_NEG_INF = 0x1; // masked datums -> -inf

constexpr std::uint32_t EDGE_MASK_ROW_DATUMS_NONE     = 0x0000; // mask no datums in a row
constexpr std::uint32_t EDGE_MASK_ROW_DATUMS_ALL      = 0xFFFF; // mask all datums in a row
constexpr std::uint32_t EDGE_MASK_ROW_DATUMS_EXCEPT_0 = 0xFFFE; // mask datums[1:15], leave datum[0]

constexpr std::uint32_t EDGE_MASK_FACE_ALL_ROWS_MASK_0  = 0x00000000; // apply mask0 to all rows
constexpr std::uint32_t EDGE_MASK_FACE_ALL_ROWS_MASK_1  = 0x55555555; // apply mask1 to all rows
constexpr std::uint32_t EDGE_MASK_FACE_ROW0_MASK_1      = 0x00000001; // apply mask1 to row 0, mask0 to rows [1:15]
constexpr std::uint32_t EDGE_MASK_FACE_ROW0_ROW8_MASK_1 = 0x00010001; // apply mask1 to row 0, row 8

namespace
{
// L1 outputs for Float16 / Float16_b pack input.
inline bool is_quasar_pack_f16_src_l1_dst_supported(const DataFormat dst)
{
    switch (dst)
    {
        case DataFormat::Float16:
        case DataFormat::Float16_b:
        case DataFormat::Fp8R:
        case DataFormat::Fp8P:
        case DataFormat::MxFp8R:
        case DataFormat::MxFp8P:
        case DataFormat::MxFp6R:
        case DataFormat::MxFp6P:
        case DataFormat::MxFp4:
        case DataFormat::MxInt8:
        case DataFormat::MxInt4:
        case DataFormat::MxInt2:
            return true;
        default:
            return false;
    }
}

// L1 outputs allowed for Float32 pack input (Quasar Packer Gasket table).
inline bool is_quasar_pack_f32_src_l1_dst_supported(const DataFormat dst)
{
    return dst == DataFormat::Float32 || dst == DataFormat::Tf32 || is_quasar_pack_f16_src_l1_dst_supported(dst);
}
} // namespace

/**
 * Whether packing from dest register format \p pack_src_format to L1 format \p pack_dst_format is
 * supported on Quasar for dynamic packer reconfiguration.
 *
 * Input is programmed via THCON PACKER IN_DATA_FORMAT; L1 output format is taken from the buffer
 * descriptor at pack time. This function validates the gasket conversion pair only.
 *
 * Rules follow the Quasar Packer Format Conversions table (gasket outside TDMA).
 */
__attribute__((noinline, optimize("no-jump-tables"))) inline bool is_quasar_pack_reconfig_pair_supported(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format)
{
    const DataFormat src = static_cast<DataFormat>(pack_src_format);
    const DataFormat dst = static_cast<DataFormat>(pack_dst_format);

    switch (src)
    {
        // -------------------------------------------------------------------------
        // Float32 — Float32, TF32, Float16, Float16_b, FP8*, MX*, MXINT*
        case DataFormat::Float32:
            return is_quasar_pack_f32_src_l1_dst_supported(dst);

        // -------------------------------------------------------------------------
        // Float16 / Float16_b — FP8/MX/L1 F16 set.
        case DataFormat::Float16:
        case DataFormat::Float16_b:
            return is_quasar_pack_f16_src_l1_dst_supported(dst);

        // -------------------------------------------------------------------------
        // INT32 — INT32, INT8, UINT8
        case DataFormat::Int32:
            switch (dst)
            {
                case DataFormat::Int32:
                case DataFormat::Int8:
                case DataFormat::UInt8:
                    return true;
                default:
                    return false;
            }

        // -------------------------------------------------------------------------
        // INT8 — INT8
        case DataFormat::Int8:
            return dst == DataFormat::Int8;

        // -------------------------------------------------------------------------
        // UINT8 — UINT8
        case DataFormat::UInt8:
            return dst == DataFormat::UInt8;

        // -------------------------------------------------------------------------
        // UINT16 — UINT16
        case DataFormat::UInt16:
            return dst == DataFormat::UInt16;

        // -------------------------------------------------------------------------
        // INT16 — INT16
        case DataFormat::Int16:
            return dst == DataFormat::Int16;

        default:
            return false;
    }
}

} // namespace ckernel::pack
