// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"

namespace ckernel::packer
{
using DataFormatType = std::underlying_type_t<DataFormat>;

constexpr std::uint32_t PACK_CNT    = 4;
constexpr std::uint32_t NUM_PACKERS = 4; // Number of packers

constexpr std::uint32_t PACK_SEL(const std::uint32_t pack_count)
{
    return (pack_count == 1) ? 0x1 : (pack_count == 2) ? 0x3 : (pack_count == 4) ? 0xF : 0x0;
}

constexpr std::uint32_t replay_buf_offset = 16; // split replay buffer usage between fpu/sfpu
                                                // fist 16 for sfpu, next 16 for fpu

// Pack config
// Word-2 layout matches THCON_SEC[01]_REG1. The REG8 banks differ: bits 17-23 are not
// implemented as below (no All_pack_disable_zero_compress/_ovrd; Add_tile_header_size is
// at bit 17 per cfg_defines.h), so REG8 readback can't be interpreted with this struct.
typedef struct
{
    // word 0
    std::uint32_t row_ptr_section_size : 16;
    std::uint32_t exp_section_size     : 16;
    // word 1
    std::uint32_t l1_dest_addr : 32;
    // word 2
    std::uint32_t uncompress              : 1;             // bit0  Disable_zero_compress
    std::uint32_t add_l1_dest_addr_offset : 1;             // bit1
    std::uint32_t addr_cnt_context        : 2;             // bits2-3
    std::uint32_t out_data_format : DATA_FORMAT_BIT_COUNT; // bits4-7
    std::uint32_t in_data_format : DATA_FORMAT_BIT_COUNT;  // bits8-11
    std::uint32_t dis_shared_exp_assembler            : 1; // bit12
    std::uint32_t force_pack_per_max_xy_plane         : 1; // bit13
    std::uint32_t enable_out_fifo                     : 1; // bit14
    std::uint32_t sub_l1_tile_header_size             : 1; // bit15
    std::uint32_t src_if_sel                          : 1; // bit16  Source_interface_selection
    std::uint32_t all_pack_disable_zero_compress      : 4; // bits17-20
    std::uint32_t all_pack_disable_zero_compress_ovrd : 1; // bit21
    std::uint32_t add_tile_header_size                : 1; // bit22
    std::uint32_t reserved_1                          : 1; // bit23 unused
    std::uint32_t l1_src_addr                         : 8; // bits24-31
    // word 3
    std::uint32_t downsample_mask                    : 16;
    std::uint32_t downsample_shift_count             : 3;
    std::uint32_t read_mode                          : 1;
    std::uint32_t exp_threshold_en                   : 1;
    std::uint32_t pack_l1_acc_disable_pack_zero_flag : 2;
    std::uint32_t reserved_2                         : 1;
    std::uint32_t exp_threshold                      : 8;

} pack_config_t;

static_assert(sizeof(pack_config_t) == (sizeof(std::uint32_t) * 4));

typedef union
{
    std::uint32_t val[4];
    pack_config_t f;
} pack_config_u;

// Relu Config
typedef struct
{
    std::uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_src      : 1;
    std::uint32_t ALU_ACC_CTRL_Zero_Flag_disabled_dst      : 1;
    std::uint32_t STACC_RELU_ApplyRelu                     : 4;
    std::uint32_t STACC_RELU_ReluThreshold                 : 16;
    std::uint32_t DISABLE_RISC_BP_Disable_main             : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_trisc            : 3;
    std::uint32_t DISABLE_RISC_BP_Disable_ncrisc           : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_main   : 1;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_trisc  : 3;
    std::uint32_t DISABLE_RISC_BP_Disable_bmp_clear_ncrisc : 1;
} relu_config_t;

static_assert(sizeof(relu_config_t) == (sizeof(std::uint32_t)));

typedef union
{
    std::uint32_t val[1];
    relu_config_t r;
} relu_config_u;

// Dest rd control
typedef struct
{
    std::uint32_t PCK_DEST_RD_CTRL_Read_32b_data  : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Read_unsigned  : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Read_int8      : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Round_10b_mant : 1;
    std::uint32_t PCK_DEST_RD_CTRL_Reserved       : 28;
} dest_rd_ctrl_t;

static_assert(sizeof(dest_rd_ctrl_t) == (sizeof(std::uint32_t)));

typedef union
{
    std::uint32_t val;
    dest_rd_ctrl_t f;
} dest_rd_ctrl_u;

// PACK_EDGE_OFFSET_SEC[0:3] register structure
//
// Lower 16b represent a mask that is applied on a single row of one face on the packer output
// Higher 16b contain information about which TILE_ROW_SET_MAPPING register is used for each packer (only in PACK_EDGE_OFFSET_SEC0)
//
// There are 4 PACK_EDGE_OFFSET_SEC[0:3] registers and 4 TILE_ROW_SET_MAPPING[0:3] registers.
// TILE_ROW_SET_MAPPING[0:3] have 2 bits for each row inside a face that determine which PACK_EDGE_OFFSET_SEC[0:3] mask is used.
// Only PACK_EDGE_OFFSET_SEC0 register has higher 16b configured to determine TILE_ROW_SET_MAPPING[0:3] registers used for each packer.
// Other PACK_EDGE_OFFSET_SEC[1:3] registers are used only for the masks in the lower 16b.
typedef struct
{
    std::uint32_t mask                      : 16;
    std::uint32_t mode                      : 1;
    std::uint32_t tile_row_set_select_pack0 : 2;
    std::uint32_t tile_row_set_select_pack1 : 2;
    std::uint32_t tile_row_set_select_pack2 : 2;
    std::uint32_t tile_row_set_select_pack3 : 2;
    std::uint32_t reserved                  : 7;
} pck_edge_offset_t;

static_assert(sizeof(pck_edge_offset_t) == (sizeof(std::uint32_t)));

typedef union
{
    std::uint32_t val;
    pck_edge_offset_t f;
} pck_edge_offset_u;

// Pack counters
typedef struct
{
    std::uint32_t pack_per_xy_plane        : 8;
    std::uint32_t pack_reads_per_xy_plane  : 8;
    std::uint32_t pack_xys_per_til         : 7;
    std::uint32_t pack_yz_transposed       : 1;
    std::uint32_t pack_per_xy_plane_offset : 8;
} pack_counters_t;

static_assert(sizeof(pack_counters_t) == (sizeof(std::uint32_t)));

typedef union
{
    std::uint32_t val;
    pack_counters_t f;
} pack_counters_u;

// Set unpacker offsets to 0, except for unpacker 0, channel 1, X, which is the tile X dimension
inline void packer_addr_counter_init()
{
    TTI_SETADCXY(0b100, 0, 0, 0, 0, 0b1011);
    TTI_SETADCZW(0b100, 0, 0, 0, 0, 0b1111);
}

/**
 * \brief Returns true if `out_l1` is a valid output of the FP32 late-conversion column.
 *
 * Ref: Wormhole Packers/FormatConversion.md, late table "From FP32".
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_fp32_late_column_output(const DataFormat out_l1)
{
    switch (out_l1)
    {
        case DataFormat::Float32:
        case DataFormat::Float16_b:
        case DataFormat::Float16:
        case DataFormat::Lf8:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b:
            return true;
        default:
            return false;
    }
}

/**
 * \brief Returns true if `out_l1` is a valid output of the combined late-conversion column.
 *
 * Combined column: "From TF32 or BF16 or E8M6 or FP16 or E5M7 or E5M6 or FP8".
 * This is exactly the FP32 column plus Tf32.
 */
__attribute__((noinline)) bool is_packer_combined_late_column_output(const DataFormat out_l1)
{
    return out_l1 == DataFormat::Tf32 || is_packer_fp32_late_column_output(out_l1);
}

/**
 * \brief Returns true if conversion is supported by EARLY packer conversion stage.
 *
 * This checks the Wormhole early conversion matrix only (Dst register format -> intermediate
 * format). For this API, `out_l1` is interpreted as the requested intermediate format code.
 *
 * Ref: Wormhole Packers/FormatConversion.md, "Early format conversion".
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_to_L1_early_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    switch (in_reg)
    {
        case DataFormat::Float32:
            return out_l1 == DataFormat::Float32 ||   // FP32 (identity)
                   out_l1 == DataFormat::Tf32 ||      // TF32
                   out_l1 == DataFormat::Float16_b || // BF16
                   out_l1 == DataFormat::Bfp8_b ||    // E8M6 (encoded as BFP8)
                   out_l1 == DataFormat::Int32 ||     // INT32 (bitcast)
                   out_l1 == DataFormat::Int8 ||      // INT8
                   out_l1 == DataFormat::UInt8;       // UINT8

        case DataFormat::Float16_b:
            return out_l1 == DataFormat::Tf32 ||      // TF32
                   out_l1 == DataFormat::Float16_b || // BF16
                   out_l1 == DataFormat::Bfp8_b ||    // E8M6 (encoded as BFP8)
                   out_l1 == DataFormat::Int8;        // INT8

        case DataFormat::Float16:
            return out_l1 == DataFormat::Float16 || // FP16
                   out_l1 == DataFormat::Bfp8 ||    // E5M7/E5M6 (encoded as BFP8a)
                   out_l1 == DataFormat::Lf8 ||     // FP8 (e5m2)
                   out_l1 == DataFormat::Int8;      // INT8

        case DataFormat::Int32:
            return out_l1 == DataFormat::Float32 ||   // FP32 (bitcast)
                   out_l1 == DataFormat::Tf32 ||      // TF32 (bitcast+round)
                   out_l1 == DataFormat::Float16_b || // BF16 (top 16b bitcast)
                   out_l1 == DataFormat::Int32 ||     // INT32 (identity)
                   out_l1 == DataFormat::Int8 ||      // INT8
                   out_l1 == DataFormat::UInt8;       // UINT8

        case DataFormat::UInt16: // INT16 identity path
            return out_l1 == DataFormat::UInt16;

        default:
            return false;
    }
}

/**
 * \brief Returns true if conversion is supported by LATE packer conversion stage.
 *
 * This checks the Wormhole late conversion matrix only (intermediate/LateFromFormat -> L1).
 *
 * Ref: Wormhole Packers/FormatConversion.md, "Late format conversion".
 */
__attribute__((noinline, optimize("no-jump-tables"))) bool is_packer_to_L1_late_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    switch (in_reg)
    {
        case DataFormat::Float32: // From FP32 column
            return is_packer_fp32_late_column_output(out_l1);

        // Combined column (TF32 / BF16 / E8M6 / FP16 / E5M7 / E5M6 / FP8).
        // Bfp4/Bfp2 and Bfp4_b/Bfp2_b are kept for backward compatibility with existing code.
        case DataFormat::Tf32:
        case DataFormat::Float16_b:
        case DataFormat::Bfp8_b:
        case DataFormat::Bfp4_b:
        case DataFormat::Bfp2_b:
        case DataFormat::Float16:
        case DataFormat::Bfp8:
        case DataFormat::Bfp4:
        case DataFormat::Bfp2:
        case DataFormat::Lf8:
            return is_packer_combined_late_column_output(out_l1);

        case DataFormat::Int32: // From INT32 column
            return out_l1 == DataFormat::Int32;

        case DataFormat::UInt16: // From INT16 column
            return out_l1 == DataFormat::UInt16;

        case DataFormat::Int8: // From INT8/UINT8 column
        case DataFormat::UInt8:
            return out_l1 == DataFormat::Int8 || out_l1 == DataFormat::UInt8;

        case DataFormat::UInt32:
            return out_l1 == DataFormat::UInt32;

        default:
            return false;
    }
}

/**
 * \brief Returns true if either EARLY or LATE packer conversion stage supports the conversion.
 */
__attribute__((noinline)) bool is_packer_to_L1_conversion_supported(const DataFormat in_reg, const DataFormat out_l1)
{
    return is_packer_to_L1_early_conversion_supported(in_reg, out_l1) || is_packer_to_L1_late_conversion_supported(in_reg, out_l1);
}

/**
 * @brief Per-packer BFP Exp_section_size (the THCON_SEC*_REG*_Exp_section_size cfg field), in 16-byte chunks.
 *
 * Chosen so a packer's data-stream Addr equals face `idx`'s datum address in the tile the unpacker reads.
 * Per OutputAddressGenerator.md, a BFP packer computes its data-stream Addr (in 16B chunks) as
 *     Addr = L1_Dest_addr + !Sub_l1_tile_header_size  (+ Packer0InitialAddr for idx >= 1)  + Exp_section_size
 * Take the tile base as 0 so the values below are just offsets within the tile. Packer 0's L1_Dest_addr is
 * the tile base, so the terms in its Addr before adding Exp_section_size sum to 0 + 1 (own header) = 1, saved
 * as Packer0InitialAddr. For idx >= 1, set_packer_l1_offset programs L1_Dest_addr = idx * l1_offset - 1 =
 * idx - 1 (BFP l1_offset == 1); the own header (+1) and the shared Packer0InitialAddr (+1) are then added:
 * (idx - 1) + 1 + 1 = idx + 1. Packer 0 is 0 + 1 = 1, also idx + 1, so the terms before Exp_section_size
 * always sum to idx + 1.
 *
 * Per UNPACR_Regular.md, face `idx`'s datums sit at, in 16B chunks from the tile base:
 *     (1 + DigestSize)                        tile header             (DigestSize == 0 -> 1)
 *   + ceil(NumExponents/16) == num_faces      exponent section        (one exponent chunk per face)
 *   + idx * datum_bytes                       datums of the idx preceding faces
 *
 * Equating Addr to that datum address and solving for Exp_section_size:
 *     Exp_section_size = (1 + num_faces + idx * datum_bytes) - (idx + 1) = num_faces + idx * (datum_bytes - 1)
 *
 * @param idx: Packer index 0..3 (SEC0_REG1 / SEC0_REG8 / SEC1_REG1 / SEC1_REG8). Only called with idx 1..3 —
 *             packer 0's value (EXP0 / the partial_face case) is written separately.
 * @param datum_bytes: Per-face data size in 16B chunks (Bfp8 16, Bfp4 8, Bfp2 4; == 16 * the ISA DatumSizeBytes 1 / 0.5 / 0.25).
 * @param num_faces: Number of faces in the tile (1, 2, or 4).
 * @return Exp_section_size for packer `idx`, in 16-byte chunks.
 *
 * @note Packers 2 and 3 write faces 2 and 3, which exist only in a 4-face tile, so num_faces is 4 whenever they
 *       fire; the code uses 4 for idx >= 2 unconditionally. This is also needed for correctness: EXP2/EXP3 are
 *       not re-cached on a data-format reconfig (only EXP0/EXP1 are), so a config at num_faces < 4 must not
 *       leave them holding a smaller value that a later 4-face pack would read. Packers 0 and 1 are re-cached
 *       every reconfig, so they use the actual num_faces.
 * @note Wormhole only — Blackhole uses a single Exp_section_size value. Centralizes the {1*D, 2*D, 3*D} triple
 *       hand-expanded in @ref cache_exponential_section_sizes_in_gprs and @ref set_packer_config.
 *
 * @see https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor/Packers/OutputAddressGenerator.md
 * @see https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor/UNPACR_Regular.md
 */
constexpr std::uint32_t bfp_exp_section_size(const std::uint32_t idx, const std::uint32_t datum_bytes, const std::uint32_t num_faces)
{
    // idx >= 2 packers only ever fire at num_faces == 4 and are not re-cached on reconfig; use 4 for them.
    const std::uint32_t effective_num_faces = (idx >= 2) ? 4 : num_faces;
    // face idx datum address (1 + effective_num_faces + idx*datum_bytes, UNPACR_Regular.md)
    //   minus Addr's terms before Exp_section_size (idx + 1, OutputAddressGenerator.md; see above)
    return 1 + effective_num_faces + idx * datum_bytes - (idx + 1);
}

// Mantissa bytes for one face row (16 datums = FACE_C_DIM) of each BFP format, i.e. the `datum_bytes`
// @ref bfp_exp_section_size expects: 16 datums * per-datum mantissa size (Bfp8 = 1 B, Bfp4 = 1/2 B,
// Bfp2 = 1/4 B). Named once here so the packer exp-section-size call sites carry no bare 16/8/4.
constexpr std::uint32_t bfp8_row_bytes = 16;
constexpr std::uint32_t bfp4_row_bytes = 8;
constexpr std::uint32_t bfp2_row_bytes = 4;

// This function saves the exponential section size/required offsets to GPR for reconfiguring
// of data format for packer. These registers are not explicitly used by the packer during
// operation, thus we do not need to use a semaphore to wait for the packer to finish before
// performing these MMIOs.
template <bool reconfiguring>
inline void cache_exponential_section_sizes_in_gprs(const std::uint32_t num_faces = 4, const bool partial_face = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    regfile[p_gpr_pack::EXP0_SEC_SIZE_BFP]  = (partial_face ? 1 : num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP8] = bfp_exp_section_size(1 /* index */, bfp8_row_bytes, num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;

    if constexpr (!reconfiguring)
    {
        regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP8] = bfp_exp_section_size(2 /* index */, bfp8_row_bytes, num_faces)
                                                  << THCON_SEC0_REG8_Exp_section_size_SHAMT;
        regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP8] = bfp_exp_section_size(3 /* index */, bfp8_row_bytes, num_faces)
                                                  << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    }

    regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP4] = bfp_exp_section_size(1 /* index */, bfp4_row_bytes, num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;

    if constexpr (!reconfiguring)
    {
        regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP4] = bfp_exp_section_size(2 /* index */, bfp4_row_bytes, num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
        regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP4] = bfp_exp_section_size(3 /* index */, bfp4_row_bytes, num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
    }

    regfile[p_gpr_pack::EXP1_SEC_SIZE_BFP2] = bfp_exp_section_size(1 /* index */, bfp2_row_bytes, num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;

    if constexpr (!reconfiguring)
    {
        regfile[p_gpr_pack::EXP2_SEC_SIZE_BFP2] = bfp_exp_section_size(2 /* index */, bfp2_row_bytes, num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
        regfile[p_gpr_pack::EXP3_SEC_SIZE_BFP2] = bfp_exp_section_size(3 /* index */, bfp2_row_bytes, num_faces) << THCON_SEC0_REG8_Exp_section_size_SHAMT;
        sync_regfile_write(p_gpr_pack::EXP3_SEC_SIZE_BFP2);
    }
    else
    {
        sync_regfile_write(p_gpr_pack::EXP1_SEC_SIZE_BFP2);
    }
}

inline void set_packer_strides(const std::uint32_t pack_src_format)
{
    std::uint32_t x_stride = datum_size_in_bytes(pack_src_format);
    std::uint32_t y_stride = FACE_C_DIM * x_stride; // Y steps across a row of FACE_C_DIM datums (== FACE_R_DIM for square faces)
    std::uint32_t z_stride = FACE_R_DIM * y_stride; // Z steps a full face of FACE_R_DIM rows (== FACE_C_DIM for square faces)
    std::uint32_t w_stride = TILE_NUM_FACES * z_stride;

    std::uint32_t xy_stride = (x_stride << PCK0_ADDR_CTRL_XY_REG_0_Xstride_SHAMT) | (y_stride << PCK0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT);
    std::uint32_t zw_stride = (z_stride << PCK0_ADDR_CTRL_ZW_REG_0_Zstride_SHAMT) | (w_stride << PCK0_ADDR_CTRL_ZW_REG_0_Wstride_SHAMT);
    TT_SETDMAREG(0, LOWER_HALFWORD(xy_stride), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(xy_stride), 0, HI_16(p_gpr_pack::TMP0));
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_XY_REG_0_Xstride_ADDR32);
    TT_SETDMAREG(0, LOWER_HALFWORD(zw_stride), 0, LO_16(p_gpr_pack::TMP0));
    TT_SETDMAREG(0, UPPER_HALFWORD(zw_stride), 0, HI_16(p_gpr_pack::TMP0));
    TTI_WRCFG(p_gpr_pack::TMP0, p_cfg::WRCFG_32b, PCK0_ADDR_CTRL_ZW_REG_0_Zstride_ADDR32);
    TTI_NOP;
    TTI_NOP;
}

template <bool is_fp32_dest_acc_en>
inline void set_packer_config(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t num_faces = 4, const bool partial_face = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    // Set packer config
    pack_config_u config;
    for (std::uint32_t i = 0; i < 4; i++)
    {
        config.val[i] = 0;
    }

    config.f.exp_section_size =
        ((pack_dst_format == to_underlying(DataFormat::Lf8)) || (masked_data_format(pack_dst_format) == to_underlying(DataFormat::Int8)))
            ? 0
            : (partial_face ? 1 : num_faces); // set to num_faces as exp section size is not used for non-bfp formats except for lf8/int8

    config.f.uncompress      = 1;
    config.f.out_data_format = pack_dst_format;
    config.f.in_data_format  = pack_src_format;

    config.f.exp_threshold_en = 0;
    config.f.exp_threshold    = 0;

    // Workaround for bug in HW: tenstorrent/budabackend#1394
    if constexpr (is_fp32_dest_acc_en)
    {
        if (IS_BFP_A_FORMAT(pack_dst_format))
        {
            config.f.exp_threshold_en = 1;
            config.f.exp_threshold    = 113;
        }
    }

    // Program:
    // THCON_SEC0_REG1_Row_start_section_size = cfg_reg_array[1][0 +: 16];
    // THCON_SEC0_REG1_Exp_section_size = cfg_reg_array[1][16 +: 16];
    // This is filled with garbage, and will be set up on every pack:
    //           THCON_SEC0_REG1_L1_Dest_addr = cfg_reg_array[1][32 +: 32];
    // THCON_SEC0_REG1_Disable_zero_compress = cfg_reg_array[1][64 +: 1];
    // THCON_SEC0_REG1_Add_l1_dest_addr_offset = cfg_reg_array[1][65 +: 1];
    // THCON_SEC0_REG1_Addr_cnt_context = cfg_reg_array[1][66 +: 2];
    // THCON_SEC0_REG1_Out_data_format = cfg_reg_array[1][68 +: 4];
    // THCON_SEC0_REG1_In_data_format = cfg_reg_array[1][72 +: 4];
    // THCON_SEC0_REG1_Dis_shared_exp_assembler = cfg_reg_array[1][76 +: 1];
    // THCON_SEC0_REG1_Force_pack_per_max_xy_plane = cfg_reg_array[1][77 +: 1];
    // THCON_SEC0_REG1_Enable_out_fifo = cfg_reg_array[1][78 +: 1];
    // THCON_SEC0_REG1_Sub_l1_tile_header_size = cfg_reg_array[1][79 +: 1];
    // THCON_SEC0_REG1_Source_interface_selection = cfg_reg_array[1][80 +: 1];
    // THCON_SEC0_REG1_All_pack_disable_zero_compress = cfg_reg_array[1][81 +: 4];
    // THCON_SEC0_REG1_All_pack_disable_zero_compress_ovrd = cfg_reg_array[1][85 +: 1];
    // THCON_SEC0_REG1_Add_tile_header_size = cfg_reg_array[1][86 +: 1];
    // THCON_SEC0_REG1_Unused00 = cfg_reg_array[1][87 +: 1];
    // THCON_SEC0_REG1_L1_source_addr = cfg_reg_array[1][88 +: 8];
    // THCON_SEC0_REG1_Downsample_mask = cfg_reg_array[1][96 +: 16];
    // THCON_SEC0_REG1_Downsample_shift_count = cfg_reg_array[1][112 +: 3];
    // THCON_SEC0_REG1_Read_mode = cfg_reg_array[1][115 +: 1];
    // THCON_SEC0_REG1_Exp_threshold_en = cfg_reg_array[1][116 +: 1];
    // THCON_SEC0_REG1_Unused1 = cfg_reg_array[1][117 +: 3];
    // THCON_SEC0_REG1_Exp_threshold = cfg_reg_array[1][120 +: 8];
    // for (uint i=0; i<4; i++) cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32+i]=config.val[i];
    // for (uint i=0; i<4; i++) cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32+i]=config.val[i];
    // for (uint i=0; i<4; i++) cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32+i]=config.val[i];
    // for (uint i=0; i<4; i++) cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32+i]=config.val[i];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 2] = config.val[2];
    cfg[THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 3] = config.val[3];
    cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 3] = config.val[3];
    cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 3] = config.val[3];
    cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 3] = config.val[3];

    dest_rd_ctrl_u dest_rd_ctrl;
    dest_rd_ctrl.val = 0;

    bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                         pack_src_format == to_underlying(DataFormat::Float32);
    bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);

    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;

    if (pack_dst_format == to_underlying(DataFormat::UInt8))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_unsigned = 1;
    }

    // Round to 10 bit mantissa from fp32 dest
    if (is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16)))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Round_10b_mant = 1;
    }
    cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32] = dest_rd_ctrl.val;

    if (IS_BFP_FORMAT(pack_dst_format))
    {
        // Override exp section size for packers 1,2,3 (see bfp_exp_section_size)
        if ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp8) || (pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp8_b))
        {
            config.f.exp_section_size                              = bfp_exp_section_size(1 /* index */, bfp8_row_bytes, num_faces);
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = bfp_exp_section_size(2 /* index */, bfp8_row_bytes, num_faces);
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = bfp_exp_section_size(3 /* index */, bfp8_row_bytes, num_faces);
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
        }
        else if ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp4) || (pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp4_b))
        {
            config.f.exp_section_size                              = bfp_exp_section_size(1 /* index */, bfp4_row_bytes, num_faces);
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = bfp_exp_section_size(2 /* index */, bfp4_row_bytes, num_faces);
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = bfp_exp_section_size(3 /* index */, bfp4_row_bytes, num_faces);
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
        }
        else if ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp2) || (pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp2_b))
        {
            config.f.exp_section_size                              = bfp_exp_section_size(1 /* index */, bfp2_row_bytes, num_faces);
            cfg[THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = bfp_exp_section_size(2 /* index */, bfp2_row_bytes, num_faces);
            cfg[THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0] = config.val[0];
            config.f.exp_section_size                              = bfp_exp_section_size(3 /* index */, bfp2_row_bytes, num_faces);
            cfg[THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0] = config.val[0];
        }
    }

    cache_exponential_section_sizes_in_gprs<false>(num_faces, partial_face);
}

inline void set_packer_l1_offset(const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM)
{
    const std::uint32_t face_dim = face_r_dim * FACE_C_DIM;

    std::uint32_t l1_offset_1 = IS_BFP_FORMAT(pack_dst_format)
                                    ? 1
                                    : ((static_cast<std::uint8_t>(pack_dst_format & 0x3) == to_underlying(DataFormat::Float32))   ? (face_dim / 16) * 4
                                       : (static_cast<std::uint8_t>(pack_dst_format & 0x3) == to_underlying(DataFormat::Float16)) ? (face_dim / 16) * 2
                                                                                                                                  : (face_dim / 16));
    std::uint32_t l1_offset_2 = 2 * l1_offset_1;
    std::uint32_t l1_offset_3 = 3 * l1_offset_1;

    // HW automatically offsets packers base address by tile header size
    // with new L1 addressing mode, the effective address for pack1/2/3
    // will be pack[i] += pack[0], which leads to double counting of tile header
    // subtract by this amount when programming the offset
    constexpr std::uint32_t PACK_TILE_HEADER_OFFSET = 1; // in 16B
    l1_offset_1 -= PACK_TILE_HEADER_OFFSET;
    l1_offset_2 -= PACK_TILE_HEADER_OFFSET;
    l1_offset_3 -= PACK_TILE_HEADER_OFFSET;
    TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_1), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
    TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_2), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
    TT_SETDMAREG(0, LOWER_HALFWORD(l1_offset_3), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
}

/**
 * @brief Configure packer exponent thresholding for the current destination and output formats.
 *
 * Packer exponent thresholding can be used toforce destination values that are not representable in
 * the packed format to zero when the exponent range narrows. This function enables it only when the
 *
 * @see https://github.com/tenstorrent/tt-isa-documentation/tree/main/WormholeB0/TensixTile/TensixCoprocessor/Packers/ExponentThresholding.md
 *
 * @tparam is_fp32_dest_acc_en True when Dest register is FP32.
 * @param pack_dst_format Pack output data format.
 */
template <bool is_fp32_dest_acc_en>
inline void reconfigure_exp_threshold(const std::uint32_t pack_dst_format)
{
    bool enable             = false;
    std::uint32_t threshold = 0;

    // Workaround for HW bug: tenstorrent/budabackend#1394
    if constexpr (is_fp32_dest_acc_en)
    {
        if (IS_BFP_A_FORMAT(pack_dst_format))
        {
            enable    = true;
            threshold = 113;
        }
    }

    static_assert(
        THCON_SEC0_REG1_Exp_threshold_en_ADDR32 == THCON_SEC0_REG1_Exp_threshold_ADDR32,
        "THCON_SEC0_REG1_Exp_threshold_en and Exp_threshold must share ADDR32 for combined RMW");

    constexpr std::uint32_t THRESHOLD_RMW_MASK = THCON_SEC0_REG1_Exp_threshold_en_MASK | THCON_SEC0_REG1_Exp_threshold_MASK;

    std::uint32_t threshold_rmw_data = (threshold << THCON_SEC0_REG1_Exp_threshold_SHAMT) | (enable << THCON_SEC0_REG1_Exp_threshold_en_SHAMT);

    cfg_reg_rmw_tensix<THCON_SEC0_REG1_Exp_threshold_ADDR32, 0, THRESHOLD_RMW_MASK>(threshold_rmw_data);
    cfg_reg_rmw_tensix<THCON_SEC1_REG1_Exp_threshold_ADDR32, 0, THRESHOLD_RMW_MASK>(threshold_rmw_data);
    cfg_reg_rmw_tensix<THCON_SEC0_REG8_Exp_threshold_ADDR32, 0, THRESHOLD_RMW_MASK>(threshold_rmw_data);
    cfg_reg_rmw_tensix<THCON_SEC1_REG8_Exp_threshold_ADDR32, 0, THRESHOLD_RMW_MASK>(threshold_rmw_data);
}

// Forward declaration: defined later in this file. Needed here so that the non-dependent call
// inside `reconfig_packer_data_format`'s LLK_ASSERT is found via unqualified lookup at
// the template's first-phase name lookup (ADL cannot find it from a std::uint32_t argument).
__attribute__((noinline)) bool is_pack_reads_per_xy_plane(const std::uint32_t expected, const std::uint32_t nop_count = 10);

template <bool is_fp32_dest_acc_en>
__attribute__((noinline)) inline void reconfig_packer_data_format(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size  = 0,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false)
{
    // Packer strides for standard tiled dest layout (PackMode::Default). Untilize uses configure_pack with PackMode::Untilize.
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(static_cast<DataFormat>(pack_src_format), static_cast<DataFormat>(pack_dst_format)),
        "Unsupported packer to L1 conversion.");
    // Configure packers
    pack_config_u config;
    config.val[2] = 0; // Only need to modify word[2][15:0]

    config.f.uncompress      = 1;
    config.f.out_data_format = pack_dst_format;
    config.f.in_data_format  = pack_src_format;

    // Wait till packer is done before changing config registers
    TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::PACK);
    TT_SETDMAREG(0, LOWER_HALFWORD(config.val[2]), 0, LO_16(p_gpr_pack::TMP_LO));
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO); // 16-bit write
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 2 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 2 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);
    TTI_REG2FLOP(2, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 2 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::TMP_LO);

    LLK_ASSERT(
        is_pack_reads_per_xy_plane(1),
        "reconfig_packer_data_format: pack_reads_per_xy_plane counter must be 1 before packer reconfig (invariant violated; reduce mask should have been "
        "cleared via _llk_pack_reduce_mask_clear_). Please uncomment DEVICE_PRINT #2111 for debugging.");

    dest_rd_ctrl_u dest_rd_ctrl;
    dest_rd_ctrl.val = 0;

    bool is_32b_format = pack_src_format == to_underlying(DataFormat::Int32) || pack_src_format == to_underlying(DataFormat::UInt32) ||
                         pack_src_format == to_underlying(DataFormat::Float32);
    bool is_int8_format = pack_src_format == to_underlying(DataFormat::Int8) || pack_src_format == to_underlying(DataFormat::UInt8);

    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_32b_data = is_32b_format || is_fp32_dest_acc_en;
    dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_int8     = !(is_fp32_dest_acc_en || is_32b_format) && is_int8_format;

    if (pack_dst_format == to_underlying(DataFormat::UInt8))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Read_unsigned = 1;
    }
    // Round to 10 bit mantissa from fp32 dest
    if (is_fp32_dest_acc_en && (pack_src_format == to_underlying(DataFormat::Float16)))
    {
        dest_rd_ctrl.f.PCK_DEST_RD_CTRL_Round_10b_mant = 1;
    }
    cfg_reg_rmw_tensix<
        PCK_DEST_RD_CTRL_Read_32b_data_ADDR32,
        PCK_DEST_RD_CTRL_Read_32b_data_SHAMT,
        PCK_DEST_RD_CTRL_Read_32b_data_MASK | PCK_DEST_RD_CTRL_Read_unsigned_MASK | PCK_DEST_RD_CTRL_Round_10b_mant_MASK>(dest_rd_ctrl.val);

    if (IS_BFP_FORMAT(pack_dst_format))
    {
        cache_exponential_section_sizes_in_gprs<true>(num_faces, partial_face);

        // Wait till the MMIO is finished
        TTI_STALLWAIT(p_stall::STALL_THCON, p_stall::TRISC_CFG);

        // Override exp section size for packers 1,2,3
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP0_SEC_SIZE_BFP);
        if ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp8) || (pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp8_b))
        {
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP8);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP8);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP8);
        }
        else if ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp4) || (pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp4_b))
        {
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP4);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP4);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP4);
        }
        else if ((pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp2) || (pack_dst_format & 0x1F) == to_underlying(DataFormat::Bfp2_b))
        {
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP1_SEC_SIZE_BFP2);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP2_SEC_SIZE_BFP2);
            TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::EXP3_SEC_SIZE_BFP2);
        }
    }
    else if ((pack_dst_format == to_underlying(DataFormat::Lf8)) || (masked_data_format(pack_dst_format) == to_underlying(DataFormat::Int8)))
    {
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 0 - THCON_CFGREG_BASE_ADDR32, p_gpr::ZERO);
    }

    // Set l1 address offset
    set_packer_l1_offset(pack_dst_format, face_r_dim);

    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_pack::TILE_HEADER));

    reconfigure_exp_threshold<is_fp32_dest_acc_en>(pack_dst_format);

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(pack_src_format);

    // Set packer strides
    set_packer_strides(pack_src_format);

    // NOTE: the packer X (datum) counter (SETADCXX PAC) is intentionally NOT programmed here. Its value
    // is pack_mode-dependent (Untilize vs Default) and is owned by _llk_pack_init_, which runs after this
    // reconfig for the specific op/mode.
}

template <bool is_fp32_dest_acc_en, PackMode pack_mode>
inline void configure_pack(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size           = 0,
    const std::uint32_t face_r_dim          = FACE_R_DIM,
    const std::uint32_t num_faces           = 4,
    const bool partial_face                 = false,
    [[maybe_unused]] const bool narrow_tile = false,
    const std::uint32_t relu_config         = 0)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize,
        "Wormhole B0 pack hardware configuration supports only PackMode::Default and PackMode::Untilize");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(
        is_packer_to_L1_conversion_supported(static_cast<DataFormat>(pack_src_format), static_cast<DataFormat>(pack_dst_format)),
        "Unsupported packer to L1 conversion.");
    // Get pointer to registers for current state ID
    volatile std::uint32_t* cfg = get_cfg_pointer();

    if (pack_src_format != pack_dst_format)
    {
        TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::PACK);
        tensix_sync();
    }

    set_packer_strides(pack_src_format);

    t6_mutex_acquire(mutex::REG_RMW);

    const std::uint32_t alu_dst_format = pack_src_format;

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG2_Dstacc_RMW>(alu_dst_format);

    // Config RELU
    relu_config_u hw_relu_config;
    hw_relu_config.r.STACC_RELU_ApplyRelu     = relu_config & 0xffff;
    hw_relu_config.r.STACC_RELU_ReluThreshold = (relu_config >> 16) & 0xffff;

    constexpr std::uint32_t hw_relu_mask = STACC_RELU_ApplyRelu_MASK | STACC_RELU_ReluThreshold_MASK;
    cfg_reg_rmw_tensix<STACC_RELU_ApplyRelu_ADDR32, 0, hw_relu_mask>(hw_relu_config.val[0]);

    t6_mutex_release(mutex::REG_RMW);

    set_packer_config<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, num_faces, partial_face);

    set_packer_l1_offset(pack_dst_format, face_r_dim);

    // NOTE: the packer X (datum) counter (SETADCXX PAC) is intentionally NOT programmed here. Its value
    // is pack_mode-dependent (Untilize packs a single face row, Default a full face) and hw-configure
    // always runs in PackMode::Default, so it cannot establish the Untilize value. _llk_pack_init_ owns
    // this counter and runs after hw-configure for the specific op/mode.

    // PACK_COUNTERS_SEC0_pack_per_xy_plane = cfg_reg_array[3][0 +: 8];
    // PACK_COUNTERS_SEC0_pack_reads_per_xy_plane = cfg_reg_array[3][8 +: 8];
    // PACK_COUNTERS_SEC0_pack_xys_per_tile = cfg_reg_array[3][16 +: 7];
    // PACK_COUNTERS_SEC0_pack_yz_transposed = cfg_reg_array[3][23 +: 1];
    pack_counters_u pack_counters;
    pack_counters.val = 0;
    pack_counters.f.pack_reads_per_xy_plane =
        1; // Default 1 — makes non-reduce operations agnostic to this counter; reduce sets it via _llk_pack_reduce_mask_config_
    for (std::uint32_t i = 0; i < NUM_PACKERS; i++)
    {
        cfg[PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32 + i] = pack_counters.val; // disable auto last generation
    }

    pck_edge_offset_u pck_edge_offset;
    pck_edge_offset.val    = 0;
    pck_edge_offset.f.mask = 0xffff;

    cfg[PCK_EDGE_OFFSET_SEC0_mask_ADDR32]                = pck_edge_offset.val;
    cfg[TILE_ROW_SET_MAPPING_0_row_set_mapping_0_ADDR32] = 0x0; // All packers use row set mapping 0, edge offset 0 mask

    regfile[p_gpr_pack::TILE_HEADER]     = tile_size;
    regfile[p_gpr_pack::TILE_HEADER + 1] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 2] = 0;
    regfile[p_gpr_pack::TILE_HEADER + 3] = 0;
    sync_regfile_write(p_gpr_pack::TILE_HEADER + 3);
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

// Flip packer dest register offset to 0 or DEST_REGISTER_HALF_SIZE
// flip-flopping between two halves
template <DstSync Dst>
inline void select_packer_dest_registers()
{
    if constexpr (Dst == DstSync::SyncFull)
    {
        TTI_WRCFG(p_gpr_pack::DEST_OFFSET_LO, p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    else
    {
        TT_WRCFG(get_packer_dest_offset_index(), p_cfg::WRCFG_128b, DEST_TARGET_REG_CFG_PACK_SEC0_Offset_ADDR32);
    }
    TTI_DMANOP;
    TTI_DMANOP;
}

// Program packer destination addresses from GPRs
inline void program_packer_destination(std::uint32_t addr, bool restore = true)
{
    LLK_ASSERT(is_valid_L1_address(addr), "L1 address must be in valid L1 memory region");
    std::uint32_t new_l1_addr = (1 << 31) | addr;
    TT_SETDMAREG(0, LOWER_HALFWORD(addr), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));

    // No STALLWAIT is needed before this config write (unlike the BH counterpart):
    // REG2FLOP executes on ThCon, in program order after the SETDMAREG GPR writes above, so it reads the
    // freshly written OUTPUT_ADDR without a fence (BH uses WRCFG on the separate Config Unit, hence its
    // STALL_CFG/THCON guard). No packer-drain (STALL_THCON/PACK) is needed either: the pack thread issues
    // its instruction stream in order, so each tile's PACR has already started -- and latched L1_Dest_addr,
    // which is sampled at PACR start -- before the next call's REG2FLOP reprograms it. The Flush PACR below
    // drains the previous pack's output buffers and arms a fresh start address.
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);

    TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0); // pack flush

    if (restore)
    {
        TT_SETDMAREG(0, UPPER_HALFWORD(addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    }
}

template <std::uint32_t block_ct_dim, std::uint32_t full_ct_dim, bool diagonal = false, std::uint32_t row_num_datums = TILE_C_DIM>
inline void program_packer_untilized_destination(const std::uint32_t addr, const std::uint32_t pack_dst_format)
{
    LLK_ASSERT(is_valid_L1_address(addr), "L1 address must be in valid L1 memory region");

    if constexpr (diagonal)
    {
        // Diagonal untilize drives only packers 0 and 1; the offset2/offset3 + packer-2/3 (SEC1) lines below are
        // intentionally disabled and kept as reference for a possible 4-packer extension.
        const std::uint32_t block_size  = SCALE_DATUM_SIZE(pack_dst_format, FACE_C_DIM);
        constexpr std::uint32_t offset0 = 0;
        const std::uint32_t offset1     = (1 * block_size) / 16;
        // const uint32_t offset2 = (2*block_size)/16;
        // const uint32_t offset3 = (3*block_size)/16;

        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset0), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 0));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset0), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 0));
        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset1), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 1));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset1), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 1));
        // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset2), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+2));
        // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset2), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+2));
        // TT_SETDMAREG(0, LOWER_HALFWORD(addr+offset3), 0, LO_16(p_gpr_pack::OUTPUT_ADDR+3));
        // TT_SETDMAREG(0, UPPER_HALFWORD(addr+offset3), 0, HI_16(p_gpr_pack::OUTPUT_ADDR+3));

        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 1);
        // TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG1_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR+2);
        // TTI_REG2FLOP(1,0,0,0,THCON_SEC1_REG8_L1_Dest_addr_ADDR32-THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR+3);

        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0); // pack flush
    }
    else
    {
        // Each packer packs 8 rows of full_ct_dim*TILE_C_DIM datums
        const std::uint32_t block_size  = SCALE_DATUM_SIZE(pack_dst_format, full_ct_dim * TILE_C_DIM * (TILE_R_DIM / 4));
        constexpr std::uint32_t offset0 = 0;
        const std::uint32_t offset1     = (1 * row_num_datums * block_size) / 16 / TILE_C_DIM;
        const std::uint32_t offset2     = (2 * row_num_datums * block_size) / 16 / TILE_C_DIM;
        const std::uint32_t offset3     = (3 * row_num_datums * block_size) / 16 / TILE_C_DIM;

        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset0), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 0));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset0), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 0));
        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset1), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 1));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset1), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 1));
        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset2), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 2));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset2), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 2));
        TT_SETDMAREG(0, LOWER_HALFWORD(addr + offset3), 0, LO_16(p_gpr_pack::OUTPUT_ADDR + 3));
        TT_SETDMAREG(0, UPPER_HALFWORD(addr + offset3), 0, HI_16(p_gpr_pack::OUTPUT_ADDR + 3));

        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 1);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 2);
        TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC1_REG8_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR + 3);

        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0); // pack flush
    }
}

inline void program_packer_dest_offset_registers(std::uint32_t dest_tile_offset)
{
    TT_SETDMAREG(0, LOWER_HALFWORD(dest_tile_offset), 0, LO_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TT_SETDMAREG(0, UPPER_HALFWORD(dest_tile_offset), 0, HI_16(p_gpr_pack::TEMP_TILE_OFFSET));
    TTI_WRCFG(p_gpr_pack::TEMP_TILE_OFFSET, p_cfg::WRCFG_32b, PCK0_ADDR_BASE_REG_0_Base_ADDR32);
    TTI_DMANOP;
    TTI_DMANOP;
}

inline void reconfigure_packer_l1_acc(const std::uint32_t pack_l1_acc)
{
    // Stall to avoid clobbering current packer configuration
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);

    // While packing, if all datums of a face are 0s, the packer will automatically set the zflags. For L1 accumulation mode, even if we pack out an entire face
    // of 0s, because the data we are accumulating with is unknown, we don't want to set the zflags.
    const std::uint32_t pack_l1_acc_disable_pack_zero_flag = pack_l1_acc ? (0b11) : (0b00);

    cfg_reg_rmw_tensix<
        THCON_SEC0_REG1_Pack_L1_Acc_ADDR32,
        THCON_SEC0_REG1_Pack_L1_Acc_SHAMT,
        THCON_SEC0_REG1_Disable_pack_zero_flags_MASK | THCON_SEC0_REG1_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
    cfg_reg_rmw_tensix<
        THCON_SEC0_REG8_Pack_L1_Acc_ADDR32,
        THCON_SEC0_REG8_Pack_L1_Acc_SHAMT,
        THCON_SEC0_REG8_Disable_pack_zero_flags_MASK | THCON_SEC0_REG8_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
    cfg_reg_rmw_tensix<
        THCON_SEC1_REG1_Pack_L1_Acc_ADDR32,
        THCON_SEC1_REG1_Pack_L1_Acc_SHAMT,
        THCON_SEC1_REG1_Disable_pack_zero_flags_MASK | THCON_SEC1_REG1_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
    cfg_reg_rmw_tensix<
        THCON_SEC1_REG8_Pack_L1_Acc_ADDR32,
        THCON_SEC1_REG8_Pack_L1_Acc_SHAMT,
        THCON_SEC1_REG8_Disable_pack_zero_flags_MASK | THCON_SEC1_REG8_Pack_L1_Acc_MASK>(pack_l1_acc_disable_pack_zero_flag);
}

// READERS FOR CONFIG STRUCTS

inline pack_config_t read_pack_config_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_config_u config = {.val = 0};

    config.val[0] = cfg[reg_addr];
    config.val[1] = cfg[reg_addr + 1];
    config.val[2] = cfg[reg_addr + 2];
    config.val[3] = cfg[reg_addr + 3];

    return config.f;
}

inline std::array<pack_config_t, NUM_PACKERS> read_pack_config()
{
    std::array<pack_config_t, NUM_PACKERS> config_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    config_vec[0] = read_pack_config_helper(THCON_SEC0_REG1_Row_start_section_size_ADDR32, cfg);
    config_vec[1] = read_pack_config_helper(THCON_SEC0_REG8_Row_start_section_size_ADDR32, cfg);
    config_vec[2] = read_pack_config_helper(THCON_SEC1_REG1_Row_start_section_size_ADDR32, cfg);
    config_vec[3] = read_pack_config_helper(THCON_SEC1_REG8_Row_start_section_size_ADDR32, cfg);

    return config_vec;
}

inline relu_config_t read_relu_config()
{
    relu_config_u config;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();
    config.val[0]                          = cfg[ALU_ACC_CTRL_Zero_Flag_disabled_src_ADDR32];

    return config.r;
}

inline dest_rd_ctrl_t read_dest_rd_ctrl()
{
    dest_rd_ctrl_u dest;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    dest.val = cfg[PCK_DEST_RD_CTRL_Read_32b_data_ADDR32];

    return dest.f;
}

inline pck_edge_offset_t read_pack_edge_offset_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pck_edge_offset_u edge = {.val = 0};
    edge.val               = cfg[reg_addr];

    return edge.f;
}

inline std::array<pck_edge_offset_t, NUM_PACKERS> read_pack_edge_offset()
{
    std::array<pck_edge_offset_t, NUM_PACKERS> edge_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    edge_vec[0] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC0_mask_ADDR32, cfg);
    edge_vec[1] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC1_mask_ADDR32, cfg);
    edge_vec[2] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC2_mask_ADDR32, cfg);
    edge_vec[3] = read_pack_edge_offset_helper(PCK_EDGE_OFFSET_SEC3_mask_ADDR32, cfg);

    return edge_vec;
}

inline pack_counters_t read_pack_counters_helper(std::uint32_t reg_addr, const volatile std::uint32_t tt_reg_ptr* cfg)
{
    pack_counters_u counters = {.val = 0};
    counters.val             = cfg[reg_addr];

    return counters.f;
}

inline std::array<pack_counters_t, NUM_PACKERS> read_pack_counters()
{
    std::array<pack_counters_t, NUM_PACKERS> counters_vec;

    // Get pointer to registers for current state ID
    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    counters_vec[0] = read_pack_counters_helper(PACK_COUNTERS_SEC0_pack_per_xy_plane_ADDR32, cfg);
    counters_vec[1] = read_pack_counters_helper(PACK_COUNTERS_SEC1_pack_per_xy_plane_ADDR32, cfg);
    counters_vec[2] = read_pack_counters_helper(PACK_COUNTERS_SEC2_pack_per_xy_plane_ADDR32, cfg);
    counters_vec[3] = read_pack_counters_helper(PACK_COUNTERS_SEC3_pack_per_xy_plane_ADDR32, cfg);

    return counters_vec;
}

/**
 * Validates that all packers' config matches the expected formats.
 * On mismatch, issues DEVICE_PRINT (when enabled) and LLK_ASSERT. Typically invoked via
 * `LLK_ASSERT_BLOCK(are_packers_configured_correctly(...))` in llk_pack_tile_api.h.
 *
 * The pack_reads_per_xy_plane counter is validated separately via is_pack_reads_per_xy_plane
 * (invoked at reconfig time), since its expected value depends on whether a reduce mask is
 * currently programmed, not on the per-pack call site.
 *
 * @param pack_src_format   Expected input data format for all packers
 * @param pack_dst_format   Expected output data format for all packers
 * @param nop_count         Number of nop operations to ensure configuration writes complete (default 10)
 */
__attribute__((noinline)) void are_packers_configured_correctly(
    const std::uint32_t pack_src_format, const std::uint32_t pack_dst_format, const std::uint32_t nop_count = 10)
{
    // Ensure configuration writes complete before subsequent operations
    tensix_sync();
    for (std::uint32_t i = 0; i < nop_count; i++)
    {
        asm volatile("nop");
    }

    volatile std::uint32_t tt_reg_ptr* cfg = get_cfg_pointer();

    // Only read config word 2 per packer (contains in_data_format and out_data_format)
    static constexpr std::uint32_t config_word2_addrs[NUM_PACKERS] = {
        THCON_SEC0_REG1_Row_start_section_size_ADDR32 + 2,
        THCON_SEC0_REG8_Row_start_section_size_ADDR32 + 2,
        THCON_SEC1_REG1_Row_start_section_size_ADDR32 + 2,
        THCON_SEC1_REG8_Row_start_section_size_ADDR32 + 2,
    };

    const std::uint32_t expected_src = masked_data_format(pack_src_format);
    const std::uint32_t expected_dst = masked_data_format(pack_dst_format);

    for (std::uint32_t i = 0; i < NUM_PACKERS; i++)
    {
        pack_config_u config = {.val = {0}};
        config.val[2]        = cfg[config_word2_addrs[i]];

        if (config.f.in_data_format != expected_src)
        {
            // DEVICE_PRINT(
            // "#2101 are_packers_configured_correctly: packer {} pack_src_format mismatch. expected: {}, actual: {}\n", i, expected_src,
            // config.f.in_data_format);
            LLK_ASSERT(
                (config.f.in_data_format == expected_src),
                "are_packers_configured_correctly: pack_src_format mismatch. Uncomment DEVICE_PRINT #2101 to inspect "
                "packer index and expected/actual.");
        }
        if (config.f.out_data_format != expected_dst)
        {
            // DEVICE_PRINT(
            // "#2102 are_packers_configured_correctly: packer {} pack_dst_format mismatch. expected: {}, actual: {}\n", i, expected_dst,
            // config.f.out_data_format);
            LLK_ASSERT(
                (config.f.out_data_format == expected_dst),
                "are_packers_configured_correctly: pack_dst_format mismatch. Uncomment DEVICE_PRINT #2102 to inspect "
                "packer index and expected/actual.");
        }
    }
}

/**
 * Validates that all packers' pack_reads_per_xy_plane counter matches the expected value.
 * Intended to be called from reconfig paths to assert the invariant: between non-reduce
 * operations the counter is 1; the reduce layer owns transitions to FACE_R_DIM via
 * `_llk_pack_reduce_mask_config_` and back to 1 via `_llk_pack_reduce_mask_clear_`.
 *
 * @param expected   Expected value of pack_reads_per_xy_plane
 * @param nop_count  Number of nop operations to ensure configuration writes complete (default 10)
 * @return true if all packers' counters match expected, false otherwise.
 */
__attribute__((noinline)) bool is_pack_reads_per_xy_plane(const std::uint32_t expected, const std::uint32_t nop_count)
{
    // Ensure configuration writes complete before reading back
    tensix_sync();
    for (std::uint32_t i = 0; i < nop_count; i++)
    {
        asm volatile("nop");
    }

    const std::array<pack_counters_t, NUM_PACKERS> counters_vec = read_pack_counters();
    for (std::uint32_t i = 0; i < NUM_PACKERS; i++)
    {
        if (counters_vec[i].pack_reads_per_xy_plane != expected)
        {
            // Debug: print which packer mismatched and dump all per-packer counter values.
            // DEVICE_PRINT(
            //     "#2111 is_pack_reads_per_xy_plane: mismatch on packer {} (expected={}) actual [P0={} P1={} P2={} P3={}]\n",
            //     i,
            //     expected,
            //     counters_vec[0].pack_reads_per_xy_plane,
            //     counters_vec[1].pack_reads_per_xy_plane,
            //     counters_vec[2].pack_reads_per_xy_plane,
            //     counters_vec[3].pack_reads_per_xy_plane);
            return false;
        }
    }
    return true;
}

} // namespace ckernel::packer
