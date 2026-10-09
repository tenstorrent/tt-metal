// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>

#include "ckernel_trisc_common.h"

namespace ckernel::math
{

/**
 * @brief Return the phase count before removing format-redundant phases.
 * @tparam fidelity: Requested approximation, values = <LoFi/HiFi2/HiFi3/HiFi4>.
 */
template <MathFidelity fidelity>
constexpr std::uint32_t math_fidelity_phases()
{
    static_assert(
        fidelity == MathFidelity::LoFi || fidelity == MathFidelity::HiFi2 || fidelity == MathFidelity::HiFi3 || fidelity == MathFidelity::HiFi4,
        "Invalid math fidelity");
    return fidelity == MathFidelity::LoFi ? 1 : to_underlying(fidelity);
}

/**
 * @brief Check whether the requested final phase contributes for these source formats.
 * @param fidelity: Requested approximation, values = <LoFi/HiFi2/HiFi3/HiFi4>.
 * @param src_a_format: Effective SrcA register format.
 * @param src_b_format: Effective SrcB register format.
 * @note Pass register formats, not L1 MX formats; unpacked MX normally uses Float16_b.
 */
constexpr bool is_math_fidelity_supported(const MathFidelity fidelity, const DataFormat src_a_format, const DataFormat src_b_format)
{
    const auto has_low_mantissa = [](const DataFormat format)
    { return format == DataFormat::Float16 || format == DataFormat::Tf32 || format == DataFormat::Float32; };
    const auto is_float = [&](const DataFormat format)
    { return has_low_mantissa(format) || format == DataFormat::Float16_b || format == DataFormat::MxFp4_2x_A || format == DataFormat::MxFp4_2x_B; };
    const auto is_integer = [](const DataFormat format)
    { return format == DataFormat::Int8 || format == DataFormat::UInt8 || format == DataFormat::Int8_2x || format == DataFormat::UInt8_2x; };

    if (is_integer(src_a_format) && is_integer(src_b_format))
    {
        return fidelity == MathFidelity::LoFi;
    }
    if (!is_float(src_a_format) || !is_float(src_b_format))
    {
        return false;
    }

    switch (fidelity)
    {
        case MathFidelity::LoFi:
            return true;
        case MathFidelity::HiFi2:
            return has_low_mantissa(src_a_format);
        case MathFidelity::HiFi3:
            return has_low_mantissa(src_b_format);
        case MathFidelity::HiFi4:
            return has_low_mantissa(src_a_format) && has_low_mantissa(src_b_format);
        default:
            return false;
    }
}

/**
 * @brief Assert that the format/fidelity combination is supported and nonredundant.
 * @tparam fidelity: Requested approximation, values = <LoFi/HiFi2/HiFi3/HiFi4>.
 * @param src_a_format: Effective SrcA register format.
 * @param src_b_format: Effective SrcB register format.
 * @note Reinitialize the operation whenever either effective source format changes.
 */
template <MathFidelity fidelity>
inline void validate_math_fidelity(const DataFormat src_a_format, const DataFormat src_b_format)
{
    static_assert(math_fidelity_phases<fidelity>() >= 1);
    LLK_ASSERT(is_math_fidelity_supported(fidelity, src_a_format, src_b_format), "Unsupported or redundant math fidelity for source register formats");
}

struct MathFidelitySchedule
{
    std::uint8_t phase_count;
    std::uint8_t phase_increment;

    constexpr bool operator==(const MathFidelitySchedule& other) const
    {
        return phase_count == other.phase_count && phase_increment == other.phase_increment;
    }
};

/**
 * @brief Resolve the issued phase count and counter increment for the source formats.
 * @tparam fidelity: Requested approximation, values = <LoFi/HiFi2/HiFi3/HiFi4>.
 * @param src_a_format: Effective SrcA register format.
 * @param src_b_format: Effective SrcB register format.
 * @note Validate the format/fidelity combination with @ref validate_math_fidelity before use.
 */
template <MathFidelity fidelity>
constexpr MathFidelitySchedule math_fidelity_schedule(const DataFormat src_a_format, const DataFormat src_b_format)
{
    if constexpr (fidelity == MathFidelity::HiFi3)
    {
        // A narrow SrcA contributes only phases 0 and 2.
        if (!is_math_fidelity_supported(MathFidelity::HiFi2, src_a_format, src_b_format))
        {
            return {2, 2};
        }
    }
    return {math_fidelity_phases<fidelity>(), fidelity == MathFidelity::LoFi ? std::uint8_t {0} : std::uint8_t {1}};
}

/**
 * @brief Return the source register format of an operand that is copied from dest instead of unpacked from L1.
 * @tparam EN_32BIT_DEST: dest is in 32-bit mode.
 * @param dest_format: dest format in 16-bit mode.
 */
template <bool EN_32BIT_DEST>
constexpr DataFormat dest_src_format(const DataFormat dest_format)
{
    return EN_32BIT_DEST ? DataFormat::Tf32 : dest_format;
}

// Each mask lists allowed requests in LoFi, HiFi2, HiFi3, HiFi4 order.
constexpr unsigned fidelity_mask(const DataFormat src_a, const DataFormat src_b)
{
    return is_math_fidelity_supported(MathFidelity::LoFi, src_a, src_b) | (is_math_fidelity_supported(MathFidelity::HiFi2, src_a, src_b) << 1) |
           (is_math_fidelity_supported(MathFidelity::HiFi3, src_a, src_b) << 2) | (is_math_fidelity_supported(MathFidelity::HiFi4, src_a, src_b) << 3);
}

static_assert(math_fidelity_phases<MathFidelity::LoFi>() == 1);
static_assert(math_fidelity_phases<MathFidelity::HiFi2>() == 2);
static_assert(math_fidelity_phases<MathFidelity::HiFi3>() == 3);
static_assert(math_fidelity_phases<MathFidelity::HiFi4>() == 4);
static_assert(fidelity_mask(DataFormat::Float16_b, DataFormat::Float16_b) == 0b0001);
static_assert(fidelity_mask(DataFormat::Tf32, DataFormat::Float16_b) == 0b0011);
static_assert(fidelity_mask(DataFormat::Float16_b, DataFormat::Tf32) == 0b0101);
static_assert(fidelity_mask(DataFormat::Tf32, DataFormat::Tf32) == 0b1111);
static_assert(fidelity_mask(DataFormat::MxFp4_2x_A, DataFormat::MxFp4_2x_B) == 0b0001);
static_assert(fidelity_mask(DataFormat::Int8, DataFormat::Int8) == 0b0001);
static_assert(fidelity_mask(DataFormat::Int8, DataFormat::Float16) == 0);
static_assert(fidelity_mask(DataFormat::MxFp4, DataFormat::MxFp4) == 0);

static_assert(math_fidelity_schedule<MathFidelity::LoFi>(DataFormat::Float16_b, DataFormat::Tf32) == MathFidelitySchedule {1, 0});
static_assert(math_fidelity_schedule<MathFidelity::HiFi2>(DataFormat::Tf32, DataFormat::Float16_b) == MathFidelitySchedule {2, 1});
static_assert(math_fidelity_schedule<MathFidelity::HiFi3>(DataFormat::Tf32, DataFormat::Tf32) == MathFidelitySchedule {3, 1});
static_assert(math_fidelity_schedule<MathFidelity::HiFi3>(DataFormat::Float16_b, DataFormat::Tf32) == MathFidelitySchedule {2, 2});
static_assert(math_fidelity_schedule<MathFidelity::HiFi4>(DataFormat::Tf32, DataFormat::Tf32) == MathFidelitySchedule {4, 1});

// Rows one FPU instruction covers: 8 on the base Quasar part, 4 on the narrow one.
constexpr static std::uint32_t ELTWISE_MATH_ROWS = MATH_ROWS;
static_assert(ELTWISE_MATH_ROWS == 4 || ELTWISE_MATH_ROWS == 8, "the math LLKs support a 4-row or 8-row FPU");

template <std::uint32_t NUM_ROWS>
constexpr auto fpu_row_offsets()
{
    static_assert(NUM_ROWS > 0 && NUM_ROWS % ELTWISE_MATH_ROWS == 0);

    std::array<std::uint32_t, NUM_ROWS / ELTWISE_MATH_ROWS> rows {};

    for (std::uint32_t i = 0; i < rows.size(); ++i)
    {
        rows[i] = i * ELTWISE_MATH_ROWS;
    }

    return rows;
}

constexpr static std::uint32_t FPU_MOV_ROWS = (ELTWISE_MATH_ROWS == 8) ? p_mov_src_to_dest::MOV_8_ROWS : p_mov_src_to_dest::MOV_4_ROWS;
static_assert(FPU_MOV_ROWS == (ELTWISE_MATH_ROWS == 8 ? p_movd2a::MOV_8_ROWS : p_movd2a::MOV_4_ROWS));
static_assert(FPU_MOV_ROWS == (ELTWISE_MATH_ROWS == 8 ? p_movd2b::MOV_8_ROWS : p_movd2b::MOV_4_ROWS));
static_assert(FPU_MOV_ROWS == (ELTWISE_MATH_ROWS == 8 ? p_movb2a::MOV_8_ROWS : p_movb2a::MOV_4_ROWS));

constexpr static bool FPU_SPLITS_DEST_ROW_GROUP = ELTWISE_MATH_ROWS < MAX_FPU_ROWS; // a dest row group takes several FPU issues

constexpr static std::uint32_t MOVE_MATH_ROWS[3] = {8, 4, 1};
constexpr static unsigned int SFP_ROWS           = 2;

// SFPU register-file base addresses: dest region vs SrcS (used by SFPU load/store)
constexpr static unsigned int SFPU_DEST_BASE_ADDR = 0x0;
constexpr static unsigned int SFPU_SRCS_BASE_ADDR = 0x400;

// Struct for the ALU addresses
constexpr std::uint32_t NUM_WORDS_ALU_FORMAT = 3;

typedef struct
{
    // word 0
    std::uint32_t ALU_FORMAT_SPEC_REG_SrcA_val        : 8;
    std::uint32_t ALU_FORMAT_SPEC_REG_SrcA_override   : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG_SrcB_val        : 8;
    std::uint32_t ALU_FORMAT_SPEC_REG_SrcB_override   : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG_Dstacc_val      : 8;
    std::uint32_t ALU_FORMAT_SPEC_REG_Dstacc_override : 1;
    std::uint32_t EMPTY0                              : 5;
    // word 1
    std::uint32_t ALU_ROUNDING_MODE_Fpu_srnd_en : 1;
    std::uint32_t UNUSED0                       : 2;
    std::uint32_t ALU_ROUNDING_MODE_Padding     : 10;
    std::uint32_t ALU_ROUNDING_MODE_GS_LF       : 1;
    std::uint32_t ALU_ROUNDING_MODE_Bfp8_HF     : 1;
    std::uint32_t ALU_FORMAT_SPEC_REG0_SrcA     : 8;
    std::uint32_t ALU_FORMAT_SPEC_REG1_SrcB     : 8;
    std::uint32_t EMPTY1                        : 1;
    // word 2
    std::uint32_t ALU_FORMAT_SPEC_REG2_Dstacc    : 8;
    std::uint32_t ALU_ACC_CTRL_Fp32_enabled      : 1;
    std::uint32_t ALU_ACC_CTRL_SFPU_Fp32_enabled : 1;
    std::uint32_t ALU_ACC_CTRL_INT8_math_enabled : 1;
    std::uint32_t UNUSED1                        : 21;
} alu_config_t;

static_assert(sizeof(alu_config_t) == NUM_WORDS_ALU_FORMAT * sizeof(std::uint32_t));

typedef union
{
    std::uint32_t val[NUM_WORDS_ALU_FORMAT];
    alu_config_t f;
} alu_config_u;

// List of possible data format config states
enum class DataFormatConfigSet : std::uint8_t
{
    UNCONFIGURED         = 0,
    DEFAULT              = 1,
    MOV_OPS_EXPLICIT_FMT = 2
};

// /**
// * @brief Helper function to calculate log2,
// * only works for 32 bit unsigned inputs
// * @param val: Input value to log2 operation
// */
// inline uint32_t trisc_log2(const uint32_t val) {
//     return 31 - __builtin_clz(val);
// }

/**
 * @brief Increments given counters
 * @tparam: SRCA_INCR: SrcA increment values = 0 - 15
 * @tparam: SRCB_INCR: SrcA increment values = 0 - 15
 * @tparam: SRCD_INCR: SrcA increment values = 0 - 15
 * @tparam: CR_INCR: SrcA increment values = 0 - 63
 */
template <std::uint32_t SRCA_INCR, std::uint32_t SRCB_INCR, std::uint32_t DEST_INCR, std::uint32_t CR_INCR>
inline void _incr_counters_()
{
    static_assert(SRCA_INCR < 32, "Value exceeds RWC_A width of 5 bits");
    static_assert(SRCB_INCR < 32, "Value exceeds RWC_B width of 5 bits");
    static_assert(DEST_INCR < 256, "Value exceeds RWC_D width of 8 bits");
    static_assert(CR_INCR < 64, "Value exceeds RWC_CR width of 6 bits");
    TTI_INCRWC(CR_INCR, SRCA_INCR, SRCB_INCR, DEST_INCR);
}

// TODO (RT): Is there now an alternative to this?
inline void _sfpu_load_config32_(const std::uint32_t dest, const std::uint32_t upper16, const std::uint32_t lower16)
{
    // registers 11 through 14 are programmable "constants" which are shared across all 4 rows
    // They are updated only through the CONFIG path, which uses LREG[0] first and then copies it to the desired register location
    TTI_SFPLOADI(p_sfpu::LREG0, 10, lower16); // insmod == A will write the lower bits, and not affect the upper bits;
    TTI_SFPLOADI(p_sfpu::LREG0, 8, upper16);  // insmod == 8 will write the upper bits, and not affect the lower bits;
    TTI_SFPCONFIG(0, dest, 0);
}

/**
 * @brief Initializes the programmable registers for the SFPU
 */
inline void _init_sfpu_config_reg_()
{
    TTI_SFPCONFIG(0, 0xF, 1);
    // Quasar simulator doesn't apply the SFPU const-lreg reset default at boot.
    // Reload programmable constant LREG11 = -1.0 (its RTL reset default) each launch: config_dest=0xB,
    // instr_mod1[0]=1 loads the default. sfpi materializes -1.0 and subtract-based float compares via LREG11.
    TTI_SFPCONFIG(0, 0xB, 1);
}

/**
 * @brief Reset given counters to 0
 * @tparam: SETRWC: which counter to reset, values = p_setrwc::[SET_A, SET_B, SET_D, SET_F]
 */
template <std::uint32_t SETRWC>
inline void _reset_counters_()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, SETRWC);
}

/**
 * @brief Inc dest counter using carriage return (why use the CR?)
 * @tparam NUM_ROWS: number of 16 datum rows to increment dest by, value must be <=255
 */
template <std::uint32_t NUM_ROWS>
inline void _inc_dst_addr_()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, NUM_ROWS, p_setrwc::SET_D);
}

/**
 * @brief Sets destination register base address depending on tile idx
 * @param tile_idx: Tile index in the dest reg
 * 16bit dest reg data format -> tile_idx = 0 - 7
 * 32bit dest reg data format -> tile_idx = 0 - 3
 */
template <ckernel::trisc::DstTileShape TILE_SHAPE>
inline void _set_dst_write_addr_(const std::uint32_t tile_index)
{
    constexpr std::uint32_t tile_shape_idx = ckernel::trisc::get_dest_tile_size_log2(TILE_SHAPE);
    const std::uint32_t dst_index          = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
}

/**
 * @brief Computes the tile-shape index (a log2-style shift exponent derived from
 *        the number of rows per tile) and stores it in GPR TEMP0 for later reuse
 *        by @ref _set_dst_write_addr_by_gpr_ and the reduce MOP instruction stream.
 *
 *        This is the "compute once" half of the pair that splits
 *        @ref _set_dst_write_addr_by_rows_ so the shift amount is calculated a
 *        single time (when the tile shape is known) and reused across many
 *        per-tile dest-base calculations.
 *
 * @param num_rows_per_tile Number of data rows per tile.
 */
inline void _set_tile_shape_idx_gpr_(const std::uint32_t num_rows_per_tile)
{
    const std::uint32_t tile_shape_idx =
        (num_rows_per_tile == 64)
            ? 6
            : ((num_rows_per_tile == 32) ? 5 : ((num_rows_per_tile == 16) ? 4 : ((num_rows_per_tile == 8) ? 3 : ((num_rows_per_tile == 4) ? 2 : 1))));
    ckernel::regfile[p_gpr_math::TILE_SHAPE_IDX] = tile_shape_idx;
}

/**
 * @brief Sets the destination register base address depending on the tile index,
 *        using the tile-shape index previously stored in GPR TEMP0 by
 *        @ref _set_tile_shape_idx_gpr_ as the left-shift amount that converts
 *        tile_index into a dest offset.
 *
 *        This is the "use many" half of the pair that splits
 *        @ref _set_dst_write_addr_by_rows_; call @ref _set_tile_shape_idx_gpr_
 *        once before invoking this for each tile in the reduce.
 *
 * @param tile_index Tile index in the dest reg.
 *        16-bit dest reg data format -> tile_index = 0 - 7
 *        32-bit dest reg data format -> tile_index = 0 - 3
 */
inline void _set_dst_write_addr_by_rows_(const std::uint32_t tile_index)
{
    const std::uint32_t tile_shape_idx = ckernel::regfile[p_gpr_math::TILE_SHAPE_IDX];
    const std::uint32_t dst_index      = (tile_index << tile_shape_idx) + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<TRISC_ID>(dst_index);
}

inline void move_d2a_fixed_face(const std::uint8_t addrmod)
{
    // MOVD2A src is relative to dest_section_base + dest_counter.
    // The FPU moves ELTWISE_MATH_ROWS rows per MOV (8 on Quasar, 4 on 4row_arch), so we emit
    // 16 / ELTWISE_MATH_ROWS MOVs to cover all 16 rows of a face — 4row_arch needs twice as many
    // as Quasar. The dest counter handles face progression.
    // NOTE: For different tile dimensions we need different amounts of MOV* instructions; see separate issue.
    static_assert(ELTWISE_MATH_ROWS == 8 || ELTWISE_MATH_ROWS == 4, "move_d2a_fixed_face supports MATH_ROWS of 8 (Quasar) or 4 (4row_arch)");
    // MATH drains the preceding math instructions so their source-bank release has landed before
    // SRCA_VLD tests the bank that MOVD2A will write.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::MATH, p_stall::SRCA_VLD);
#pragma GCC unroll 4
    for (const auto row : fpu_row_offsets<ckernel::FACE_R_DIM>())
    {
        TTI_MOVD2A(0, row, addrmod, FPU_MOV_ROWS, row);
    }
}

inline void move_d2b_fixed_face(const std::uint8_t addrmod)
{
    // MOVD2B src is relative to dest_section_base + dest_counter.
    // The FPU moves ELTWISE_MATH_ROWS rows per MOV (8 on Quasar, 4 on 4row_arch), so we emit
    // 16 / ELTWISE_MATH_ROWS MOVs to cover all 16 rows of a face — 4row_arch needs twice as many
    // as Quasar. The dest counter handles face progression.
    // NOTE: For different tile dimensions we need different amounts of MOV* instructions; see separate issue.
    static_assert(ELTWISE_MATH_ROWS == 8 || ELTWISE_MATH_ROWS == 4, "move_d2b_fixed_face supports MATH_ROWS of 8 (Quasar) or 4 (4row_arch)");
    // MATH drains the preceding math instructions so their source-bank release has landed before
    // SRCB_VLD tests the bank that MOVD2B will write.
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::NOTHING, p_stall::MATH, p_stall::SRCB_VLD);
#pragma GCC unroll 4
    for (const auto row : fpu_row_offsets<ckernel::FACE_R_DIM>())
    {
        TTI_MOVD2B(0, row, addrmod, FPU_MOV_ROWS, 0, row);
    }
}

template <EltwiseBinaryReuseDestType binary_reuse_dest>
inline void eltwise_binary_reuse_dest_as_src()
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        move_d2a_fixed_face(ADDR_MOD_3);
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        move_d2b_fixed_face(ADDR_MOD_3);
    }
}

} // namespace ckernel::math
