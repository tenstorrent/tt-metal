// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "llk_assert.h"
#include "llk_memory_checks.h"
#include "sanitizer/api.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// If `x` is the result of loading from memory, placing `consume_discard(x)` somewhere
// will ensure that code after `consume_discard(x)` doesn't start until the load is complete.
#define consume_discard(x) __asm volatile("andi x0, %0, 0" : : "r"((x)) : "memory")

// Reconfig behaviour for dim and stride
enum class p_dim_stride_target
{
    IGNORE,        // Do not modify dim/stride
    FACE_ROW_MAJOR // Set dim/stride for unpacking face in row major format
};

// This function stores a value to memory, and then immediately reads it back.
// The load result will not be available until the store has completed.
// This will make sure any subsequent instruction will see the store as complete.
/**
 * @brief Store a value to memory then read it back, fencing subsequent code on the store.
 *
 * The dependent load cannot retire until the store completes, so any instruction consuming the
 * returned value observes the store as finished. Used to serialize against pending memory writes.
 *
 * @param addr: Memory location to store to and load back from.
 * @param to_store: Value written to addr.
 * @return The value read back from addr after the store.
 */
static inline __attribute__((always_inline)) std::uint32_t store_then_load(volatile std::uint32_t *addr, std::uint32_t to_store)
{
    std::uint32_t result;
    __asm volatile("sw %2, %1; lw %0, %1" : "=r"(result) : "m"(*addr), "r"(to_store));
    return result;
}

/**
 * @brief Configure the unpacker hardware for both operands A and B.
 *
 * Programs the per-operand source/destination data formats, face dimensions and face counts via
 * configure_unpack_AB, and stores the per-operand tile sizes into the unpack GPRs.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @param unpA_src_format: Source data format of operand A in L1.
 * @param unpB_src_format: Source data format of operand B in L1.
 * @param unpA_dst_format: Destination data format operand A is converted to.
 * @param unpB_dst_format: Destination data format operand B is converted to.
 * @param unpA_face_r_dim: Rows per face for operand A.
 * @param unpB_face_r_dim: Rows per face for operand B.
 * @param unpA_num_faces: Number of faces for operand A, valid values = <1, 2, 4>.
 * @param unpB_num_faces: Number of faces for operand B, valid values = <1, 2, 4>.
 * @param unpA_tile_size: Tile size of operand A stored to the tile-size GPR.
 * @param unpB_tile_size: Tile size of operand B stored to the tile-size GPR.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_unpack_hw_configure_(
    const std::uint32_t unpA_src_format,
    const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t unpB_dst_format,
    const std::uint32_t unpA_face_r_dim,
    const std::uint32_t unpB_face_r_dim,
    const std::uint32_t unpA_num_faces,
    const std::uint32_t unpB_num_faces,
    const std::uint32_t unpA_tile_size = 0,
    const std::uint32_t unpB_tile_size = 0)
{
    LLK_ASSERT(unpA_num_faces == 1 || unpA_num_faces == 2 || unpA_num_faces == 4, "unpA_num_faces must be 1, 2, or 4");
    LLK_ASSERT(unpB_num_faces == 1 || unpB_num_faces == 2 || unpB_num_faces == 4, "unpB_num_faces must be 1, 2, or 4");

    // sstanisic todo: add tile_size_a and tile_size_b to operand state? (see #47440)
    llk::san::unpack_operand_configure(
        is_fp32_dest_acc_en,
        unpA_src_format,
        unpB_src_format,
        unpA_dst_format,
        unpB_dst_format,
        unpA_face_r_dim,
        unpB_face_r_dim,
        unpA_num_faces,
        unpB_num_faces);

    configure_unpack_AB<is_fp32_dest_acc_en, false, false, false>(
        unpA_src_format, unpB_src_format, unpA_dst_format, unpB_dst_format, unpA_face_r_dim, unpB_face_r_dim, 0, unpA_num_faces, unpB_num_faces);

    TT_SETDMAREG(0, LOWER_HALFWORD(unpA_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A));
    TT_SETDMAREG(0, LOWER_HALFWORD(unpB_tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B));
}

/**
 * @brief Configure stochastic rounding for the unpacker, FPU and packer.
 *
 * Sets the ALU rounding-mode register bits enabling stochastic rounding on the units selected by
 * the template mode.
 *
 * @tparam stoch_rnd_mode: Which units use stochastic rounding, values = <None/Fpu/Pack/All>
 */
template <StochRndType stoch_rnd_mode>
inline void _llk_unpack_configure_stoch_rnd_()
{
    constexpr std::uint32_t alu_stoch_rnd_mask =
        ALU_ROUNDING_MODE_Fpu_srnd_en_MASK | ALU_ROUNDING_MODE_Gasket_srnd_en_MASK | ALU_ROUNDING_MODE_Packer_srnd_en_MASK;
    constexpr bool fpu_srnd_en                     = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Fpu);
    constexpr bool pack_srnd_en                    = (stoch_rnd_mode == StochRndType::All) || (stoch_rnd_mode == StochRndType::Pack);
    alu_config_u alu_payload                       = {.val = 0};
    alu_payload.f.ALU_ROUNDING_MODE_Fpu_srnd_en    = fpu_srnd_en;
    alu_payload.f.ALU_ROUNDING_MODE_Gasket_srnd_en = pack_srnd_en;
    alu_payload.f.ALU_ROUNDING_MODE_Packer_srnd_en = pack_srnd_en;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_stoch_rnd_mask>(alu_payload.val);
}

/**
 * @brief Reconfigure the operand A (SrcA) data format at runtime.
 *
 * Updates the SrcA tile-descriptor source format, destination format and tile-size GPR; optionally
 * reprograms dim/stride registers for face-row-major unpacking.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam to_from_int8: Reconfiguring to/from an Int8 format (requires FP32 dest mode).
 * @param unpack_src_format: New source data format of operand A in L1.
 * @param unpack_dst_format: New destination data format operand A is converted to.
 * @param tile_size: New tile size of operand A stored to the tile-size GPR.
 * @param unpack_face_r_dim: Rows per face, used when reprogramming dim/stride.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note Caller contract: the SrcA-unsigned ALU bit (ALU_FORMAT_SPEC_REG0_SrcAUnsigned), and the math-side
 *       INT8 math-enable in @ref _llk_math_reconfig_data_format_srca_, are only reprogrammed when
 *       to_from_int8 is set. Set to_from_int8 = true for ANY reconfig that transitions to OR from an
 *       8-bit-integer / Int8 / Int32 format (e.g. UInt8 -> float); otherwise the previous unsigned/INT8
 *       state is left stale and the next op misinterprets the operand.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void _llk_unpack_reconfig_data_format_srca_impl_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t unpack_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpack_num_faces  = 4)
{
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    llk::san::unpack_operand_configure<true>(
        llk::san::IGNORE,
        unpack_src_format,
        llk::san::IGNORE,
        unpack_dst_format,
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE);

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring unpack to/from Int8 formats requires FP32 Dest mode enabled");
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW>((unpack_src_format == to_underlying(DataFormat::UInt8)) ? 1 : 0);
    }

    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(unpack_dst_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A)); // update gpr which holds tile size A

    if constexpr (dim_stride_target == p_dim_stride_target::FACE_ROW_MAJOR)
    {
        // Re-establish the canonical Z-stride baseline for srcA. Per-op brackets that mutate
        // this register (unpack-to-dest in unpack_A / unpack_tilize) restore to this baseline,
        // so it must be re-committed whenever the dst format changes.
        cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(canonical_unpA_z_stride(unpack_dst_format));

        // Re-establish the canonical Y-stride baseline for srcA. Per-op inits that mutate
        // this register (e.g. bcastA_B) restore back to this baseline on uninit, so the
        // baseline must be re-committed whenever the dst format changes.
        cfg_reg_rmw_tensix<UNP0_ADDR_CTRL_XY_REG_1_Ystride_ADDR32, UNP0_ADDR_CTRL_XY_REG_0_Ystride_SHAMT, UNP0_ADDR_CTRL_XY_REG_1_Ystride_MASK>(
            canonical_unpA_y_stride(unpack_dst_format));

        // Program unpacker0 per context x_dim (face size in l1)
        // Overrides value set by tile descriptor when thread override bit is set in unpack instruction
        const std::uint32_t face_dim = unpack_face_r_dim * FACE_C_DIM;
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(face_dim | (face_dim << 16));

        // Set Z-dim to number of faces
        cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, TILE_DESC_UPPER_HALFWORD_MASK>(0 | (unpack_num_faces << 16));
    }
}

/**
 * @brief Reconfigure the operand B (SrcB) data format at runtime.
 *
 * Updates the SrcB tile-descriptor source format, destination format and tile-size GPR; optionally
 * reprograms dim/stride registers for face-row-major unpacking.
 *
 * @tparam is_fp32_dest_acc_en: Whether the dest register accumulates in FP32.
 * @tparam dim_stride_target: Whether to reprogram dim/stride, values = <IGNORE/FACE_ROW_MAJOR>
 * @tparam to_from_int8: Reconfiguring to/from an Int8 format (requires FP32 dest mode).
 * @param unpack_src_format: New source data format of operand B in L1.
 * @param unpack_dst_format: New destination data format operand B is converted to.
 * @param tile_size: New tile size of operand B stored to the tile-size GPR.
 * @param unpack_face_r_dim: Rows per face, used when reprogramming dim/stride.
 * @param unpack_num_faces: Number of faces, valid values = <1, 2, 4>.
 * @note Caller contract: the SrcB-unsigned ALU bit (ALU_FORMAT_SPEC_REG0_SrcBUnsigned), and the math-side
 *       INT8 math-enable in @ref _llk_math_reconfig_data_format_srcb_, are only reprogrammed when
 *       to_from_int8 is set. Set to_from_int8 = true for ANY reconfig that transitions to OR from an
 *       8-bit-integer / Int8 / Int32 format (e.g. UInt8 -> float); otherwise the previous unsigned/INT8
 *       state is left stale and the next op misinterprets the operand.
 */
// TODO NC: Clean up as the part of tt-metal#34499
template <bool is_fp32_dest_acc_en, p_dim_stride_target dim_stride_target, bool to_from_int8 = false>
inline void _llk_unpack_reconfig_data_format_srcb_impl_(
    const std::uint32_t unpack_src_format,
    const std::uint32_t unpack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t unpack_face_r_dim = FACE_R_DIM,
    const std::uint32_t unpack_num_faces  = 4)
{
    LLK_ASSERT(unpack_num_faces == 1 || unpack_num_faces == 2 || unpack_num_faces == 4, "unpack_num_faces must be 1, 2, or 4");

    LLK_ASSERT(
        is_unpacker_format_conversion_supported_fp32_acc(
            static_cast<DataFormat>(unpack_src_format), static_cast<DataFormat>(unpack_dst_format), is_fp32_dest_acc_en),
        "Unsupported unpacker to register conversion.");

    llk::san::unpack_operand_configure<true>(
        llk::san::IGNORE,
        llk::san::IGNORE,
        unpack_src_format,
        llk::san::IGNORE,
        unpack_dst_format,
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE,
        llk::san::IGNORE);

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);
    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring unpack to/from Int8 formats requires FP32 Dest mode enabled");
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcBUnsigned_RMW>((unpack_src_format == to_underlying(DataFormat::UInt8)) ? 1 : 0);
    }

    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpack_dst_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B)); // update gpr which holds tile size B

    if constexpr (dim_stride_target == p_dim_stride_target::FACE_ROW_MAJOR)
    {
        std::uint32_t unpack_ch1_x_stride = datum_size_in_bytes(unpack_dst_format);
        std::uint32_t unpack_ch1_z_stride = FACE_C_DIM * FACE_R_DIM * unpack_ch1_x_stride;
        cfg_reg_rmw_tensix<UNP1_ADDR_CTRL_ZW_REG_1_Zstride_RMW>(unpack_ch1_z_stride);

        // Set X-dim to face_r_dim * FACE_C_DIM
        cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, TILE_DESC_UPPER_HALFWORD_MASK>((unpack_face_r_dim * FACE_C_DIM) << 16);

        // Set Z-dim to number of faces
        cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1, 0, TILE_DESC_UPPER_HALFWORD_MASK>(0 | (unpack_num_faces << 16));
    }
}

/**
 * @brief Enable Int8 math on the FPU.
 */
inline void _llk_enable_int8_fpu_math_()
{
    enable_int8_fpu_math();
}

/**
 * @brief Mark SrcA and SrcB as data-valid without unpacking real data.
 *
 * Issues zero-source UNPACR NOPs that set the data-valid flag on both source registers, e.g. to
 * unblock the math thread when an operand is not actually fed from L1.
 *
 * @ref _llk_math_transpose_dest_ on the math thread relies on this to mark SrcB valid for its MOVD2B/MOVB2D sequence.
 */
inline void _llk_unpack_set_srcb_dummy_valid_()
{
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK | p_stall::SRCA_CLR | p_stall::SRCB_CLR);
    TTI_UNPACR_NOP(SrcB, p_unpacr_nop::UNP_SET_DVALID);
    TTI_UNPACR_NOP(SrcA, p_unpacr_nop::UNP_SET_DVALID);
}

/**
 * @brief Validates L1 addresses and configures unpack base addresses in the configuration registers
 *
 * This helper function validates that both address_a and address_b are within the valid L1 memory region,
 * then configures the appropriate THCON base address registers based on the unpack configuration context.
 *
 * @param address_a: Address for unpacker A (THCON_SEC0).
 * @param address_b: Address for unpacker B (THCON_SEC1).
 * @param cfg: Pointer to configuration registers.
 */
inline void _llk_unpack_configure_addresses_(const std::uint32_t address_a, const std::uint32_t address_b, volatile std::uint32_t tt_reg_ptr *cfg)
{
    LLK_ASSERT(is_valid_L1_address(address_a), "L1 address_a must be in valid L1 memory region");
    LLK_ASSERT(is_valid_L1_address(address_b), "L1 address_b must be in valid L1 memory region");

    // Program srcA and srcB base addresses
    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_b;
    }
}

/**
 * @brief Validates L1 address and configures unpack base address for a single unpacker
 *
 * This helper function validates that the address is within the valid L1 memory region,
 * then configures the appropriate THCON_SEC0 base address register based on the unpack configuration context.
 *
 * @param address: Address for unpacker A (THCON_SEC0).
 * @param cfg: Pointer to configuration registers.
 */
inline void _llk_unpack_configure_single_address_(const std::uint32_t address, volatile std::uint32_t tt_reg_ptr *cfg)
{
    LLK_ASSERT(is_valid_L1_address(address), "L1 base_address must be in valid L1 memory region");

    // Program srcA base address
    if (0 == unp_cfg_context)
    {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
    }
    else
    {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
    }
}
