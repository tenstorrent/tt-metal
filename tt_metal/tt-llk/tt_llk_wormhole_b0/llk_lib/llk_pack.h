// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "llk_pack_common.h"
#include "sanitizer/api.h"

using namespace ckernel;
using namespace ckernel::packer;

namespace llk_pack_internal
{
static std::uint32_t configured_num_tiles   = 1;
static std::uint32_t configured_zero_output = 0;

/**
 * @brief Emit the final close/reset PACR for a multi-tile pack run.
 *
 * The multi-tile MOP closes tiles 0..N-2 internally; this emits the one remaining PACR that closes the
 * last tile and restores the packer row counters to the single-tile state for the next caller.
 *
 * @tparam zero_output: When true, the packer emits zeros instead of dest data.
 */
template <bool zero_output = false>
inline void finalize_multitile_pack_tail()
{
    constexpr std::uint32_t ZERO_OUTPUT_FLAG = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;
    // The multi-tile MOP closes tiles 0..N-2 inside the template and advances
    // the L1 destination address after each one. The last tile still needs one
    // final PACR to close the tile and restore packer row counters to the
    // normal single-tile state for the next caller.
    TTI_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, 0xf, 0, 1, 0, 1);
}

/**
 * @brief Pack multiple tiles in one MOP run, advancing the L1 destination per tile.
 *
 * Selects the source dest tile, programs the L1 destination address, runs the multi-tile MOP, and
 * emits the final close/reset PACR. Kept out of line to share the sequence across call sites and keep
 * the single-tile hot path inline.
 *
 * @param tile_index: Index of the first source tile in the destination register.
 * @param address: L1 destination base address for the packed tiles.
 */
// Keep the multi-tile path out of line to share this pack sequence across call
// sites and reduce TRISC code size. The single-tile hot path stays inline.
static __attribute__((noinline, noclone)) void pack_multitile(const std::uint32_t tile_index, const std::uint32_t address)
{
    set_dst_write_addr(tile_index);
    LLK_ASSERT(is_valid_L1_address(address), "L1 address must be in valid L1 memory region");
    std::uint32_t new_l1_addr = (1 << 31) | address;
    TT_SETDMAREG(0, LOWER_HALFWORD(address), 0, LO_16(p_gpr_pack::OUTPUT_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(new_l1_addr), 0, HI_16(p_gpr_pack::OUTPUT_ADDR));
    TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR);
    // The programmed MOP performs the blocked sequence for this whole call;
    // the explicit tail below is only the final close/reset step.
    mop_run(1, 1);
    if (llk_pack_internal::configured_zero_output == p_pacr::P_ZERO_OUTPUT_ENABLED)
    {
        finalize_multitile_pack_tail<true>();
    }
    else
    {
        finalize_multitile_pack_tail<false>();
    }
}
} // namespace llk_pack_internal

/**
 * @brief Compute the packed L1 footprint (in bytes) of a tile for a given format and datum count.
 *
 * Returns the real packed byte size, correcting SCALE_DATUM_SIZE's one-byte-per-datum result for the
 * sub-byte BFP payloads (Bfp4 = 2 datums/byte, Bfp2 = 4 datums/byte) and adding the per-16-datum
 * exponent byte that all BFP formats store alongside their mantissas.
 *
 * @param pack_dst_format: Destination (L1) data format.
 * @param datum_count: Number of datums in the tile.
 * @return Packed tile size in bytes.
 */
inline std::uint32_t _llk_pack_output_size_bytes_(const std::uint32_t pack_dst_format, const std::uint32_t datum_count)
{
    std::uint32_t packed_tile_size_bytes = SCALE_DATUM_SIZE(pack_dst_format, datum_count);

    // SCALE_DATUM_SIZE keeps one-byte-per-datum compatibility for the sub-byte
    // BFP payload formats. Pack address programming needs the real packed L1
    // footprint instead: Bfp4 payload is 2 datums/byte, Bfp2 is 4 datums/byte,
    // and all BFP formats also store one exponent byte per 16 datums
    // alongside the mantissas.
    if (pack_dst_format == to_underlying(DataFormat::Bfp4) || pack_dst_format == to_underlying(DataFormat::Bfp4_b))
    {
        packed_tile_size_bytes /= 2;
    }
    else if (pack_dst_format == to_underlying(DataFormat::Bfp2) || pack_dst_format == to_underlying(DataFormat::Bfp2_b))
    {
        packed_tile_size_bytes /= 4;
    }

    if (IS_BFP_FORMAT(pack_dst_format))
    {
        packed_tile_size_bytes += datum_count / 16;
    }

    return packed_tile_size_bytes;
}

/**
 * @brief Compute the per-tile L1 address offset, in 16-byte words, for multi-tile packing.
 *
 * Sizes one tile of the given geometry via @ref _llk_pack_output_size_bytes_ and returns the result as
 * a count of 16-byte words, the stride used to advance the L1 destination between tiles.
 *
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @return Per-tile L1 offset in 16-byte words.
 */
inline std::uint32_t _llk_pack_output_addr_offset_words_(
    const std::uint32_t pack_dst_format, const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    const std::uint32_t tile_elements = face_r_dim * FACE_C_DIM * num_faces;
    std::uint32_t tile_size           = _llk_pack_output_size_bytes_(pack_dst_format, tile_elements);

    return tile_size >> 4;
}

/**
 * @brief Configure the packer address-modification (ADDR_MOD) slots for the selected pack mode.
 *
 * Programs ADDR_MOD_0/1/2 with the src/dest Y and Z increment/clear patterns the pack MOP relies
 * on to traverse the destination register and step through faces for the given layout.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 */
template <PackMode pack_mode = PackMode::Default>
inline void _llk_pack_configure_addrmod_()
{
    addr_mod_pack_t {
        .y_src = {.incr = 15}, // 4-bit value so max is 15. incadcxy will increment it by 1
        .y_dst = {.incr = 1},
    }
        .set(ADDR_MOD_0);

    if constexpr (pack_mode == PackMode::Untilize)
    {
        addr_mod_pack_t {
            .y_src = {.incr = 1, .clr = 0, .cr = 1},
            .y_dst = {.incr = 1, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);
    }
    else
    {
        addr_mod_pack_t {
            .y_src = {.incr = 0, .clr = 1, .cr = 0},
            .y_dst = {.incr = 0, .clr = 1, .cr = 0},
            .z_src = {.incr = 0, .clr = 0},
            .z_dst = {.incr = 0, .clr = 0},
        }
            .set(ADDR_MOD_1);
    }

    addr_mod_pack_t {
        .y_src = {.incr = 0, .clr = 1, .cr = 0},
        .y_dst = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);
}

/**
 * @brief Build and program the packer MOP template for the selected pack mode.
 *
 * Programs the ckernel MOP with the PACR instruction sequence that packs one tile worth of data,
 * selecting packer interfaces and ADDR_MODs per the layout.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 * @param narrow_tile: True if the tile occupies fewer than the full set of packer interfaces.
 * @param num_tiles: Number of tiles processed per MOP run.
 * @note @ref _llk_pack_configure_addrmod_ must have programmed the ADDR_MOD slots for the same pack_mode.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false>
inline void _llk_pack_mop_config_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false,
    const std::uint32_t num_tiles  = 1)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    LLK_ASSERT(num_tiles >= 1, "num_tiles must be >= 1");

    if constexpr (pack_mode != PackMode::Untilize)
    {
        if (num_tiles > 1)
        {
            LLK_ASSERT(num_faces == 4, "multi-tile pack currently supports full 4-face tiles");
            LLK_ASSERT(!partial_face, "multi-tile pack does not support partial-face tiles");
            LLK_ASSERT(!narrow_tile, "multi-tile pack does not support narrow tiles");
            TT_SETDMAREG(
                p_setdmareg::PAYLOAD_IMMEDIATE,
                _llk_pack_output_addr_offset_words_(pack_dst_format, face_r_dim, num_faces),
                p_setdmareg::MODE_IMMEDIATE,
                LO_16(p_gpr_pack::OUTPUT_ADDR_OFFSET));
        }
    }

    const std::uint32_t PACKCNT               = (partial_face && IS_BFP_FORMAT(pack_dst_format)) ? 1 : num_faces;
    constexpr std::uint32_t MEGAROW           = 1;
    constexpr std::uint32_t ZERO_OUTPUT_FLAG  = zero_output ? p_pacr::P_ZERO_OUTPUT_ENABLED : p_pacr::P_ZERO_OUTPUT_DISABLED;
    constexpr std::uint32_t MOP_INNER_LOOP    = 1;
    llk_pack_internal::configured_num_tiles   = num_tiles;
    llk_pack_internal::configured_zero_output = ZERO_OUTPUT_FLAG;

    if constexpr (pack_mode != PackMode::Untilize)
    {
        if (partial_face && IS_BFP_FORMAT(pack_dst_format))
        {
            LLK_ASSERT(num_tiles == 1, "multi-tile partial-face BFP pack is not supported");
            constexpr std::uint32_t MOP_OUTER_LOOP = 1;
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0)); // Don't close the tile, point to the next face
            tmp.set_loop_op0(TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));                                     // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
            tmp.set_loop_op1(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1)); // Close the tile
            tmp.program();
        }
        else if (num_tiles == 1)
        {
            constexpr std::uint32_t MOP_OUTER_LOOP = 1;
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.program();
        }
        else
        {
            // Multi-tile blocked pack is encoded as:
            // 1. start_op: pack tile 0
            // 2. outer loop (num_tiles - 1 times): advance source/destination to
            //    the next tile and commit the new L1 destination address
            // 3. end_ops: flush the new destination address into FLOP space and
            //    reset per-tile pack counters so the final explicit PACR can
            //    close the last packed tile
            ckernel::ckernel_template tmp(
                num_tiles - 1,
                1,
                TT_OP_INCADCZW(p_setadc::PAC, 0, 0, 1, 0),
                TT_OP_ADDDMAREG(p_adddmareg::REG_PLUS_REG, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR, p_gpr_pack::OUTPUT_ADDR_OFFSET));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.set_end_ops(
                TT_OP_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG1_L1_Dest_addr_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_pack::OUTPUT_ADDR),
                TT_OP_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 0));
            tmp.program();
        }
    }
    else
    {
        const std::uint32_t MOP_OUTER_LOOP = ((face_r_dim == 1) || narrow_tile) ? 1 : (face_r_dim >> 1);
        LLK_ASSERT(num_tiles == 1, "multi-tile pack is only supported for non-untilize mode");

        if ((face_r_dim == 1) || narrow_tile)
        {
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 1));
            tmp.program();
        }
        else
        {
            // Inc ch0_y+=1 (addr_mod_0 will increment by 15)
            ckernel::ckernel_template tmp(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_INCADCXY(p_setadc::PAC, 0, 0, 1, 0));
            tmp.set_start_op(TT_OP_PACR(ADDR_MOD_0, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
            tmp.set_end_op(TT_OP_PACR(ADDR_MOD_1, ZERO_OUTPUT_FLAG, PACK_SEL(PACKCNT), 0, MEGAROW, 0, 0));
            tmp.program();
        }
    }
}

/**
 * @brief Reconfigure the packer source/destination data formats and tile geometry at runtime.
 *
 * Used to switch the packer to a new data format without a full HW re-configure.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param tile_size: Size of one output tile in bytes.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 * @param narrow_tile: True if the tile occupies fewer than the full set of packer interfaces.
 */
template <bool is_fp32_dest_acc_en>
inline void _llk_pack_reconfig_data_format_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim          = FACE_R_DIM,
    const std::uint32_t num_faces           = 4,
    const bool partial_face                 = false,
    [[maybe_unused]] const bool narrow_tile = false)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    llk::san::pack_operand_configure<true>(
        is_fp32_dest_acc_en, pack_src_format, pack_dst_format, face_r_dim, llk::san::IGNORE, num_faces, partial_face, narrow_tile);

    reconfig_packer_data_format<is_fp32_dest_acc_en>(pack_src_format, pack_dst_format, tile_size, face_r_dim, num_faces, partial_face);
}

/**
 * @brief Enable or disable reading the destination register as 32-bit data for the packer.
 *
 * @param enable: True to read dest as 32-bit (FP32) data, false otherwise.
 * @note Stalls on the pack pipe before modifying the PCK_DEST_RD_CTRL config register.
 */
inline void _llk_pack_set_fp32_dest_acc_(bool enable)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::PACK);
    cfg_reg_rmw_tensix<PCK_DEST_RD_CTRL_Read_32b_data_RMW>(enable);
}

/**
 * @brief One-time hardware configuration of the packer for a given data format and tile geometry.
 *
 * Programs the packer config registers (formats, strides, relu) for the chosen pack mode. Call once
 * before the init/execute sequence.
 *
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @param pack_src_format: Source (dest register) data format.
 * @param pack_dst_format: Destination (L1) data format.
 * @param tile_size: Size of one output tile in bytes.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 * @param narrow_tile: True if the tile occupies fewer than the full set of packer interfaces.
 * @param relu_config: Packed relu mode and threshold configuration (0 disables relu).
 */
template <bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_hw_configure_(
    const std::uint32_t pack_src_format,
    const std::uint32_t pack_dst_format,
    const std::uint32_t tile_size,
    const std::uint32_t face_r_dim  = FACE_R_DIM,
    const std::uint32_t num_faces   = 4,
    const bool partial_face         = false,
    const bool narrow_tile          = false,
    const std::uint32_t relu_config = 0)
{
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    llk::san::pack_operand_configure(is_fp32_dest_acc_en, pack_src_format, pack_dst_format, face_r_dim, llk::san::IGNORE, num_faces, partial_face, narrow_tile);

    configure_pack<is_fp32_dest_acc_en, pack_mode>(pack_src_format, pack_dst_format, tile_size, face_r_dim, num_faces, partial_face, narrow_tile, relu_config);
}

/**
 * @brief Initialize the packer (addrmod + MOP) for a pack op.
 *
 * Programs the ADDR_MODs, the MOP template, and the packer X (datum) counter. The skip_* template flags
 * let a caller reuse state already established by a prior init or hw-configure.
 *
 * @tparam pack_mode: Packing layout, values = <Default/Untilize>
 * @tparam zero_output: When true, packer emits zeros instead of dest data.
 * @tparam skip_addrmod_config: When true, leave ADDR_MOD slots untouched (assume already programmed).
 * @tparam skip_packer_strides: Deprecated no-op kept for API/ABI symmetry. Init no longer programs the
 *         packer strides / L1 offset (those are owned by @ref configure_pack / @ref reconfig_packer_data_format),
 *         so this flag has no effect; retained because existing callers (e.g. SDPA `compute_streaming.hpp`)
 *         and shared tests still pass it.
 * @param pack_dst_format: Destination (L1) data format.
 * @param face_r_dim: Number of rows per face.
 * @param num_faces: Faces per tile, valid values = <1, 2, 4>
 * @param partial_face: True if packing a partial (sub-face-row) face.
 * @param narrow_tile: True if the tile occupies fewer than the full set of packer interfaces.
 * @param num_tiles: Number of tiles processed per MOP run.
 * @note Init programs ADDR_MOD + MOP + the packer X (datum) counter. The X counter is pack_mode-dependent
 *       (Untilize packs a single face row, Default a full face), so it is per-op state owned here; the
 *       format-level state (packer strides, L1 offset) is owned by @ref configure_pack
 *       (@ref _llk_pack_hw_configure_) and @ref reconfig_packer_data_format
 *       (@ref _llk_pack_reconfig_data_format_), one of which must run before this init.
 * @note Pair with @ref _llk_pack_uninit_ after the matching @ref _llk_pack_ execute calls.
 */
template <PackMode pack_mode = PackMode::Default, bool zero_output = false, bool skip_addrmod_config = false, [[maybe_unused]] bool skip_packer_strides = false>
inline void _llk_pack_init_(
    const std::uint32_t pack_dst_format,
    const std::uint32_t face_r_dim = FACE_R_DIM,
    const std::uint32_t num_faces  = 4,
    const bool partial_face        = false,
    const bool narrow_tile         = false,
    const std::uint32_t num_tiles  = 1)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize, "Wormhole B0 pack init supports only PackMode::Default and PackMode::Untilize");
    LLK_ASSERT(num_faces == 1 || num_faces == 2 || num_faces == 4, "num_faces must be 1, 2, or 4");
    llk::san::pack_operand_check(llk::san::IGNORE, llk::san::IGNORE, pack_dst_format, face_r_dim, llk::san::IGNORE, num_faces, partial_face, narrow_tile);
    // sstanisic todo: sanitizer: propagate enum (see #47440)
    llk::san::operation_init<llk::san::Operation::Pack>(pack_mode == PackMode::Untilize);

    if constexpr (!skip_addrmod_config)
    {
        _llk_pack_configure_addrmod_<pack_mode>();
    }
    _llk_pack_mop_config_<pack_mode, zero_output>(pack_dst_format, face_r_dim, num_faces, partial_face, narrow_tile, num_tiles);
    if constexpr (!skip_packer_strides)
    {
        set_packer_l1_offset(pack_dst_format, face_r_dim);
    }

    // Program the packer X (datum) counter. This value is pack_mode-dependent (Untilize packs a single
    // face row, Default packs a full face), so it is per-op state that must be (re)established by init
    // for the geometry/mode of this pack. configure_pack / reconfig_packer_data_format only ever run in
    // PackMode::Default, so they cannot establish the Untilize value — init owns this counter.
    const std::uint32_t face_dim   = face_r_dim * FACE_C_DIM;
    const std::uint32_t pack_x_dim = (narrow_tile || pack_mode != PackMode::Untilize) ? face_dim : FACE_R_DIM;
    TT_SETADCXX(p_setadc::PAC, pack_x_dim - 1, 0x0);
}

/**
 * @brief No-op teardown after a pack op.
 *
 * The packer x-start/x-end is transient and reprogrammed by each operation's init (see tt-llk#1036),
 * so there is nothing to restore here.
 *
 * @note Call @ref _llk_pack_init_ before this function.
 */
inline void _llk_pack_uninit_()
{
    // sstanisic todo: contract cannot be enforced if Pack has an uninit, without killing performance
    // llk::san::operation_uninit<llk::san::Operation::Pack>();
}

/**
 * @brief Pack one tile from the destination register to an L1 address.
 *
 * Selects the source dest tile, programs the L1 destination address, and runs the packer MOP.
 * Untilize mode emits a closing PACR, and the multi-tile path runs the multi-tile pack sequence.
 *
 * @tparam Dst: Destination sync mode, values = <SyncHalf/SyncFull>
 * @tparam is_fp32_dest_acc_en: True if the destination register accumulates in FP32.
 * @tparam pack_mode: Packing layout, values = <Default/Untilize> (Tilize not supported here)
 * @param tile_index: Index of the source tile in the destination register.
 * @param address: L1 destination address for the packed tile.
 * @note Call @ref _llk_pack_init_ with matching template/runtime args before this function, and
 *       @ref _llk_pack_uninit_ once all pack calls are complete.
 */
template <DstSync Dst, bool is_fp32_dest_acc_en, PackMode pack_mode = PackMode::Default>
inline void _llk_pack_(const std::uint32_t tile_index, const std::uint32_t address)
{
    static_assert(
        pack_mode == PackMode::Default || pack_mode == PackMode::Untilize, "Wormhole B0: _llk_pack_ supports PackMode::Default and PackMode::Untilize only");
    // sstanisic todo: sanitizer: propagate enum (see #47440)
    llk::san::operation_check<llk::san::Operation::Pack>(pack_mode == PackMode::Untilize);

    if constexpr (pack_mode != PackMode::Untilize)
    {
        if (llk_pack_internal::configured_num_tiles > 1)
        {
            llk_pack_internal::pack_multitile(tile_index, address);
            return;
        }
    }

    set_dst_write_addr(tile_index);

    program_packer_destination(address);

    mop_run(1, 1);

    if constexpr (pack_mode == PackMode::Untilize)
    {
        TTI_PACR(ADDR_MOD_2, 0, 0xf, 0, 0, 1, 1); // close tile
    }
}
