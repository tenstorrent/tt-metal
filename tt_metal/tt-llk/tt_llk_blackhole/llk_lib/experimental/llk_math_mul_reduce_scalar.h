// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_sfpu_common.h"
#include "tensor_shape.h"
#include "tensor_shape_coverage_math.h"

using namespace ckernel;

/*************************************************************************
 * LLK MUL REDUCE SCALAR - Low-level reduce operations for fused mul+reduce
 *************************************************************************/

// Helper macros for moving destination to source registers
#define MOVD2A_4_ROWS(base_offset, row_offset) \
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + (row_offset) * 4, ADDR_MOD_0, p_movd2a::MOV_4_ROWS, (base_offset) + (row_offset) * 4);

#define MOVD2A_8_ROWS(base_offset, row_offset) \
    TTI_MOVD2A(0, p_mova2d::MATH_HALO_ROWS + (row_offset) * 8, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, (base_offset) + (row_offset) * 8);

#define MOVD2A_16_ROWS(base_offset) \
    MOVD2A_8_ROWS(base_offset, 0)   \
    MOVD2A_8_ROWS(base_offset, 1)   \
    MOVD2A_8_ROWS(base_offset, 2)   \
    MOVD2A_8_ROWS(base_offset, 3)   \
    MOVD2A_8_ROWS(base_offset, 4)   \
    MOVD2A_8_ROWS(base_offset, 5)   \
    MOVD2A_8_ROWS(base_offset, 6)   \
    MOVD2A_8_ROWS(base_offset, 7)

#define MOVD2A_FACE(base_offset, face)         \
    MOVD2A_4_ROWS(base_offset, (face) * 4 + 0) \
    MOVD2A_4_ROWS(base_offset, (face) * 4 + 1) \
    MOVD2A_4_ROWS(base_offset, (face) * 4 + 2) \
    MOVD2A_4_ROWS(base_offset, (face) * 4 + 3)

/**
 * @brief Execute GAPOOL operations with optional high fidelity phases
 *
 * Executes the appropriate number of GAPOOL instructions based on fidelity level.
 * For high fidelity (math_fidelity > LoFi), executes Number of Fidelity Phases - 1
 * iterations with ADDR_MOD_2, followed by one final GAPOOL with ADDR_MOD_0.
 *
 * @tparam math_fidelity: MathFidelity value LoFi, HiFi2, HiFi3, HiFi4
 */
template <MathFidelity math_fidelity, std::uint32_t last_addr_mod = ADDR_MOD_0>
inline void execute_high_fidelity_gapool()
{
    if constexpr (math_fidelity >= MathFidelity::HiFi2)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }
    if constexpr (math_fidelity >= MathFidelity::HiFi3)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }
    if constexpr (math_fidelity >= MathFidelity::HiFi4)
    {
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_2, p_gpool::INDEX_DIS, 0);
    }

    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, last_addr_mod, p_gpool::INDEX_DIS, 0);
}

/**
 * @brief Move destination tile to source registers for mul_reduce_scalar
 *
 * Moves a whole tile to source A, or its first 16 rows to source B.
 *
 * @tparam binary_reuse_dest Direction: DEST_TO_SRCA or DEST_TO_SRCB
 * @param idst Destination tile index (0-7)
 */
template <EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_mul_reduce_scalar_move_dest_to_src_([[maybe_unused]] std::uint32_t idst = 0)
{
    if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCA)
    {
        if (idst == 0)
        {
            TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, get_dest_buffer_base());
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        }

        math::srca_bank_wait();

        switch (idst)
        {
            case 0:
                MOVD2A_16_ROWS(0);
                break;
            case 1:
                MOVD2A_16_ROWS(64);
                break;
            case 2:
                MOVD2A_16_ROWS(128);
                break;
            case 3:
                MOVD2A_16_ROWS(192);
                break;
            case 4:
                MOVD2A_16_ROWS(256);
                break;
            case 5:
                MOVD2A_16_ROWS(320);
                break;
            case 6:
                MOVD2A_16_ROWS(384);
                break;
            case 7:
                MOVD2A_16_ROWS(448);
                break;
            default:
                break;
        }
    }
    else if constexpr (binary_reuse_dest == EltwiseBinaryReuseDestType::DEST_TO_SRCB)
    {
        math::srcb_bank_wait();
        // dst may have been written by the SFPU just before this move (e.g. a fill). The MATH bit in the
        // bank wait only tracks FPU writes, so also drain the SFPU before the FPU copies dst into SrcB.
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);

        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_BD);

        // GAPOOL reads SrcB rows 0 to 3 and only row 0 reaches the result; ADDR_MOD_0 keeps SrcA's row for the passes.
        TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET + 0, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 0);
    }
}

inline void _llk_math_mul_reduce_scalar_move_faces_to_srca_(const std::uint32_t idst, const ckernel::TensorShape tensor_shape)
{
    // Later moves find the bank still valid, and MOVD2A itself waits for pending DEST writes.
    if (idst == 0)
    {
        // dest_offset_id is 0 or 1 and flips every section: a shift, not a select, keeps it off the branch predictor.
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dest_offset_id * DEST_REGISTER_HALF_SIZE);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
        math::srca_bank_wait();
    }

    const std::uint32_t num_faces = tensor_shape.total_num_faces();
// A whole slot moves in eight-row steps; a one- or two-face tile keeps four-row moves, which read faster there.
#define MOVD2A_FACES(base_offset, num_faces) \
    if ((num_faces) > 2)                     \
    {                                        \
        MOVD2A_16_ROWS(base_offset)          \
    }                                        \
    else                                     \
    {                                        \
        MOVD2A_FACE(base_offset, 0)          \
        if ((num_faces) > 1)                 \
        {                                    \
            MOVD2A_FACE(base_offset, 1)      \
        }                                    \
    }
    switch (idst)
    {
        case 0:
            MOVD2A_FACES(0, num_faces);
            break;
        case 1:
            MOVD2A_FACES(64, num_faces);
            break;
        case 2:
            MOVD2A_FACES(128, num_faces);
            break;
        case 3:
            MOVD2A_FACES(192, num_faces);
            break;
        case 4:
            MOVD2A_FACES(256, num_faces);
            break;
        case 5:
            MOVD2A_FACES(320, num_faces);
            break;
        case 6:
            MOVD2A_FACES(384, num_faces);
            break;
        case 7:
            MOVD2A_FACES(448, num_faces);
            break;
        default:
            break;
    }
#undef MOVD2A_FACES
}

// A 16x32 product takes 32 rows, so two fit one 64-row DEST slot.
inline bool _llk_math_mul_reduce_scalar_shares_slot_(const ckernel::TensorShape tensor_shape)
{
    return tensor_shape.face_r_dim == ckernel::MAX_FACE_R_DIM && tensor_shape.num_faces_r_dim == 1 && tensor_shape.num_faces_c_dim == 2;
}

// Multiplies one 16x32 product into rows 32 * (product_index % 2) of DEST slot product_index / 2.
template <MathFidelity math_fidelity>
inline void _llk_math_mul_reduce_scalar_mul_half_slot_(const ckernel::TensorShape tensor_shape, const std::uint32_t product_index)
{
    LLK_ASSERT(_llk_math_mul_reduce_scalar_shares_slot_(tensor_shape), "the half-slot multiply is for 16x32 products");
    LLK_VALIDATE_TENSOR_SHAPE_MATH("_llk_math_mul_reduce_scalar_mul_half_slot_", tensor_shape);
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "mul_half_slot: Src zero-substitution flag is not in DEFAULT state");
    math::set_dst_write_addr<DstTileShape::Tile32x16, UnpackDestination::SrcRegs>(product_index);
    const std::uint32_t fidelity_loop = is_high_fidelity(math_fidelity) ? tensor_shape.total_num_faces() : 1;
#pragma GCC unroll 0
    for (std::uint32_t i = 0; i < fidelity_loop; i++)
    {
        ckernel_template::run();
    }
    math::clear_dst_reg_addr();
}

// share_slots must match how the products were written.
template <bool share_slots>
inline void _llk_math_mul_reduce_scalar_move_product_(const std::uint32_t i, const std::uint32_t num_products, const ckernel::TensorShape tensor_shape)
{
    if (!share_slots || !_llk_math_mul_reduce_scalar_shares_slot_(tensor_shape))
    {
        _llk_math_mul_reduce_scalar_move_faces_to_srca_(i, tensor_shape);
    }
    else if ((i & 1) == 0)
    {
        _llk_math_mul_reduce_scalar_move_faces_to_srca_(i / 2, i + 1 < num_products ? ckernel::DEFAULT_TENSOR_SHAPE : tensor_shape);
    }
}

/**
 * @brief Configure address modifiers for mul_reduce_scalar operations
 *
 * Sets up address modifiers for source A, source B, destination, and fidelity
 * registers based on the specified math fidelity level.
 *
 * @tparam MathFidelity math_fidelity -> MathFidelity value LoFi, HiFi2, HiFi3, HiFi4
 */
template <MathFidelity math_fidelity>
inline void mul_reduce_scalar_configure_addrmod()
{
    constexpr std::uint32_t FIDELITY_INCREMENT = 1;
    constexpr bool HIGH_FIDELITY               = is_high_fidelity(math_fidelity);

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);
    addr_mod_t {.srca = {.incr = 16}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.clr = 1}}.set(ADDR_MOD_1);

    if constexpr (HIGH_FIDELITY)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = FIDELITY_INCREMENT}}.set(ADDR_MOD_2);
    }
}

/**
 * @brief Initialize reduce for mul_reduce_scalar operation
 *
 * Configures address modifiers and resets counters for the fused
 * multiply-reduce-scalar operation.
 *
 * @tparam is_fp32_dest_acc_en If true, enables FP32 destination accumulation
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 * @tparam enforce_fp32_accumulation If true, enforces FP32 accumulation (requires is_fp32_dest_acc_en)
 */
template <bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void _llk_math_mul_reduce_scalar_init_()
{
    mul_reduce_scalar_configure_addrmod<math_fidelity>();

    if constexpr (enforce_fp32_accumulation)
    {
        static_assert(is_fp32_dest_acc_en, "FP32 Dest must be enabled for FP32 accumulation");
    }
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);
    math::reset_counters(p_setrwc::SET_ABD_F);
    math::_configure_default_zero_flag_state_();
}

/**
 * @brief Perform column reduction for mul_reduce_scalar (accumulates across tiles)
 *
 * Performs column-wise reduction that accumulates results into the specified
 * destination tile. Used in a loop to accumulate multiple input tiles.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 * @param tensor_shape Shape of the operand tile (4 faces for 32x32, 2 faces for a 16x32 tiny tile)
 * @note Without tile_setup, the SFPU must have drained and the DEST address must already be at the accumulation tile.
 */
template <MathFidelity math_fidelity, bool tile_setup = true>
inline void _llk_math_mul_reduce_column_(const std::uint32_t dst_index, const ckernel::TensorShape tensor_shape = ckernel::DEFAULT_TENSOR_SHAPE)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    const std::uint32_t num_row_tiles = tensor_shape.num_faces_r_dim;

    // dst[dst_index] may have just been zeroed through the SFPU; GAPOOL accumulates into it, so drain first.
    if constexpr (tile_setup)
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
        math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }

    // ADDR_MOD_1 steps SrcA to the next face after the last fidelity pass, in face order.
    const std::uint32_t num_faces = tensor_shape.total_num_faces();
#pragma GCC unroll 4
    for (std::uint32_t face = 0; face < num_faces; face++)
    {
        execute_high_fidelity_gapool<math_fidelity, ADDR_MOD_1>();
    }
    if (num_row_tiles > 1)
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_AD);
    }
    else
    {
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
    }
}

// Zeroes the face the column passes pool into; call once the SFPU has drained.
template <bool is_fp32_dest_acc_en>
inline void _llk_math_mul_reduce_scalar_clear_pool_face_(const std::uint32_t dst_index)
{
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    constexpr std::uint32_t tiles_per_bank = is_fp32_dest_acc_en ? MAX_TILES_IN_HALF_DEST >> 1 : MAX_TILES_IN_HALF_DEST;
    TT_ZEROACC(p_zeroacc::CLR_16, is_fp32_dest_acc_en, 0, ADDR_MOD_0, get_dest_index_in_faces(dst_index & (tiles_per_bank - 1), 0));
}

/**
 * @brief Perform final scalar reduction for mul_reduce_scalar
 *
 * Collapses a column-reduced tile into a single scalar value stored in the
 * first element of the output tile.
 *
 * @tparam MATH_FIDELITY_DESC Math fidelity descriptor (0 = default, higher = more precision)
 */
template <MathFidelity math_fidelity>
inline void _llk_math_mul_reduce_scalar_()
{
    math::srcb_bank_wait();

    // Copy row 0 from dest to srcB (rows 16-31 as scratch) and transpose
    TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
    TTI_GATESRCRST(0b1, 0b1);
    TTI_TRNSPSRCB;
    TTI_GATESRCRST(0b1, 0b1);

    // Copy all 16 rows from srcB to srcA
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
    TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
    TTI_GATESRCRST(0b1, 0b1);

    TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 0);

    execute_high_fidelity_gapool<math_fidelity>();
}

/**
 * @brief Clear data valid flags after mul_reduce_scalar operation
 *
 * Clears DVALID flags and resets all counters. Should be called after
 * mul_reduce_scalar operations are complete.
 */
inline void _llk_math_mul_reduce_scalar_clear_dvalid_()
{
    TTI_CLEARDVALID(p_setrwc::CLR_AB, 0);
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
}

/*************************************************************************
 * Compiled row of 32x32 tiles, two faces per SrcA/SrcB bank: the unpacker writes faces 2h and 2h + 1 of tile i to
 * SrcA/SrcB rows 0 to 31. At LoFi each bank turn multiplies them and pools the same two faces of product i - 1, which
 * MOVD2A puts in SrcA rows 32 to 63, with the scaler in SrcB rows 32 to 35 of both banks. The pooling runs product by
 * product, faces in order, as in the two-phase row.
 *************************************************************************/

// Two to eight 32x32 tiles.
inline bool _llk_math_mul_reduce_scalar_two_face_banks_(const ckernel::TensorShape tensor_shape, const std::uint32_t num_tiles)
{
    return num_tiles >= 2 && tensor_shape.face_r_dim == ckernel::MAX_FACE_R_DIM && tensor_shape.total_num_faces() == 4;
}

// Multiplies tile i at any fidelity, each bank's fidelity phases in the order the binary MOP runs them per face.
template <MathFidelity math_fidelity>
inline void _llk_math_mul_reduce_scalar_mul_two_face_banks_(const std::uint32_t tile_index)
{
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "mul_two_face_banks: Src zero-substitution flag is not in DEFAULT state");
    constexpr std::uint32_t phases = is_high_fidelity(math_fidelity) ? to_underlying(math_fidelity) : 1;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(tile_index);
#pragma GCC unroll 2
    for (std::uint32_t bank = 0; bank < 2; bank++)
    {
#pragma GCC unroll 4
        for (std::uint32_t phase = 0; phase < phases; phase++)
        {
            TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0);
            TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0);
            TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0, 0);
            if (phase + 1 < phases)
            {
                TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
            }
            else
            {
                TTI_ELWMUL(p_setrwc::CLR_AB, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
            }
        }
    }
    math::clear_dst_reg_addr();
}

// The row's address modes and zero flag, set once by mul_reduce_scalar_init; each row then starts with
// _llk_math_mul_reduce_scalar_interleaved_start_.
template <bool is_fp32_dest_acc_en>
inline void _llk_math_mul_reduce_scalar_interleaved_init_()
{
    _llk_math_mul_reduce_scalar_init_<is_fp32_dest_acc_en, MathFidelity::LoFi>();
    addr_mod_t {.srca = {.incr = 8}, .srcb = {.incr = 8}, .dest = {.incr = 0}}.set(ADDR_MOD_3);
    addr_mod_t {.srca = {.incr = 24}, .srcb = {.incr = 8}, .dest = {.incr = 0}}.set(ADDR_MOD_4);
    addr_mod_t {.srca = {.incr = 32}, .srcb = {.incr = 32}, .dest = {.incr = 0}}.set(ADDR_MOD_5);
}

inline void _llk_math_mul_reduce_scalar_interleaved_start_()
{
    LLK_ASSERT(
        math::src_zero_flag_hw == (requires_disabled_src_zero_flag(math::src_zero_flag_srca_fmt, math::src_zero_flag_srcb_fmt) ? 1u : 0u),
        "interleaved row: Src zero-substitution flag is not in DEFAULT state");
    math::reset_counters(p_setrwc::SET_ABD_F);
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dest_offset_id * DEST_REGISTER_HALF_SIZE);
}

// Multiplies the two faces at DEST row dst. SrcA and SrcB address only their low six bits, so from 64 the four steps
// read rows 0 to 31 and leave SrcA at 48 and SrcB at 32, the second pooled face and the scaler.
template <std::uint32_t dst>
inline void _llk_math_mul_reduce_scalar_mul_two_faces_()
{
    TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, dst);
    TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, dst + 8);
    TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, dst + 16);
    TTI_ELWMUL(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_4, dst + 24);
}

// Fills pool rows 0 to 3 and 8 to 11 with the scaler once product 0's first two faces have left them; only pool row 0
// reaches the result.
inline void _llk_math_mul_reduce_scalar_fill_scaler_(const float scaler)
{
    // DEST is still at the section base the row started from, and immediate rows leave its counter at 0.
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    sfpi::vFloat value = scaler;
    sfpi::dst_reg[0]   = value;
    sfpi::dst_reg[1]   = value;
    sfpi::dst_reg[4]   = value;
    sfpi::dst_reg[5]   = value;
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
}

// Bank turn h of tile i: multiplies faces 2h and 2h + 1 and, from tile 1 on, pools the same faces of product i - 1.
template <std::uint32_t i, std::uint32_t h>
inline void _llk_math_mul_reduce_scalar_interleaved_turn_(const float scaler)
{
    constexpr std::uint32_t cur = 64 * i + 32 * h;
    if constexpr (i > 0)
    {
        constexpr std::uint32_t prev = cur - 64;
        // MATH: until the SETRWC that clears the last bank lands, the valid bits still name that bank.
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
        // The same faces of product i - 1 to SrcA rows 32 to 63.
        constexpr std::uint32_t last_move_addr_mod = i == 1 ? ADDR_MOD_0 : ADDR_MOD_5;
        TTI_MOVD2A(0, 32, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev);
        TTI_MOVD2A(0, 40, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 8);
        TTI_MOVD2A(0, 48, ADDR_MOD_0, p_movd2a::MOV_8_ROWS, prev + 16);
        TTI_MOVD2A(0, 56, last_move_addr_mod, p_movd2a::MOV_8_ROWS, prev + 24);
        if constexpr (i == 1)
        {
            if constexpr (h == 0)
            {
                _llk_math_mul_reduce_scalar_fill_scaler_(scaler);
            }
            // Scaler to SrcB rows 32 to 35; ADDR_MOD_5 takes SrcA and SrcB to the pooled rows as the last move does.
            TTI_MOVD2B(0, 32, ADDR_MOD_5, p_movd2b::MOV_4_ROWS, 8 * h);
            if constexpr (h == 0)
            {
                TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 0);
            }
        }
        // The second pool waits for the first one's DEST write either way, so the multiply goes between them.
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_5, p_gpool::INDEX_DIS, 0);
        _llk_math_mul_reduce_scalar_mul_two_faces_<cur>();
        TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
    }
    else
    {
        _llk_math_mul_reduce_scalar_mul_two_faces_<cur>();
    }
    TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AB);
}

template <std::uint32_t i>
inline void _llk_math_mul_reduce_scalar_interleaved_tile_ct_(const float scaler)
{
    _llk_math_mul_reduce_scalar_interleaved_turn_<i, 0>(scaler);
    _llk_math_mul_reduce_scalar_interleaved_turn_<i, 1>(scaler);
}

template <std::uint32_t... is>
inline void _llk_math_mul_reduce_scalar_interleaved_tiles_(const float scaler, std::integer_sequence<std::uint32_t, is...>)
{
    (_llk_math_mul_reduce_scalar_interleaved_tile_ct_<is>(scaler), ...);
}

// Multiplies the row's tiles and, from tile 1 on, pools product i - 1 in the same bank turns, unrolled so the math
// thread never waits on a per-tile dispatch. Rows of fewer than two tiles do not interleave.
template <std::uint32_t row_tiles>
inline void _llk_math_mul_reduce_scalar_interleaved_row_(const float scaler)
{
    if constexpr (row_tiles >= 2)
    {
        _llk_math_mul_reduce_scalar_interleaved_tiles_(scaler, std::make_integer_sequence<std::uint32_t, row_tiles> {});
    }
}

// Pools the last product in the bank the unpacker zeroed, then collapses the pool row to the scalar.
inline void _llk_math_mul_reduce_scalar_interleaved_tail_(const std::uint32_t num_tiles)
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::MATH | p_stall::SRCA_VLD | p_stall::SRCB_VLD);
    TTI_MOVD2B(0, p_movd2b::SRC_ZERO_OFFSET, ADDR_MOD_0, p_movd2b::MOV_4_ROWS, 8);
    _llk_math_mul_reduce_scalar_move_faces_to_srca_(num_tiles - 1, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_math_mul_reduce_column_<MathFidelity::LoFi, false>(0, ckernel::DEFAULT_TENSOR_SHAPE);
    _llk_math_mul_reduce_scalar_<MathFidelity::LoFi>();
    _llk_math_mul_reduce_scalar_clear_dvalid_();
}
