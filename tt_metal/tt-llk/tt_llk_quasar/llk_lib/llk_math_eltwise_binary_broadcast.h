// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TODO: Plumb MATH_FIDELITY
#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#include "tensor_shape.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @brief Sets up mop config for elementwise binary broadcast operations.
 *
 * Broadcast only operates on the SrcB register.
 *
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam BROADCAST_TYPE: Sets the broadcast type (must not be NONE for this op), values = <COL/ROW/SCALAR>
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Face grid and face row/column dimensions for the operand tile
 * @param acc_to_dest: When true, accumulate into dest even at LoFi. HiFi partial products still accumulate; without this flag the first phase overwrites.
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, BroadcastType BROADCAST_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_broadcast_mop_config_(const TensorShape& tensor_shape, bool acc_to_dest = false)
{
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");
    // A face shorter than one FPU instruction still needs that instruction, or the
    // inner loop is 0 and math never clears the Src dvalids unpack raised.
    const std::uint32_t num_eltwise_instrn_per_face =
        (tensor_shape.face_r_dim < ELTWISE_MATH_ROWS) ? 1u : (tensor_shape.face_r_dim >> rows_log2(ELTWISE_MATH_ROWS));

    constexpr auto SRCB_BROADCAST_TYPE = (BROADCAST_TYPE == BroadcastType::COL)
                                             ? p_elwise::SRCB_BCAST_COL
                                             : ((BROADCAST_TYPE == BroadcastType::ROW) ? p_elwise::SRCB_BCAST_ROW : p_elwise::SRCB_BCAST_ALL);

    constexpr bool high_fidelity = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    static_assert(!(high_fidelity && ELTWISE_BINARY_TYPE != EltwiseBinaryType::ELWMUL), "Math fidelity larger than LoFi only works with Eltwise MUL");
    // The address-advancing op is a later fidelity phase when HiFi, so it accumulates.
    // LoFi accumulates only when the caller asked to fold tiles into dest.
    const std::uint32_t advancing_acc = (high_fidelity || acc_to_dest) ? 1u : 0u;

    const std::uint32_t MOP_OUTER_LOOP = tensor_shape.total_num_faces();
    const std::uint32_t MOP_INNER_LOOP = num_eltwise_instrn_per_face;

    const std::uint32_t eltwise_binary_op = eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_NONE, SRCB_BROADCAST_TYPE, ADDR_MOD_0>(advancing_acc);
    const std::uint32_t eltwise_binary_op_clr_srcAB_valid =
        eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_SRCAB_VLD, SRCB_BROADCAST_TYPE, ADDR_MOD_1>(advancing_acc);

    constexpr std::uint32_t replay_buf_len = high_fidelity ? to_underlying(MATH_FIDELITY_TYPE) - 1 : 0;

    if constexpr (high_fidelity)
    {
        load_replay_buf<0, replay_buf_len>(
            [replay_buf_len, SRCB_BROADCAST_TYPE, acc_to_dest]
            {
                for (std::uint32_t i = 0; i < replay_buf_len; ++i)
                {
                    // First phase overwrites unless this tile folds into the previous dest result.
                    // TTI encodings must be immediates, so the accumulate bit is a literal in each branch.
                    if (acc_to_dest || i > 0)
                    {
                        TTI_ELWMUL(p_elwise::CLR_NONE, 1, SRCB_BROADCAST_TYPE, ADDR_MOD_3, 0);
                    }
                    else
                    {
                        TTI_ELWMUL(p_elwise::CLR_NONE, 0, SRCB_BROADCAST_TYPE, ADDR_MOD_3, 0);
                    }
                }
            });
    }

    /*
    SCALAR -> Unpack only unpacks 1 face: Face 0, SrcB Inc = 0
    ROW -> Unpacker unpacks 4 (default in 32x32 tile) faces: F0, F1, F0, F1, SrcB Inc = 0
    COL -> Unpacker unpacks 4 (default in 32x32 tile) faces: F0, F0, F2, F2, SrcB Inc += ELTWISE_MATH_ROWS
    */
    ckernel_template temp = high_fidelity ? ckernel_template(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), eltwise_binary_op)
                                          : ckernel_template(MOP_OUTER_LOOP, MOP_INNER_LOOP, eltwise_binary_op);

    // Only need to clear per face for ROW/COL, since SCALAR only has 1 face from the unpacker
    if constexpr (BROADCAST_TYPE != BroadcastType::SCALAR)
    {
        // A face that fits in one FPU instruction never advanced SrcB. Clearing the SrcB
        // counter on that only instruction samples the column before the face is visible.
        // Longer faces keep the counter clear programmed in the address-mod setup.
        if constexpr (BROADCAST_TYPE == BroadcastType::COL)
        {
            if (num_eltwise_instrn_per_face == 1u)
            {
                addr_mod_t {
                    .srca     = {.incr = ELTWISE_MATH_ROWS},
                    .srcb     = {.incr = 0},
                    .dest     = {.incr = ELTWISE_MATH_ROWS},
                    .fidelity = {.incr = 0, .clr = high_fidelity}}
                    .set(ADDR_MOD_2);
            }
            const std::uint32_t eltwise_binary_op_clr_srcB =
                eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_SRCB_VLD, SRCB_BROADCAST_TYPE, ADDR_MOD_2>(advancing_acc);
            temp.set_last_inner_loop_instr(eltwise_binary_op_clr_srcB);
        }
        else
        {
            const std::uint32_t eltwise_binary_op_clr_srcB =
                eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_SRCB_VLD, SRCB_BROADCAST_TYPE, ADDR_MOD_0>(advancing_acc);
            temp.set_last_inner_loop_instr(eltwise_binary_op_clr_srcB);
        }
    }

    temp.set_last_outer_loop_instr(eltwise_binary_op_clr_srcAB_valid);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods for elementwise binary broadcast operations.
 *
 * @tparam BROADCAST_TYPE: Sets the broadcast type (must not be NONE for this op), values = <COL/ROW/SCALAR>
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 */
template <BroadcastType BROADCAST_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_broadcast_addrmod_()
{
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");

    constexpr std::uint8_t num_srb_rows_inc = (BROADCAST_TYPE == BroadcastType::COL) ? ELTWISE_MATH_ROWS : 0;
    constexpr bool math_fidelity_enable     = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;

    // For ELWADD/SUB/MUL, can increment source and dest registers
    addr_mod_t {
        .srca     = {.incr = ELTWISE_MATH_ROWS},
        .srcb     = {.incr = num_srb_rows_inc},
        .dest     = {.incr = ELTWISE_MATH_ROWS},
        .fidelity = {.incr = 0, .clr = math_fidelity_enable}}
        .set(ADDR_MOD_0);

    // Reset Src counters, inc dest
    addr_mod_t {.srca = {.clr = 1}, .srcb = {.clr = 1}, .dest = {.incr = ELTWISE_MATH_ROWS}, .fidelity = {.incr = 0, .clr = math_fidelity_enable}}.set(
        ADDR_MOD_1);

    if constexpr (BROADCAST_TYPE == BroadcastType::COL)
    {
        // Clear srcB counter for new face, but keep counters for dest & SrcA
        addr_mod_t {
            .srca = {.incr = ELTWISE_MATH_ROWS}, .srcb = {.clr = 1}, .dest = {.incr = ELTWISE_MATH_ROWS}, .fidelity = {.incr = 0, .clr = math_fidelity_enable}}
            .set(ADDR_MOD_2);
    }

    if constexpr (math_fidelity_enable)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 1, .clr = 0}}.set(ADDR_MOD_3);
    }
}

/**
 * @brief Sets up initialization for elementwise binary broadcast operation where Output = SrcA [+, -, *] SrcB.
 *
 * SrcB either has row, col or scalar datums broadcasted to the rest of the tile before elementwise operation.
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register.
 *
 * In a 32 x 32 tile, faces layout would be the following:
 * --------------------
 * Face 0    | Face 1
 * --------------------
 * Face 2    | Face 3
 * --------------------
 * For SCALAR broadcast -> first datum of SrcB tile (datum[0] of SrcB face0)
 * will be used for all the datums of the eltwise binary operation. Result = SrcA [+,-,*] datum[0] of SrcB register
 *
 * For ROW broadcast -> first row of SrcB tile (datums[0:16] of SrcB face0 and face1)
 * will be used broadcasted to the rest of the rows of srcB register.
 * Result face 0 = face 0 SrcA [+,-,*] datums[0:16] of face 0 SrcB register
 * Result face 1 = face 1 SrcA [+,-,*] datums[0:16] of face 1 SrcB register
 * Result face 2 = face 2 SrcA [+,-,*] datums[0:16] of face 0 SrcB register
 * Result face 3 = face 3 SrcA [+,-,*] datums[0:16] of face 1 SrcB register
 *
 ** For COL broadcast -> first column of SrcB tile (datums[0, 16, 32, 48, ...240] of SrcB face0 and face2)
 * will be used broadcasted to the rest of the columns of srcB register.
 * Result face 0 = face 0 SrcA [+,-,*] datums[0, 16, 32, 48, ...240] of face 0 SrcB register
 * Result face 1 = face 1 SrcA [+,-,*] datums[0, 16, 32, 48, ...240] of face 0 SrcB register
 * Result face 2 = face 2 SrcA [+,-,*] datums[0, 16, 32, 48, ...240] of face 2 SrcB register
 * Result face 3 = face 3 SrcA [+,-,*] datums[0, 16, 32, 48, ...240] of face 2 SrcB register
 *
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam BROADCAST_TYPE: Sets the broadcast type (must not be NONE for this op), values = <COL/ROW/SCALAR>
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 *values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Face grid and face row/column dimensions for the operand tile
 * @param acc_to_dest: When true, accumulate into dest even at LoFi. HiFi partial products still accumulate; without this flag the first phase overwrites.
 * @note On the unpack thread, pair with @ref _llk_unpack_binary_broadcast_operands_init_ (T0) with matching BROADCAST_TYPE; on the pack thread, pair with
 *       @ref _llk_pack_init_ (T2).
 * @note @ref _llk_math_eltwise_binary_broadcast_ runs the configured op with matching template args.
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, BroadcastType BROADCAST_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_broadcast_init_(const TensorShape& tensor_shape, bool acc_to_dest = false)
{
    _llk_math_eltwise_binary_broadcast_addrmod_<BROADCAST_TYPE, MATH_FIDELITY_TYPE>();
    _llk_math_eltwise_binary_broadcast_mop_config_<ELTWISE_BINARY_TYPE, BROADCAST_TYPE, MATH_FIDELITY_TYPE>(tensor_shape, acc_to_dest);

    // Each dest tile uses its total face rows, but takes at least one full face.
    _set_tile_shape_idx_gpr_(find_max(FACE_R_DIM, tensor_shape.face_r_dim * tensor_shape.total_num_faces()));

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Perform an elementwise binary broadcast operation where Output = SrcA [+, -, *] SrcB.
 *
 * SrcB either has row, col or scalar datums broadcasted to the rest of the tile before elementwise operation.
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register.
 *
 * @param tile_idx: Tile index into the destination register. If dest reg in 16-bit mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in
 * full mode. If dest reg in 32-bit mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 * @note Call @ref _llk_math_eltwise_binary_broadcast_init_ with matching template args before this function.
 */
inline void _llk_math_eltwise_binary_broadcast_(const std::uint32_t tile_idx)
{
    _set_dst_write_addr_by_rows_(tile_idx);

    // Run MOP
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
