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
 * @brief Sets up mop config for elementwise binary broadcast operations
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = [ELWADD, ELWSUB, ELWMUL]
 * @tparam BROADCAST_TYPE: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * BROADCAST only operates on SRCB register
 * @tparam MATH_FIDELITY: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when input is Tf32 format
 * @param tensor_shape: Face grid and face row/column dimensions for the operand tile
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, BroadcastType BROADCAST_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_broadcast_mop_config_(const TensorShape& tensor_shape)
{
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");
    const std::uint32_t num_eltwise_instrn_per_face = (tensor_shape.face_r_dim >> math_rows_log2(ELTWISE_MATH_ROWS));

    constexpr auto SRCB_BROADCAST_TYPE = (BROADCAST_TYPE == BroadcastType::COL)
                                             ? p_elwise::SRCB_BCAST_COL
                                             : ((BROADCAST_TYPE == BroadcastType::ROW) ? p_elwise::SRCB_BCAST_ROW : p_elwise::SRCB_BCAST_ALL);

    constexpr std::uint32_t EN_DST_ACC = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    static_assert(!(EN_DST_ACC && ELTWISE_BINARY_TYPE != EltwiseBinaryType::ELWMUL), "Math fidelity larger than LoFi only works with Eltwise MUL");

    const std::uint32_t MOP_OUTER_LOOP = tensor_shape.total_num_faces();
    const std::uint32_t MOP_INNER_LOOP = num_eltwise_instrn_per_face;

    const std::uint32_t eltwise_binary_op = eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_NONE, SRCB_BROADCAST_TYPE, ADDR_MOD_0>(EN_DST_ACC);
    const std::uint32_t eltwise_binary_op_clr_srcAB_valid =
        eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_SRCAB_VLD, SRCB_BROADCAST_TYPE, ADDR_MOD_1>(EN_DST_ACC);

    constexpr std::uint32_t replay_buf_len = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 0 : to_underlying(MATH_FIDELITY_TYPE) - 1;

    if constexpr (EN_DST_ACC)
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            [replay_buf_len, SRCB_BROADCAST_TYPE]
            {
                for (std::uint32_t i = 0; i < replay_buf_len; ++i)
                {
                    TTI_ELWMUL(p_elwise::CLR_NONE, true, SRCB_BROADCAST_TYPE, ADDR_MOD_3, 0);
                }
            });
    }

    /*
    SCALAR -> Unpack only unpacks 1 face: Face 0, SrcB Inc = 0
    ROW -> Unpacker unpacks 4 (default in 32x32 tile) faces: F0, F1, F0, F1, SrcB Inc = 0
    COL -> Unpacker unpacks 4 (default in 32x32 tile) faces: F0, F0, F2, F2, SrcB Inc += ELTWISE_MATH_ROWS
    */
    ckernel_template temp = EN_DST_ACC ? ckernel_template(MOP_OUTER_LOOP, MOP_INNER_LOOP, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), eltwise_binary_op)
                                       : ckernel_template(MOP_OUTER_LOOP, MOP_INNER_LOOP, eltwise_binary_op);

    // Only need to clear per face for ROW/COL, since SCALAR only has 1 face from the unpacker
    if constexpr (BROADCAST_TYPE != BroadcastType::SCALAR)
    {
        constexpr std::uint32_t ADDR_MOD = (BROADCAST_TYPE == BroadcastType::COL) ? ADDR_MOD_2 : ADDR_MOD_0;
        const std::uint32_t eltwise_binary_op_clr_srcB =
            eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_SRCB_VLD, SRCB_BROADCAST_TYPE, ADDR_MOD>(EN_DST_ACC);
        temp.set_last_inner_loop_instr(eltwise_binary_op_clr_srcB);
    }

    temp.set_last_outer_loop_instr(eltwise_binary_op_clr_srcAB_valid);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Sets up addrmods for elementwise binary broadcast operations
 */
template <BroadcastType BROADCAST_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_broadcast_addrmod_()
{
    WAYPOINT("MBA0");
    static_assert((BROADCAST_TYPE != BroadcastType::NONE), "Broadcast type cannot be NONE for this operation");

    // Quasar: addr_mod_t::set() below writes TENSIX_CFG directly from the MATH RISC-V core
    // (volatile MMIO stores), unlike Wormhole where .set() emits SETC16 *instructions* that the
    // backend auto-orders behind in-flight math. If a preceding op (e.g. a row reduce, whose
    // Quasar reduce_uninit is a no-op on the MATH thread) is still draining in the math backend,
    // the config-bus store races that backend over the config path and deadlocks (observed:
    // MATH wedged at the ADDR_MOD_1 .set(), waypoint MBA3). Gate the config writes behind math
    // completion, recovering the ordering WH gets implicitly.
    WAYPOINT("MBAS");
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, p_stall::MATH, p_stall::WAIT_SFPU);
    WAYPOINT("MBAT");

    constexpr std::uint8_t num_srb_rows_inc = (BROADCAST_TYPE == BroadcastType::COL) ? ELTWISE_MATH_ROWS : 0;
    constexpr bool math_fidelity_enable     = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;

    // For ELWADD/SUB/MUL, can increment source and dest registers
    WAYPOINT("MBA1");
    addr_mod_t {
        .srca     = {.incr = ELTWISE_MATH_ROWS},
        .srcb     = {.incr = num_srb_rows_inc},
        .dest     = {.incr = ELTWISE_MATH_ROWS},
        .fidelity = {.incr = 0, .clr = math_fidelity_enable}}
        .set(ADDR_MOD_0);
    WAYPOINT("MBA2");

    // Reset Src counters, inc dest
    WAYPOINT("MBA3");
    addr_mod_t {.srca = {.clr = 1}, .srcb = {.clr = 1}, .dest = {.incr = ELTWISE_MATH_ROWS}, .fidelity = {.incr = 0, .clr = math_fidelity_enable}}.set(
        ADDR_MOD_1);
    WAYPOINT("MBA4");

    if constexpr (BROADCAST_TYPE == BroadcastType::COL)
    {
        // Clear srcB counter for new face, but keep counters for dest & SrcA
        WAYPOINT("MBA5");
        addr_mod_t {
            .srca = {.incr = ELTWISE_MATH_ROWS}, .srcb = {.clr = 1}, .dest = {.incr = ELTWISE_MATH_ROWS}, .fidelity = {.incr = 0, .clr = math_fidelity_enable}}
            .set(ADDR_MOD_2);
        WAYPOINT("MBA6");
    }

    if constexpr (math_fidelity_enable)
    {
        WAYPOINT("MBA7");
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 1, .clr = 0}}.set(ADDR_MOD_3);
        WAYPOINT("MBA8");
    }
    WAYPOINT("MBA9");
}

/**
 * @brief Sets up Initialization for elementwise binary broadcast operation where Output = SrcA [+, -, *] SrcB
 * SrcB either has row, col or scalar datums broadcasted to the rest of the tile before elementwise operation
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
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
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = [ELWADD, ELWSUB, ELWMUL]
 * @tparam BROADCAST_TYPE: Sets the broadcast type, values = [NONE, COL, ROW, SCALAR]
 * @tparam MATH_FIDELITY: 0 = LoFi, 2 = HiFi2, 3 = HiFi3, 4 = HiFi4 - controls precision of multiplication when input is Tf32 format
 * @param tensor_shape: Face grid and face row/column dimensions for the operand tile
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, BroadcastType BROADCAST_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_broadcast_init_(const TensorShape& tensor_shape)
{
    WAYPOINT("MBI0");
    _llk_math_eltwise_binary_broadcast_addrmod_<BROADCAST_TYPE, MATH_FIDELITY_TYPE>();
    WAYPOINT("MBI1");
    _llk_math_eltwise_binary_broadcast_mop_config_<ELTWISE_BINARY_TYPE, BROADCAST_TYPE, MATH_FIDELITY_TYPE>(tensor_shape);
    WAYPOINT("MBI2");

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
    WAYPOINT("MBI3");
}

/**
 * @brief Perform an elementwise binary broadcast operation where Output = SrcA [+, -, *] SrcB
 * SrcB either has row, col or scalar datums broadcasted to the rest of the tile before elementwise operation
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register
 * @param tile_idx: Tile index into the destination register.
 * If dest reg in float16 mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in full mode
 * If dest reg in float32 mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 */
inline void _llk_math_eltwise_binary_broadcast_(const std::uint32_t tile_idx)
{
    WAYPOINT("MBC0");
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);
    WAYPOINT("MBC1");

    // Run MOP
    WAYPOINT("MBC2");
    ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    WAYPOINT("MBC3");

    // Reset all counters
    WAYPOINT("MBC4");
    _reset_counters_<p_setrwc::SET_ABD_F>();
    WAYPOINT("MBC5");
}
