// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"
#include "tensor_shape.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

/**
 * @brief Build the encoded FPU instruction (ELWADD/ELWSUB/ELWMUL) for the given binary op type.
 *
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam CLR_SRC: Source-clear mode applied by the instruction (p_setrwc/p_elwise CLR_* value)
 * @tparam SRCB_BROADCAST_TYPE: SrcB broadcast mode (p_elwise SRCB_* value)
 * @tparam ADDR_MOD: Address-mod slot used by the instruction
 * @param EN_DST_ACC: Enable accumulation into the destination register
 * @return Encoded TT_OP instruction word.
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, std::uint8_t CLR_SRC, std::uint8_t SRCB_BROADCAST_TYPE, std::uint8_t ADDR_MOD>
inline std::uint32_t eltwise_binary_func(std::uint8_t EN_DST_ACC)
{
    if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWADD)
    {
        return TT_OP_ELWADD(CLR_SRC, EN_DST_ACC, SRCB_BROADCAST_TYPE, ADDR_MOD, 0);
    }
    else if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWSUB)
    {
        return TT_OP_ELWSUB(CLR_SRC, EN_DST_ACC, SRCB_BROADCAST_TYPE, ADDR_MOD, 0);
    }
    else
    {
        return TT_OP_ELWMUL(CLR_SRC, EN_DST_ACC, SRCB_BROADCAST_TYPE, ADDR_MOD, 0);
    }
}

//----------------------
// Direct Indexing Method
//----------------------
/**
 * @brief Build the encoded direct-indexing FPU instruction (ELWADDDI/ELWSUBDI/ELWMULDI) for the given binary op type.
 *
 * Direct indexing passes explicit SrcA/SrcB/Dest addresses instead of relying on address-mod increments.
 *
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @param CLR_SRC: Source-clear mode applied by the instruction (p_setrwc/p_elwise CLR_* value)
 * @param EN_DST_ACCUM: Enable accumulation into the destination register
 * @param SRCB_BROADCAST_TYPE: SrcB broadcast mode (p_elwise SRCB_* value)
 * @param SRCB_ADDR: SrcB read address
 * @param SRCA_ADDR: SrcA read address
 * @param ADDR_MOD: Address-mod slot used by the instruction
 * @param DST_ADDR: Destination write address
 * @return Encoded TT instruction word.
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE>
inline std::uint32_t eltwise_di_binary_func(
    std::uint8_t CLR_SRC,
    std::uint8_t EN_DST_ACCUM,
    std::uint8_t SRCB_BROADCAST_TYPE,
    std::uint8_t SRCB_ADDR,
    std::uint8_t SRCA_ADDR,
    std::uint8_t ADDR_MOD,
    std::uint8_t DST_ADDR)
{
    std::uint8_t INSTR_MOD = ((SRCB_BROADCAST_TYPE << 0) | (EN_DST_ACCUM << 2));
    if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWADD)
    {
        return TT_ELWADDDI(CLR_SRC, INSTR_MOD, SRCB_ADDR, SRCA_ADDR, ADDR_MOD, DST_ADDR);
    }
    else if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWSUB)
    {
        return TT_ELWSUBDI(CLR_SRC, INSTR_MOD, SRCB_ADDR, SRCA_ADDR, ADDR_MOD, DST_ADDR);
    }
    else
    {
        return TT_ELWMULDI(CLR_SRC, INSTR_MOD, SRCB_ADDR, SRCA_ADDR, ADDR_MOD, DST_ADDR);
    }
}

//----------------------
/**
 * @brief Sets up mop config for elementwise binary operations.
 *
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam reuse_dest: When not NONE, reuses the destination register as SrcA or SrcB, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc
 * @param acc_to_dest: When true, accumulate the result into the destination register instead of overwriting
 */
template <
    EltwiseBinaryType ELTWISE_BINARY_TYPE,
    ckernel::MathFidelity MATH_FIDELITY_TYPE,
    EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_mop_config_(const ckernel::TensorShape& tensor_shape, bool acc_to_dest = false)
{
    const std::uint32_t rows_per_mop_run =
        (reuse_dest != EltwiseBinaryReuseDestType::NONE) ? tensor_shape.face_r_dim : (tensor_shape.total_num_faces() * tensor_shape.face_r_dim);
    constexpr bool high_fidelity = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    static_assert(!(high_fidelity && ELTWISE_BINARY_TYPE != EltwiseBinaryType::ELWMUL), "Math fidelity larger than LoFi only works with Eltwise MUL");
    // For reuse_dest + Elwmul we need dest accumulation (dest = old_dest + srcA*srcB) ; LoFi alone sets EN_DST_ACC=0.
    const std::uint32_t EN_DST_ACC = acc_to_dest ? 1u : (high_fidelity ? 1u : 0u);

    constexpr std::uint8_t addrmod_fid    = high_fidelity ? ADDR_MOD_2 : ADDR_MOD_0;
    const std::uint32_t eltwise_binary_op = eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_NONE, p_elwise::SRCB_NO_BCAST, addrmod_fid>(EN_DST_ACC);

    const std::uint32_t MOP_OUTER_LOOP     = (rows_per_mop_run >> rows_log2(ELTWISE_MATH_ROWS));
    constexpr std::uint32_t MOP_INNER_LOOP = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);

    const std::uint32_t eltwise_binary_op_clr_valid =
        eltwise_binary_func<ELTWISE_BINARY_TYPE, p_setrwc::CLR_AB, p_elwise::SRCB_NO_BCAST, ADDR_MOD_1>(EN_DST_ACC);
    ckernel_template temp(MOP_OUTER_LOOP, MOP_INNER_LOOP, eltwise_binary_op);
    temp.set_last_outer_loop_instr(eltwise_binary_op_clr_valid);

    if (high_fidelity)
    {
        const std::uint32_t eltwise_binary_op_clr_fidelity =
            eltwise_binary_func<ELTWISE_BINARY_TYPE, p_elwise::CLR_NONE, p_elwise::SRCB_NO_BCAST, ADDR_MOD_0>(EN_DST_ACC);
        temp.set_last_inner_loop_instr(eltwise_binary_op_clr_fidelity); // clear math fidelity
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

//----------------------
// Direct Indexing Method
//----------------------
/**
 * @brief Sets up mop config for elementwise binary operations using the direct-indexing instruction variant.
 *
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc
 * @param acc_to_dest: When true, accumulate the result into the destination register instead of overwriting
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_di_binary_mop_config_(const ckernel::TensorShape& tensor_shape, bool acc_to_dest = false)
{
    const std::uint32_t total_num_rows_per_tile = tensor_shape.total_num_faces() * tensor_shape.face_r_dim;
    const std::uint32_t REPLAY_BUF_LEN          = (total_num_rows_per_tile >> rows_log2(ELTWISE_MATH_ROWS));
    const std::uint32_t MOP_INNER_LOOP          = to_underlying(MATH_FIDELITY_TYPE) + 1;
    constexpr bool high_fidelity                = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    static_assert(!(high_fidelity && ELTWISE_BINARY_TYPE != EltwiseBinaryType::ELWMUL), "Math fidelity larger than LoFi only works with Eltwise MUL");
    const std::uint32_t EN_DST_ACC = acc_to_dest ? 1u : static_cast<std::uint32_t>(high_fidelity);

    load_replay_buf(
        0u,
        REPLAY_BUF_LEN,
        false,
        0,
        0,
        [&]()
        {
            for (std::uint32_t i = 0; i < REPLAY_BUF_LEN - 1; ++i)
            {
                eltwise_di_binary_func<ELTWISE_BINARY_TYPE>(
                    p_elwise::CLR_NONE,
                    EN_DST_ACC,
                    p_elwise::SRCB_NO_BCAST,
                    (i * ELTWISE_MATH_ROWS) >> 2, // Srcb addr
                    (i * ELTWISE_MATH_ROWS) >> 2, // Srca addr
                    0x0,
                    (i * ELTWISE_MATH_ROWS) >> 2); // Dest addr
            }

            if (high_fidelity)
            {
                eltwise_di_binary_func<ELTWISE_BINARY_TYPE>(
                    p_elwise::CLR_NONE,
                    EN_DST_ACC,
                    p_elwise::SRCB_NO_BCAST,
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2,  // Srcb addr
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2,  // Srca addr
                    ADDR_MOD_1,                                       // Increment Fidelity
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2); // Dest addr
            }
            else
            {
                eltwise_di_binary_func<ELTWISE_BINARY_TYPE>(
                    p_setrwc::CLR_AB,
                    EN_DST_ACC,
                    p_elwise::SRCB_NO_BCAST,
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2,  // Srcb addr
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2,  // Srca addr
                    ADDR_MOD_1,                                       // Increment Fidelity
                    ((REPLAY_BUF_LEN - 1) * ELTWISE_MATH_ROWS) >> 2); // Dest addr
            }
        });

    ckernel_template temp(1 /* outer loop */, MOP_INNER_LOOP, TT_OP_REPLAY(0, REPLAY_BUF_LEN, 0, 0, 0, 0));

    if (high_fidelity)
    {
        temp.set_end_op(TT_OP_SETRWC(p_setrwc::CLR_AB, 0, 0, p_setrwc::SET_ABD_F));
    }

    temp.program_bank0_sw_cntl(instrn_buffer);
}

//----------------------

/**
 * @brief Sets up addrmods for elementwise binary operations.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_binary_addrmod_()
{
    constexpr bool high_fidelity = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;

    // For ELWADD/SUB/MUL, can increment source
    //  and dest registers
    addr_mod_t {
        .srca     = {.incr = ELTWISE_MATH_ROWS},
        .srcb     = {.incr = ELTWISE_MATH_ROWS},
        .dest     = {.incr = ELTWISE_MATH_ROWS},
        .fidelity = {.incr = 0, .clr = high_fidelity}}
        .set(ADDR_MOD_0);

    // Reset Src counters, inc dest
    addr_mod_t {.srca = {.clr = 1}, .srcb = {.clr = 1}, .dest = {.incr = ELTWISE_MATH_ROWS}, .fidelity = {.incr = 0, .clr = high_fidelity}}.set(ADDR_MOD_1);

    if constexpr (high_fidelity)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 1}}.set(ADDR_MOD_2);
    }
}

//----------------------
// Direct Indexing Method
//----------------------
/**
 * @brief Sets up addrmods for elementwise binary operations using the direct-indexing instruction variant.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_eltwise_di_binary_addrmod_()
{
    constexpr bool high_fidelity               = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr std::uint32_t fidelity_increment = high_fidelity ? 1 : 0;
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = fidelity_increment, .clr = 0},
    }
        .set(ADDR_MOD_1);
}

//----------------------
/**
 * @brief Initialize FPU to perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 *
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register.
 *
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam reuse_dest: When not NONE, reuses the destination register as SrcA or SrcB, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @tparam ENABLE_DIRECT_INDEXING: Enable the direct-indexing instruction variant
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc
 * @param acc_to_dest: When true, accumulate the result into the destination register instead of overwriting
 * @note On the unpack thread (T0): for reuse_dest == NONE pair with @ref _llk_unpack_binary_operands_init_; for DEST_TO_SRCA/DEST_TO_SRCB pair with
 *       @ref _llk_unpack_unary_operand_init_ (the dummy-dvalid path that lets MOVD2A/B fill the reused source register). On the pack thread, pair with
 *       @ref _llk_pack_init_ (T2).
 * @note @ref _llk_math_eltwise_binary_ runs the configured op with matching template args.
 */
template <
    EltwiseBinaryType ELTWISE_BINARY_TYPE,
    ckernel::MathFidelity MATH_FIDELITY_TYPE,
    EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool ENABLE_DIRECT_INDEXING           = false>
inline void _llk_math_eltwise_binary_init_(const ckernel::TensorShape& tensor_shape, bool acc_to_dest = false)
{
    if constexpr (ENABLE_DIRECT_INDEXING)
    {
        _llk_math_eltwise_di_binary_addrmod_<MATH_FIDELITY_TYPE>();
        _llk_math_eltwise_di_binary_mop_config_<ELTWISE_BINARY_TYPE, MATH_FIDELITY_TYPE>(tensor_shape, acc_to_dest);
    }
    else
    {
        _llk_math_eltwise_binary_addrmod_<MATH_FIDELITY_TYPE>();
        _llk_math_eltwise_binary_mop_config_<ELTWISE_BINARY_TYPE, MATH_FIDELITY_TYPE, reuse_dest>(tensor_shape, acc_to_dest);
    }

    if constexpr (reuse_dest != EltwiseBinaryReuseDestType::NONE)
    {
        addr_mod_t {}.set(ADDR_MOD_3);
    }

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Perform an elementwise binary operation where Output = SrcA [+, -, *] SrcB.
 *
 * SrcA/SrcB contain 1 tile each, and output is 1 tile in destination register.
 *
 * @tparam ELTWISE_BINARY_TYPE: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam reuse_dest: When not NONE, reuses the destination register as SrcA or SrcB. The MOVD2A/B instruction copies a face from dest to the source register
 * before each MOP run, values = <NONE/DEST_TO_SRCA/DEST_TO_SRCB>
 * @param tile_idx: Tile index into the destination register. If dest reg in 16-bit mode -> values = [0 - 8] in double buffering mode, values = [0 - 16] in
 * full mode. If dest reg in 32-bit mode -> values = [0 - 4] in double buffering mode, values = [0 - 8] in full mode
 * @param tensor_shape: Contains all the information of the tensor shape: num faces, face row/col dim, etc
 * @param clear_in_fp32_mode: When true, clears the dest face in Float32 mode during dest reuse
 * @note Call @ref _llk_math_eltwise_binary_init_ with matching template args before this function.
 */
template <EltwiseBinaryType ELTWISE_BINARY_TYPE, EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void _llk_math_eltwise_binary_(const std::uint32_t tile_idx, const ckernel::TensorShape& tensor_shape, const bool clear_in_fp32_mode = false)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(tile_idx);

    if constexpr (reuse_dest != EltwiseBinaryReuseDestType::NONE)
    {
        [[maybe_unused]] auto tile_start = tile_idx * tensor_shape.total_num_faces();
        for (std::uint32_t face_num = 0; face_num < tensor_shape.total_num_faces(); face_num++)
        {
            eltwise_binary_reuse_dest_as_src<reuse_dest>();
            if constexpr (ELTWISE_BINARY_TYPE == EltwiseBinaryType::ELWMUL)
            {
                // ELWMUL needs HiFi (therefore dest_acc). Clear dest face-by-face when reusing dest as srcA/B
                TT_ZEROACC(p_zeroacc::CLR_16, clear_in_fp32_mode, 0, ADDR_MOD_3, tile_start + face_num);
            }
            ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        }
    }
    else
    {
        // Run MOP for the entire tile
        ckernel::ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    }

    // Reset all counters
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
