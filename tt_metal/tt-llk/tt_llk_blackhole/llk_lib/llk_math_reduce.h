// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"
#include "llk_math_common.h"
#include "lltt.h"
#include "tensor_shape.h"

using namespace ckernel;

// local function declarations
inline void reduce_configure_addrmod();

template <ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop();

template <bool enforce_fp32_accumulation, bool is_int_fpu_en>
inline void reduce_row_perform_transpose()
{
    if (enforce_fp32_accumulation)
    {
        // BH Issue #449 W/A: SFPU-staged 2-pass transpose (v4: Phase 1 lo-only).
        // Phase 1: SFPU extracts lo16 only to LO16_STAGE (16 ops; was 24 in v3).
        //   Hi16 does NOT need staging — Phase 2 hi reads hi16 directly from the fp32 face
        //   via MOVD2B(Fp32=0) with dst_32bit_addr_en=1 (routes read_dst16b through
        //   read_dst32b(adj_row) >> 16 → pure hi16 regardless of tile offset D).
        // Phase 2 hi: dst_32bit_addr_en=1, MOVD2B reads hi16 from face (dst=0),
        //   TRNSPSRCB, MOVB2D writes transposed hi16 back to face via adj_row (lo16=0 as
        //   side effect of write_dst32b(adj_row, data<<16)). No HI16_STAGE scratch.
        // Phase 2 lo: dst_32bit_addr_en=0, unchanged — transposes LO16_STAGE scratch.
        // Phase 3: SFPLOAD INT32 from face (reads face hi16+0 via adj_row via read_dst32b,
        //   no dst_32bit_addr_en dependency) + SFPLOAD LO16_ONLY from LO16_STAGE
        //   (preserves LReg hi16) + SFPSTORE INT32 = 24 ops (unchanged count).
        // Total SFPU: 40 ops (-8 vs 48 in v3). Uses only dest_32b_lo=0 in MOV* (avoids
        // BH Issue #449 HW bug, which affects dest_32b_lo=1).
        constexpr std::uint32_t LO16_STAGE = 144;

        // v6: cfg_rmw(Fp32=0) dropped. The BH HW bug (#449) only affects MOV* with
        // dest_32b_lo=1. Since we only use dest_32b_lo=0 (DEST_NORM), keeping Fp32=1
        // is safe. dst_32bit_addr_en=1 toggle around Phase 2 hi is still needed to
        // make ttsim's MOVD2B/MOVB2D route through Adj32 (ttsim models Adj16=identity
        // in Fp32=0 mode, but our MOV*s run with Fp32 unchanged which may differ).

        // Phase 1: SFPU extracts lo16 only (hi16 read directly from face by Phase 2 hi).
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
        _llk_math_dbg_feature_disable_(); // dst_32bit_addr_en = 1
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
#pragma GCC unroll 4
        for (std::uint32_t g = 0; g < 4; g++)
        {
            const std::uint32_t src_row = g * 4;
            const std::uint32_t lo_row  = LO16_STAGE + g * 4;
#pragma GCC unroll 2
            for (std::uint32_t parity = 0; parity < 4; parity += 2)
            {
                TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, src_row + parity);
                TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::LO16_ONLY, ADDR_MOD_0, lo_row + parity);
            }
        }

        // Phase 2 hi: enable dst_32bit_addr_en=1 so MOVD2B/MOVB2D route through
        // read_dst32b/write_dst32b(adj_row). Reads hi16 from face (dst=0), transposes,
        // writes transposed hi16 back to face. (Face lo16 gets zeroed as side effect.)

        TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        TTI_TRNSPSRCB;
        TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 0);
        TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 4);
        TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 8);
        TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, 12);

        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
        // Restore dst_32bit_addr_en=0 before Phase 2 lo (which needs direct physical
        // addressing for LO16_STAGE scratch access).
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
        _llk_math_dbg_feature_enable_(); // dst_32bit_addr_en = 0
        TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);

        // Phase 2 lo: unchanged — transposes LO16_STAGE scratch in place.
        TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, LO16_STAGE);
        TTI_TRNSPSRCB;
        TTI_MOVD2B(p_mov::DEST_NORM, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, LO16_STAGE);
        TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, LO16_STAGE);
        TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 4, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, LO16_STAGE + 4);
        TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 8, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, LO16_STAGE + 8);
        TTI_MOVB2D(p_mov::DEST_NORM, p_movb2d::SRC_ROW16_OFFSET + 12, ADDR_MOD_0, p_movb2d::MOV_4_ROWS, LO16_STAGE + 12);

        // Phase 3: write transposed lo16 to face lo16 rows via SFPSTORE HI16_ONLY.
        // Two recorded 2-op templates (slot 0: dst_base=8; slot 2: dst_base=16), each
        // replayed 4×. ADDR_MOD_7 on SFPSTORE advances SFPU DstRWC by 2 per iter.
        // Slot 0 covers dst={8,10,12,14} with DstRWC 0..6; slot 2 covers dst={24,26,28,30}
        // with DstRWC 8..14. Src base=LO16_STAGE gives src=144..158.
        // 8 replay triggers vs previous 16 inline ops.
        TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
#pragma GCC unroll 4
        for (std::uint32_t i = 0; i < 4; i++)
        {
            lltt::replay(0, 2);
        }
#pragma GCC unroll 4
        for (std::uint32_t i = 0; i < 4; i++)
        {
            lltt::replay(2, 2);
        }
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
    }
    else
    {
        // Datums stored in int32 dest cannot be moved to SrcB which is configured for int8 inputs
        // Cast int32 datums to int8 using SFPU instructions (load int32, store int8) before moving data to srcB
        // Besides SFPU instructions to do cast we also need to set chicken bit FP16A_FORCE_Enable to force dest
        // view to be fp16a as int8 datums are stored in src registers as fp16a
        if constexpr (is_int_fpu_en)
        {
            TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 0 /*DEST offset*/);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT8, ADDR_MOD_0, 0 /*DEST offset*/);
            TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_0, 2 /*DEST offset*/);
            TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::INT8, ADDR_MOD_0, 2 /*DEST offset*/);
            TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_SFPU);
            TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x1);
        }

        // Move back to B and transpose
        // we avoid clobbering weights in src B by moving to rows 16 - 31
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);
        /*
        if constexpr (is_fp32_dest_acc_en) {
            if (0 == (((std::uint32_t)unpack_dst_format[0]>>2)&0x1)) { // fp32 to fp16_a conversion
                TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
                TTI_SFPLOAD(0, 0, 3, 0);
                TTI_SFP_STOCH_RND(0,0,0,0,0,8);
                TTI_SFPSTORE(0,1,3,0);
            }
        }
        */
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        // Note: transpose on src B on works on rows 16 - 31
        TTI_TRNSPSRCB;
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 0);
        if constexpr (is_int_fpu_en)
        {
            TTI_SETC16(FP16A_FORCE_Enable_ADDR32, 0x0);
        }

        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_B, 0, 8, 0, p_setrwc::SET_B);
        TTI_ZEROSRC(0, 1, 0, 1); // Clear src A
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
        TTI_ELWADD(0, 0, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
    }
}

template <PoolType type, bool high_fidelity, std::uint32_t clear_mode, std::uint32_t index = 0>
inline void reduce_pool_op()
{
    // Transpose for each face in src A done at unpacker, and pool
    if constexpr (type == PoolType::MAX)
    {
        TTI_GMPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, index);
    }
    else if constexpr (high_fidelity)
    {
        ckernel_template::run();
        if constexpr (clear_mode != p_setrwc::CLR_NONE)
        {
            TTI_CLEARDVALID(clear_mode, 0);
        }
    }
    else
    {
        TTI_GAPOOL(clear_mode, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, index);
    }
}

template <
    PoolType type,
    ReduceDim dim,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity,
    bool is_int_fpu_en             = false,
    bool enforce_fp32_accumulation = false>
inline void _llk_math_reduce_(const std::uint32_t dst_index, const ckernel::TensorShape& tensor_shape)
{
    LLK_ASSERT(validate_tensor_shape_tile_dependent_ops_(tensor_shape), "Invalid tensor shape for tile-dependent op");

    // Supported narrow tiles per BH Tiny Tile Summary: [16]x16 (num_faces=1) and [32]x16 (num_faces=2) only
    LLK_ASSERT(
        !((tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim) /* narrow_tile */ && tensor_shape.total_num_faces() == 4),
        "Reduce narrow tile requires num_faces 1 or 2; num_faces=4 is full-width 32x32");

    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    if constexpr (dim == ReduceDim::REDUCE_ROW)
    {
        // Reduce all faces in a row and perform transpose
        for (std::uint32_t col_num = 0; col_num < static_cast<std::uint32_t>(tensor_shape.num_faces_c_dim - 1); col_num++)
        {
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 0>();
        }
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
        reduce_row_perform_transpose<enforce_fp32_accumulation, is_int_fpu_en>();

        // If there is only 1 row of faces, then we are done
        if (tensor_shape.num_faces_r_dim > 1)
        {
            // Increment dest by 32 or 16 if narrow tile for the next accumulation
            if (!(tensor_shape.num_faces_c_dim < tensor_shape.num_faces_r_dim) /* narrow_tile */)
            {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            }
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_BD);

            // Reduce all faces in a row and perform transpose
            for (std::uint32_t col_num = 0; col_num < static_cast<std::uint32_t>(tensor_shape.num_faces_c_dim - 1); col_num++)
            {
                reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 0>();
            }
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            reduce_row_perform_transpose<enforce_fp32_accumulation, is_int_fpu_en>();
        }

        TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_BD);
    }
    else if constexpr (dim == ReduceDim::REDUCE_COL)
    {
        for (std::uint32_t row_num = 0; row_num < tensor_shape.num_faces_r_dim; row_num++)
        {
            // Just pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            if (tensor_shape.num_faces_c_dim > 1)
            {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
                TTI_SETRWC(p_setrwc::CLR_AB, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);

                reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 0>();
            }
            // Reset Dest Counter
            TTI_SETRWC(p_setrwc::CLR_AB, 0, 0, 0, 0, p_setrwc::SET_AD);
        }
    }
    else if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        for (std::uint32_t face_num = 0; face_num < static_cast<std::uint32_t>(tensor_shape.total_num_faces() - 1); face_num++)
        {
            // Wait and pool
            reduce_pool_op<type, high_fidelity, p_setrwc::CLR_AB, 4>();
        }
        // Wait and pool
        reduce_pool_op<type, high_fidelity, p_setrwc::CLR_NONE, 4>();

        // Need row in dest as column in src A
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 0, 0, 0, p_setrwc::SET_AB);

        // copy over from dest to B and do transpose
        // use rows 16 - 31 in src B as scratch
        TTI_MOVD2B(0, p_movd2b::SRC_ROW16_OFFSET, ADDR_MOD_0, p_movd2b::MOV_1_ROW, 4);
        TTI_GATESRCRST(0b1, 0b1);
        TTI_TRNSPSRCB;
        // gate math instructions until src B has been updated
        TTI_GATESRCRST(0b1, 0b1);
        // copy over all 16 rows from B to A
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 0, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 0);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 4, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 4);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 8, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 8);
        TTI_MOVB2A(p_movb2a::SRCA_ZERO_OFFSET + 12, ADDR_MOD_0, p_movb2a::MOV_4_ROWS, p_movb2a::SRCB_ROW16_OFFSET + 12);
        // gate math instructions until src A has been updated by MOV instructions
        TTI_GATESRCRST(0b1, 0b1);
        // zero out scratch in dest
        TTI_ZEROACC(p_zeroacc::CLR_SPECIFIC, 0, 0, ADDR_MOD_0, 4);

        if constexpr (type == PoolType::MAX)
        {
            TTI_GMPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
        else
        {
            if constexpr (high_fidelity)
            {
                for (std::uint32_t i = 0; i < to_underlying(math_fidelity) - 1; i++)
                {
                    TTI_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0);
                }
            }
            TTI_GAPOOL(p_setrwc::CLR_AB, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0);
        }
    }
}

template <PoolType type, MathFidelity math_fidelity>
inline void reduce_configure_addrmod()
{
    constexpr bool high_fidelity               = is_high_fidelity(math_fidelity);
    constexpr std::uint32_t fidelity_increment = high_fidelity ? 1 : 0;

    addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = 0, .clr = 1}}.set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 1},
        .dest = {.incr = 1},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 8},
        .dest = {.incr = 8},
    }
        .set(ADDR_MOD_2);

    if constexpr (high_fidelity)
    {
        addr_mod_t {.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}, .fidelity = {.incr = fidelity_increment}}.set(ADDR_MOD_3);
    }
}

template <ReduceDim dim, MathFidelity math_fidelity>
inline void reduce_configure_mop()
{
    constexpr int inner_loop_len = to_underlying(math_fidelity); // inner loop length is the number of fidelity phases 0, 2, 3, 4 (LoFi, Hifi2, Hifi3, Hifi4)

    if constexpr (dim == ReduceDim::REDUCE_SCALAR)
    {
        ckernel_template tmp(1, inner_loop_len, TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 4));
        tmp.set_last_inner_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4));
        tmp.set_last_outer_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 4));
        tmp.program();
    }
    else
    {
        ckernel_template tmp(1, inner_loop_len, TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_3, p_gpool::INDEX_DIS, 0));
        tmp.set_last_inner_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
        tmp.set_last_outer_loop_instr(TT_OP_GAPOOL(p_setrwc::CLR_NONE, p_gpool::DIM_16X16, ADDR_MOD_0, p_gpool::INDEX_DIS, 0));
        tmp.program();
    }
}

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, MathFidelity math_fidelity, bool enforce_fp32_accumulation = false>
inline void _llk_math_reduce_init_()
{
    constexpr bool high_fidelity = is_high_fidelity(math_fidelity);

    reduce_configure_addrmod<type, math_fidelity>();
    if constexpr (high_fidelity)
    {
        reduce_configure_mop<dim, math_fidelity>();
    }

    if constexpr (enforce_fp32_accumulation)
    {
        static_assert(is_fp32_dest_acc_en, "FP32 Dest must be enabled for FP32 accumulation");
        // Phase 3 replay: advance SFPU DstRWC by 2 per iter via ADDR_MOD_7 on SFPSTORE.
        // SFPLOAD uses ADDR_MOD_0 (no advance) — both ops in an iter share the same DstRWC.
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ADDR_MOD_7);

        // Record two 2-op templates for Phase 3:
        //   Slot 0: dst base = 8, covers dst={8,10,12,14} (DstRWC 0..6)
        //   Slot 2: dst base = 16, covers dst={24,26,28,30} (DstRWC 8..14)
        // SFPLOAD base=LO16_STAGE, DstRWC=0..14 gives src=144..158.
        constexpr std::uint32_t LO16_STAGE_REPLAY = 144;
        lltt::record<lltt::NoExec>(0, 2);
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::HI16_ONLY, ADDR_MOD_0, LO16_STAGE_REPLAY);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, 8);

        lltt::record<lltt::NoExec>(2, 2);
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::HI16_ONLY, ADDR_MOD_0, LO16_STAGE_REPLAY);
        TT_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::HI16_ONLY, ADDR_MOD_7, 16);
    }
    TTI_SETC16(CLR_DVALID_SrcA_Disable_ADDR32, 0);

    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool enforce_fp32_accumulation = false>
inline void _llk_math_reduce_uninit_()
{
    if constexpr (enforce_fp32_accumulation)
    {
        // Clear bit 11 (restore from workaround for budabackend#1372)
        // Uses helper from llk_math_common.h which includes tensix_sync()
        _llk_math_dbg_feature_enable_();
        // Note: BH doesn't need format restoration (init doesn't change it)
    }
}
