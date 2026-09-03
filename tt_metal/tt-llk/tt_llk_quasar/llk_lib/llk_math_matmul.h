// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "llk_math_common.h"

// [#55076 / 0x19 DIAGNOSTIC -- DO NOT UPSTREAM]
// Waypoints inside _llk_math_matmul_block_ to pinpoint which Tensix instruction the MATH RISC wedges
// on. The kernel-side markers only bound it to "somewhere inside matmul_block" (MACQ), and mapping the
// reported PC needs TRISC1's runtime text base, which is assigned per-build by the host
// (kernel_text_offsets[processor_index] in tt_metal/impl/program/program.cpp) and cannot be derived
// statically. Waypoints sidestep that: the last one written names the site directly.
// __has_include guard so the standalone tt-llk test build (which has no tt-metal debug headers on its
// include path) is unaffected; WAYPOINT is already a no-op unless WATCHER_ENABLED.
#if __has_include("api/debug/waypoint.h")
#include "api/debug/waypoint.h"
#endif
#ifndef WAYPOINT
#define WAYPOINT(x)
#endif
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

// [#55076 / 0x19] Where the DEST section base gets programmed.
//   false (STOCK): _llk_math_matmul_block_ writes cfg[DEST_TARGET_REG_CFG_MATH_SECn_Offset] on every
//                  call -- the original behaviour, and the store MATH was proven to park on (SDW1).
//   true          : hoisted to _llk_math_matmul_init_ instead. Value-preserving (a block always
//                  targets dest tile 0, so it rewrote what _llk_math_dest_section_done_ had just
//                  written on the bank flip) and it moved the wedge SDW1 -> MB2, but did not fix.
// Default STOCK so a waveform captures the unmodified failure.
constexpr bool kHoistDestBaseToInit = false;

/**
 * @brief Initializes addrmod for matrix multiply operation.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, programs addr_mods for the MXFP4_2x non-DI MOP variant (8 MVMULs covering only A0/A1 and B0/B1; SrcA in MxFp4_2x_A/B
 * drives the 2x sub-element expansion).
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_addrmod_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    constexpr bool high_fidelity      = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT  = high_fidelity ? 1 : 0;
    const std::uint16_t num_tile_incr = (ct_dim >= rt_dim) ? 64 : ct_dim * 64;

    if constexpr (ENABLE_2X_FORMAT)
    {
        // Non-DI MXFP4_2x traversal (mirrors the DI X2 (srca,srcb,dest) sequence):
        //   #0 (0, 0, 0)     #1 (0, 8, 8)
        //   #2 (16, 0,16)    #3 (16, 8,24)
        //   #4 (0,16,32)     #5 (0,24,40)
        //   #6 (16,16,48)    #7 (16,24,56)
        // SrcB needs two distinct "wrap" targets (0 then 16). We exploit RWC_SrcB_Cr:
        // at #1->#2 it is still 0 so srcb cr=1 wraps to 0; at #3->#4 we pump it up to
        // 16 via {cr=1, incr=16}; at #5->#6 srcb cr=1 then wraps to 16.

        // Common in-replay step (used between #0->#1, #2->#3, #4->#5, #6->#7).
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 8, .clr = 0, .cr = 0},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // #1 -> #2: srca steps to A1, srcb wraps back to 0 (RWC_SrcB_Cr is 0 here).
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_1);

        // #3 -> #4: srca wraps back to 0, srcb advances RWC_SrcB_Cr from 0 to 16 in
        // the same step ({cr=1, incr=16} -> srcb = 0+16 = 16, RWC_SrcB_Cr := 16).
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 1},
            .srcb = {.incr = 16, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_2);

        // #5 -> #6: srca steps to A1, srcb wraps to RWC_SrcB_Cr (= 16 now).
        addr_mod_t {
            .srca = {.incr = 16, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 1},
            .dest = {.incr = 8, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_3);

        // matmul_op (intermediate fidelity phase): reset src registers, snap dest
        // back to start of this tile, advance fidelity counter.
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 0},
            .srcb     = {.incr = 0, .clr = 1, .cr = 0},
            .dest     = {.incr = 0, .clr = 0, .cr = 1},
            .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
        }
            .set(ADDR_MOD_4);

        // matmul_op_last: end-of-tile, advance dest to next tile, clear fidelity.
        addr_mod_t {
            .srca     = {.incr = 0, .clr = 1, .cr = 0},
            .srcb     = {.incr = 0, .clr = 1, .cr = 0},
            .dest     = {.incr = num_tile_incr, .clr = 0, .cr = 1},
            .fidelity = {.incr = 0, .clr = 1},
        }
            .set(ADDR_MOD_5);
        return;
    }

    // MVMUL does D = B*A

    // Inner Loop --> 32/8 = 4 times for the full 32x16 face
    // DEST -- 8 rows are calculated each time
    // SRCB -- 8 rows are needed
    // SRCA -- full 16x16 gets used -- hardware will pair cols of A with rows of B
    // D[8,16] = B[8,16] * A[16,16]
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 8, .clr = 0, .cr = 0},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 1},
        .srcb = {.incr = 32, .clr = 0, .cr = 1},
        .dest = {.incr = 8, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);

    // reset all, increment dest carriage return
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = num_tile_incr, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_3);

    addr_mod_t {
        .srca = {.incr = 32, .clr = 0, .cr = 1},
        .srcb = {.incr = 48, .clr = 0, .cr = 1}, // cr=32 before, cr+48=16 after wrapping
        .dest = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_4);

    // reset all, increment fidelity if we have more fidelity phases
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_5);
}

/**
 * @brief Initializes addrmod for matrix multiply operation using the direct-indexing instruction variant.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE>
inline void _llk_math_matmul_di_addrmod_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    constexpr bool high_fidelity      = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT  = high_fidelity ? 1 : 0;
    const std::uint16_t num_tile_incr = (ct_dim >= rt_dim) ? 64 : ct_dim * 64;

    // Direct indexing supplies absolute srcb/srca/dest indices in each MVMULDI, so the
    // replayed instructions (which all select ADDR_MOD_0) must apply no auto-increment.
    // Program it explicitly: otherwise ADDR_MOD_0 is inherited from a prior matmul kernel
    // (e.g. a regular MVMUL matmul leaves dest/srcb +=8), perturbing the dest addressing.
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // only increment fidelity if we have more fidelity phases
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 0},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_1);

    addr_mod_t {
        .srca     = {.incr = 0, .clr = 0, .cr = 0},
        .srcb     = {.incr = 0, .clr = 0, .cr = 0},
        .dest     = {.incr = num_tile_incr, .clr = 0, .cr = 0},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_2);
}

/**
 * @brief Number of MVMULs recorded into the replay buffer for one Tile x Tile matrix multiply.
 *
 * One less than the total MVMUL count: the closing MVMUL of the Tile x Tile operation is issued from
 * outside the replay buffer by the MOP in @ref _llk_math_matmul_mop_config_, or directly from the
 * RISC core in the experimental no-MOP path.
 *
 * @tparam ENABLE_2X_FORMAT: Select the MXFP4_2x traversal (8 MVMULs) instead of the plain one (16).
 */
template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint32_t _llk_math_matmul_replay_buf_len_()
{
    return ENABLE_2X_FORMAT ? (8 - 1) : (16 - 1);
}

/**
 * @brief Addrmod slot used by the per-fidelity-phase closing MVMUL of a Tile x Tile matrix multiply.
 *
 * @tparam ENABLE_2X_FORMAT: Select the MXFP4_2x addrmod layout instead of the plain one.
 * @note Paired with @ref _llk_math_matmul_op_last_addr_mod_; both slots are programmed by @ref _llk_math_matmul_addrmod_.
 */
template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint8_t _llk_math_matmul_op_addr_mod_()
{
    return ENABLE_2X_FORMAT ? ADDR_MOD_4 : ADDR_MOD_5;
}

/**
 * @brief Addrmod slot used by the final MVMUL of a Tile x Tile matrix multiply (advances dest to the next tile).
 *
 * @tparam ENABLE_2X_FORMAT: Select the MXFP4_2x addrmod layout instead of the plain one.
 * @note Paired with @ref _llk_math_matmul_op_addr_mod_; both slots are programmed by @ref _llk_math_matmul_addrmod_.
 */
template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint8_t _llk_math_matmul_op_last_addr_mod_()
{
    return ENABLE_2X_FORMAT ? ADDR_MOD_5 : ADDR_MOD_3;
}

/**
 * @brief Records the MVMUL sequence for one Tile x Tile matrix multiply into replay buffer slot 0.
 *
 * Extracted from @ref _llk_math_matmul_mop_config_ so the experimental no-MOP matmul replays this exact
 * sequence rather than restating it. Length is @ref _llk_math_matmul_replay_buf_len_.
 *
 * @tparam ENABLE_2X_FORMAT: When true, records the non-DI MXFP4_2x variant (7-MVMUL replay traversing only A0/A1 and B0/B1; relies on SrcA being unpacked as
 * MxFp4_2x_A/B for the 2x sub-element expansion).
 * @note Call @ref _llk_math_matmul_addrmod_ with the matching template args first, the recorded MVMULs select its addrmod slots.
 */
template <bool ENABLE_2X_FORMAT>
inline void _llk_math_matmul_load_replay_()
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    constexpr std::uint32_t replay_buf_len = _llk_math_matmul_replay_buf_len_<ENABLE_2X_FORMAT>();

    if constexpr (ENABLE_2X_FORMAT)
    {
        // Non-DI MXFP4_2x: 7-MVMUL replay + matmul_op = 8 MVMULs per tile (vs 16 in plain non-DI).
        // (srca,srcb,dest) sequence mirrors the DI X2 path:
        //   #0 (0,  0,  0)  B0[0:7]*A0
        //   #1 (0,  8,  8)  B0[8:15]*A0
        //   #2 (16, 0, 16)  B0[0:7]*A1     <- ADDR_MOD_1 (srca+=16, srcb cr->0)
        //   #3 (16, 8, 24)  B0[8:15]*A1
        //   #4 (0, 16, 32)  B1[0:7]*A0     <- ADDR_MOD_2 (srca cr->0, srcb cr+=16 lifts RWC_SrcB_Cr to 16)
        //   #5 (0, 24, 40)  B1[8:15]*A0
        //   #6 (16,16, 48)  B1[0:7]*A1     <- ADDR_MOD_3 (srca+=16, srcb cr->16)
        //   #7 (16,24, 56)  B1[8:15]*A1    <- matmul_op (ADDR_MOD_4) / matmul_op_last (ADDR_MOD_5)
        load_replay_buf<0, replay_buf_len>(
            []
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #0 -> srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // #1 -> srca+=16, srcb cr->0, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #2 -> srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // #3 -> srca cr->0, srcb cr+=16 (=16), dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #4 -> srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // #5 -> srca+=16, srcb cr->16, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #6 -> srcb+=8, dest+=8
            });
    }
    else
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            []
            {
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16/32, srcb=0, dest+=8  // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8,  dest+=8  // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1 // srca=0,    srcb=32,  dest+=8  // A1 -> A2 if transposed

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=srca, srcb+=8,  dest+=8 // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // B2A1 // srca=32/16,srcb=16,  dest=0  // A1 -> A2 && srca=16 if transposed

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16,  dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8,  dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3 // srca=32,   srcb=48,  dest+=8

                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2 // srca=srca, srcb+=8,  dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2 // srca+=16,  srcb=0,   dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3 // srca=srca, srcb+=8,  dest+=8
            });
    }
}

/**
 * @brief Initializes mop config for matrix multiply operation.
 *
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]; output is a matrix block of dimension [rt_dim, ct_dim].
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * Expands to FIDELITY_PHASES iterations of [REPLAY(0, replay_buf_len), matmul_op], with matmul_op_last
 * replacing matmul_op in the final iteration.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, emits the non-DI MXFP4_2x variant (7-MVMUL replay traversing only A0/A1 and B0/B1; relies on SrcA being unpacked as
 * MxFp4_2x_A/B for the 2x sub-element expansion).
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_mop_config_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);

    const bool reuse_a = ct_dim >= rt_dim;

    constexpr std::uint32_t replay_buf_len = _llk_math_matmul_replay_buf_len_<ENABLE_2X_FORMAT>();

    _llk_math_matmul_load_replay_<ENABLE_2X_FORMAT>();

    constexpr std::uint8_t matmul_op_addr_mod      = _llk_math_matmul_op_addr_mod_<ENABLE_2X_FORMAT>();
    constexpr std::uint8_t matmul_op_last_addr_mod = _llk_math_matmul_op_last_addr_mod_<ENABLE_2X_FORMAT>();
    constexpr static std::uint32_t matmul_op       = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, matmul_op_addr_mod, 0);
    // NOTE: this clears only the ITERATED operand. The REUSED one is cleared separately by the
    // TTI_SETRWC(CLR_B/CLR_A) in _llk_math_matmul_block_ below, so the two are already balanced --
    // do not "fix" this to CLR_AB, that double-clears the reused operand's bank.
    const std::uint32_t matmul_op_last =
        reuse_a ? TT_OP_MVMUL(p_setrwc::CLR_A, 0, matmul_op_last_addr_mod, 0) : TT_OP_MVMUL(p_setrwc::CLR_B, 0, matmul_op_last_addr_mod, 0);

    ckernel_template temp(1 /* outer loop */, FIDELITY_PHASES, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), matmul_op);
    temp.set_last_outer_loop_instr(matmul_op_last);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes mop config for matrix multiply operation using the direct-indexing instruction variant.
 *
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: Enable matrix multiplication with MXFP_2X mode (double the performance)
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT>
inline void _llk_math_matmul_di_mop_config_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker will always load faces in f0,f1,f2,f3 order
    // if in1 is transposed then faces 1&2 need to be swapped during read
    // by changing address increment amount via addr_mods
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);
    const bool reuse_a                      = ct_dim >= rt_dim;

    constexpr std::uint32_t replay_buf_len =
        ENABLE_2X_FORMAT ? 8 - 1 : 16 - 1; // -1 since the last instruction for the Tile * Tile operation will come out of the MOP
    if constexpr (ENABLE_2X_FORMAT)
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            []
            {
                // [B0] x [A0 A1]
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x0, 0x0, 0x0); // B0[0:7]*A0  srcb=0x0<<2='d0, srca=0x0<<2='d0, dest=0x0<<2='d0
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x0, 0x0, 0x2); // B0[8:15]*A0 srcb=0x2<<2='d8, srca=0x0<<2='d0, dest=0x2<<2='d8
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x4, 0x0, 0x4); // B0[0:7]*A1  srcb=0x0<<2='d0, srca=0x4<<2='d16, dest=0x4<<2='d16
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x4, 0x0, 0x6); // B0[8:15]*A1 srcb=0x2<<2='d8, srca=0x4<<2='d16, dest=0x6<<2='d24
                // [B1] x [A0 A1]
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x0, 0x0, 0x8); // B1[0:7]*A0  srcb=0x4<<2='d16, srca=0x0<<2='d0, dest=0x8<<2='d32
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x0, 0x0, 0xA); // B1[8:15]*A0 srcb=0x6<<2='d24, srca=0x0<<2='d0, dest=0xA<<2='d40
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x4, 0x0, 0xC); // B1[0:7]*A1  srcb=0x4<<2='d16, srca=0x4<<2='d16, dest=0xC<<2='d48
            });
    }
    else
    {
        load_replay_buf<0, replay_buf_len>(
            // Lambda function to load reply buffer
            []
            {
                // [B0] x [A0 A1]
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x0, 0x0, 0x0); // B0[0:7]*A0  srcb=0x0<<2='d0, srca=0x0<<2='d0, dest=0x0<<2='d0
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x0, 0x0, 0x2); // B0[8:15]*A0 srcb=0x2<<2='d8, srca=0x0<<2='d0, dest=0x2<<2='d8
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x0, 0x4, 0x0, 0x4); // B0[0:7]*A1  srcb=0x0<<2='d0, srca=0x4<<2='d16, dest=0x4<<2='d16 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x8 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x2, 0x4, 0x0, 0x6); // B0[8:15]*A1 srcb=0x2<<2='d8, srca=0x4<<2='d16, dest=0x6<<2='d24 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x8 if transposed.

                // [B2] x [A0 A1]
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x8, 0x0, 0x0, 0x8); // B2[0:7]*A0  srcb=0x8<<2='d32, srca=0x0<<2='d0, dest=0x8<<2='d32
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xA, 0x0, 0x0, 0xA); // B2[8:15]*A0 srcb=0xA<<2='d40, srca=0x0<<2='d0, dest=0xA<<2='d40
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x8, 0x4, 0x0, 0xC); // B2[0:7]*A1  srcb=0x8<<2='d32, srca=0x4<<2='d16, dest=0xC<<2='d48 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x8 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xA, 0x4, 0x0, 0xE); // B2[8:15]*A1 srcb=0xA<<2='d40, srca=0x4<<2='d16, dest=0xE<<2='d56 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x8 if transposed.

                // [B1] x [A2 A3] (Accumulates to the result of [B0] x [A0 A1] )
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0x8, 0x0, 0x0); // B1[0:7]*A2  srcb=0x4<<2='d16, srca=0x8<<2='d32, dest=0x0<<2='d0 // A2 -> A1 if
                                                                          // transposed. That is, srca should be set 0x4 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x8, 0x0, 0x2); // B1[8:15]*A2 srcb=0x6<<2='d24, srca=0x8<<2='d32, dest=0x2<<2='d8 // A2 -> A1 if
                                                                          // transposed. That is, srca should be set 0x4 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x4, 0xC, 0x0, 0x4); // B1[0:7]*A3  srcb=0x4<<2='d16, srca=0xC<<2='d48, dest=0x4<<2='d16
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0xC, 0x0, 0x6); // B1[8:15]*A3 srcb=0x6<<2='d24, srca=0xC<<2='d48, dest=0x6<<2='d24

                // [B3] x [A2 A3] (Accumulates to the result of [B2] x [A0 A1]  )
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xC, 0x8, 0x0, 0x8); // B3[0:7]*A2  srcb=0xC<<2='d48, srca=0x8<<2='d32, dest=0x8<<2='d32 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x4 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xE, 0x8, 0x0, 0xA); // B3[8:15]*A2 srcb=0xE<<2='d56, srca=0x8<<2='d32, dest=0xA<<2='d40 // A1 -> A2 if
                                                                          // transposed. That is, srca should be set 0x4 if transposed.
                TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xC, 0xC, 0x0, 0xC); // B3[0:7]*A3  srcb=0xC<<2='d48, srca=0xC<<2='d48, dest=0xC<<2='d48
            });
    }

    /* Just choose what is more readable*/
    constexpr static std::uint32_t matmul_op =
        ENABLE_2X_FORMAT ? TT_OP_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0x6, 0x4, ADDR_MOD_1, 0xE)
                         :                                                     // B1[8:15]*A1 srcb=0x6<<2='d24, srca=0x4<<2='d16, dest=0xE<<2='d56
            TT_OP_MVMULDI(p_setrwc::CLR_NONE, 0x0, 0xE, 0xC, ADDR_MOD_1, 0xE); // B3[8:15]*A3 srcb=0xE<<2='d56, srca=0xC<<2='d48, dest=0xE<<2='d56
    std::uint32_t matmul_op_last;
    if constexpr (ENABLE_2X_FORMAT)
    {
        matmul_op_last =
            reuse_a ? TT_OP_MVMULDI(p_setrwc::CLR_A, 0x0, 0x6, 0x4, ADDR_MOD_2, 0xE) : TT_OP_MVMULDI(p_setrwc::CLR_B, 0x0, 0x6, 0x4, ADDR_MOD_2, 0xE);
    }
    else
    {
        matmul_op_last =
            reuse_a ? TT_OP_MVMULDI(p_setrwc::CLR_A, 0x0, 0xE, 0xC, ADDR_MOD_2, 0xE) : TT_OP_MVMULDI(p_setrwc::CLR_B, 0x0, 0xE, 0xC, ADDR_MOD_2, 0xE);
    }

    ckernel_template temp(1 /* outer loop */, FIDELITY_PHASES, TT_OP_REPLAY(0, replay_buf_len, 0, 0, 0, 0), matmul_op);
    temp.set_last_outer_loop_instr(matmul_op_last);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Initializes addrmod and config for matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA.
 *
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]; output is a matrix block of dimension [rt_dim, ct_dim].
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_DIRECT_INDEXING: Enable direct indexing matrix multiplication
 * @tparam ENABLE_2X_FORMAT: Enable matrix multiplication with MXFP_2X mode (double the performance)
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @note On the unpack thread, pair with @ref _llk_unpack_matmul_init_ (T0); on the pack thread, pair with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_math_matmul_tile_ or @ref _llk_math_matmul_block_ runs the configured matmul with matching template args.
 */

template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_DIRECT_INDEXING = false, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_init_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    if constexpr (ENABLE_DIRECT_INDEXING)
    {
        // Direct-indexing path. Supports plain DI and DI+X2 (DI+X2 is the original
        // MXFP4_2x matmul implementation).
        _llk_math_matmul_di_addrmod_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim);
        _llk_math_matmul_di_mop_config_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
    }
    else
    {
        _llk_math_matmul_addrmod_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
        _llk_math_matmul_mop_config_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
    }

    // [#55076 / 0x19] See kHoistDestBaseToInit at the top of this file.
    if constexpr (kHoistDestBaseToInit)
    {
        _set_dst_write_addr_<DstTileShape::Tile32x32>(0);
    }

    _reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Does matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA for a single tile.
 *
 * Input 0 = 1 tile -> SrcB reg, Input 1 = 1 tile -> SrcA reg, output = 1 tile -> Dst reg at specified dst_index.
 *
 * @param dst_index: Tile index in destination register. For DstSync::SyncHalf: values = [0-7] for 16-bit formats, values = [0-3] for 32-bit formats. For
 * DstSync::SyncFull: values = [0-15] for 16-bit formats, values = [0-7] for 32-bit formats
 * @note Call @ref _llk_math_matmul_init_ with matching template args before this function.
 */
inline void _llk_math_matmul_tile_(const std::uint32_t dst_index)
{
    _set_dst_write_addr_<DstTileShape::Tile32x32>(dst_index);
    ckernel_template::run_bank0_sw_cntl(instrn_buffer);
    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_ABD_F);
}

/**
 * @brief Does matrix multiply operation of Input 0 * Input 1 -> SrcB * SrcA over a block of tiles.
 *
 * Input 0 dim = [rt_dim, 1], Input 1 dim = [1, ct_dim]; output is a matrix block of dimension [rt_dim, ct_dim].
 * For DstSync::SyncHalf: ct_dim * rt_dim <= 8 tiles in a 16-bit format, ct_dim * rt_dim <= 4 tiles in a 32-bit format.
 * For DstSync::SyncFull: ct_dim * rt_dim <= 16 tiles in a 16-bit format, ct_dim * rt_dim <= 8 tiles in a 32-bit format.
 *
 * IMPORTANT NOTES:
 * 1. Dest index always assumed to start at 0 for this operation.
 * 2. If matrix multiplication includes kt_dim > 1 such that matrix multiplication is:
 *    Input 0 [rt_dim, kt_dim] x Input 1 [kt_dim, ct_dim] = Output [rt_dim, ct_dim],
 *    be aware that this function does not iterate over kt_dim; iterate over kt_dim externally to this function.
 *
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @note Call @ref _llk_math_matmul_init_ with matching template args before this function.
 */
inline void _llk_math_matmul_block_(std::uint8_t ct_dim, std::uint8_t rt_dim)
{
    // [#55076 diagnostic] MB0 = entered matmul_block.
    WAYPOINT("MB0");

    // [#55076 / 0x19] Per-call DEST section base write. Waypointing proved MATH deadlocks parked on
    // exactly this store (last marker SDW1): _llk_math_dest_section_done_ (llk_math_common.h:357)
    // issues STALLWAIT(STALL_CFG, wait: MATH, WAIT_SFPU) before its own base write, stalling the CFG
    // resource until math is idle; the MATH RISC then runs ahead into the next subblock and this
    // store blocks behind that stall. Set kHoistDestBaseToInit=true to move it to init instead.
    if constexpr (!kHoistDestBaseToInit)
    {
        _set_dst_write_addr_<DstTileShape::Tile32x32>(0);
    }

    // MB1 = past the dest-base programming.
    WAYPOINT("MB1");

    const bool reuse_a          = ct_dim >= rt_dim;
    const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim; // reuse-dim

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            // [#55076 diagnostic] MB2 = about to launch the MVMUL MOP for this (t, rut).
            WAYPOINT("MB2");

            ckernel_template::run_bank0_sw_cntl(instrn_buffer);

            // MB3 = the MOP launch (TTI_MOP) was accepted into the instruction buffer. If MATH's last
            // waypoint is MB2 the RISC wedged pushing the MOP itself (ibuf full because the MVMULs are
            // stalled); if MB3, the MVMULs were queued and it wedged on what follows.
            WAYPOINT("MB3");

            // Clear srcB or srcA at end of reuse (once per u block row)
            if (rut == (rut_dim - 1))
            {
                // [#55076 / 0x19] Drain the MU pipeline before releasing the reused operand's Src bank.
                //
                // This is the Quasar port of tt-metal PR #55107 ("[LLK] Drain the math pipeline in the
                // remaining Dest->Src bank waits"), which fixed WH+BH and explicitly EXCLUDED Quasar. On
                // Tensix the Src bank pointer advances only in the EPILOGUE of the preceding Matrix-Unit
                // instruction, so a bank op issued while the last MVMULs of the MOP are still in flight
                // acts on the pre-flip bank.
                //
                // Quasar is structurally more exposed than WH/BH here. On WH/BH this clear is folded into
                // the MOP itself via tmp.set_end_op(TT_OP_SETRWC(...)) (tt_llk_wormhole_b0/llk_lib/
                // llk_math_matmul.h:483-487, tt_llk_blackhole/...:441-445), so the MOP sequencer emits it
                // after the last MVMUL. Quasar instead issues it from the RISC as a separate TTI right
                // after run_bank0_sw_cntl() launches the MOP -- same thread and therefore in-order at
                // issue, but with nothing forcing the in-flight MVMULs to have finished reading SrcB
                // before the dvalid clear / bank flip takes effect.
                //
                // Fits the observed wedge: TRISC0 UPMW (unpacker blocked in its MOP because the Src bank
                // it wants was never properly released), TRISC1 MACQ (math blocked inside matmul_block),
                // TRISC2 PPAK (downstream). Fits "needs >= 3 subblocks" -- Src is double-banked, so the
                // third call is the first bank REUSE. Fits the timing sensitivity and DPRINT masking.
                // And it is Src, not Dest: the serialization test (MDRD) proved math still stalls with
                // the packer provably drained, so Dest was never the contended resource.
                //
                // RESULT 2026-09-02 20:36: tested, hang unchanged (UPMW/MACQ/PPAK, PC 0x381bc -> 0x381b8,
                // so the rebuild DID pick this up -- the PC moved, unlike the two earlier LLK edits that
                // the JIT cache silently skipped). Eliminated. Gated off rather than deleted: it costs a
                // pipeline drain per u-block row, but #55107 is a genuine WH/BH fix that Quasar was
                // excluded from, so this may still be a latent correctness gap worth revisiting.
                // RETEST 2026-09-03: the earlier "no change" result was INVALID. At that point MATH
                // wedged at SDW1 -- on the DEST-section-base CFG store, which is upstream of this
                // SETRWC -- so the drain was never reached in the wedged iteration. With that store
                // hoisted out of the loop, MATH now reaches MB2 (the MOP launch), so this is finally
                // on the executed path and genuinely testable.
                constexpr bool kDrainMathBeforeSrcBankClear = false; // WAVE: off = stock LLK
                if constexpr (kDrainMathBeforeSrcBankClear)
                {
                    TTI_STALLWAIT(p_stall::STALL_MATH, 0, 0, p_stall::MATH);
                }
                if (reuse_a)
                {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_AB_F);
                }
                else
                {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB_F);
                }
                // MB4 = past the Src bank clear for this u-block row.
                WAYPOINT("MB4");
            }
        }

        //  When rt_dim > ct_dim, the matmul block dest tile indices are not equal to 0,1,2,3..7
        //  Instead they have a ct_dim stride, for instance:
        //  If rt_dim = 4, ct_dim = 2, dest tile indices = 0,2,4,6,  1,3,5,7
        //  If rt_dim = 4, ct_dim = 3, dest tile indices = 0,3,6,9,  1,4,7,10,  2,5,8,11
        //  Below offsets by 1 tile * (t+1), for every subsequence above to start from the next dest_idx
        if (!reuse_a && ct_dim >= 2)
        {
            TT_SETRWC(p_setrwc::CLR_NONE, 0, 64 * (t + 1), p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::C_TO_CR_MODE, 0, p_setrwc::SET_D);
        }
    }
    // MB5 = past every u-block row, matmul_block fully issued. If MATH's last waypoint is MB5 then
    // it is NOT wedged inside matmul_block and MACQ was only RISC run-ahead -- look downstream.
    WAYPOINT("MB5");
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
