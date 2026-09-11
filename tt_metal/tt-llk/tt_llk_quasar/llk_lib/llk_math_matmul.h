// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstdint>

#include "llk_math_common.h"
using namespace ckernel;
using namespace ckernel::trisc;
using namespace ckernel::math;

constexpr std::uint8_t FULL_TILE_MVMULS_PER_K_FACE = NUM_FACES * (MAX_FACE_R_DIM / MAX_FPU_ROWS);

// Counter increments programmed into one matmul ADDR_MOD slot.
struct _llk_math_matmul_step_t
{
    std::uint8_t src_a;
    std::uint8_t src_b;
    std::uint16_t dest;
};

/**
 * @brief Replay window and per-slot counter increments for one matmul shape.
 *
 * The replay image is a fixed nest. ADDR_MOD_0 applies after every MVMUL, ADDR_MOD_1
 * after every second, ADDR_MOD_2 after every fourth, and ADDR_MOD_3 after every eighth.
 * The active traversal axes fill the slots from ADDR_MOD_0 upward, in the order FPU row
 * group, then column face, then face row. ADDR_MOD_3 always advances K. A shape with
 * fewer active axes leaves the deeper slots inert, and @ref replay_start_idx selects a
 * window that never executes them:
 *
 *   SrcB face_r_dim | output faces (r x c) | ADDR_MOD_0     | ADDR_MOD_1  | ADDR_MOD_2
 *   ----------------+----------------------+----------------+-------------+------------
 *   16              | 2 x 2                | SrcB row group | column face | face row
 *   16              | 1 x 2                | SrcB row group | column face | inert
 *   16              | 2 x 1                | SrcB row group | face row    | inert
 *   16              | 1 x 1                | SrcB row group | inert       | inert
 *   <= 8            | 1 x 2                | SrcA column    | inert       | inert
 *   <= 8            | 1 x 1                | inert          | inert       | inert
 *
 * A sub-16 face_r_dim always comes with num_faces_r_dim == 1, so the two remaining rows
 * of that table are unreachable. See @ref validate_matmul_tensor_shapes_.
 *
 * Every MVMUL writes the next eight destination rows, so ADDR_MOD_0, ADDR_MOD_1, and
 * ADDR_MOD_2 all carry a Dest increment of one FPU row group wherever they execute.
 * ADDR_MOD_3 rewinds Dest to the start of the tile through its carriage return instead.
 */
struct _llk_math_matmul_execution_geometry_t
{
    std::uint8_t replay_start_idx;
    std::uint8_t replay_len;
    std::uint16_t dst_rows_per_tile;
    std::uint16_t dst_tile_row_increment;
    _llk_math_matmul_step_t next_fpu_rows;      ///< ADDR_MOD_0: advance Dest by one FPU row group.
    _llk_math_matmul_step_t next_face;          ///< ADDR_MOD_1: next available face, column before row.
    _llk_math_matmul_step_t next_face_row_incr; ///< ADDR_MOD_2: next row of the output face grid.
    _llk_math_matmul_step_t next_k_face_src;    ///< ADDR_MOD_3: next K face on each of SrcB and SrcA.
    // One flag per slot that uses a carriage return. ADDR_MOD_2 and ADDR_MOD_3 apply theirs
    // to both SrcA and SrcB; ADDR_MOD_1 applies it to SrcB only.
    bool has_next_face;
    bool has_next_face_row;
    bool has_next_k_face;
};

/**
 * @brief Derives the replay window and address-modifier geometry for matmul.
 *
 * Standard 32x32 operands have four 16x16 faces, two K faces, and two FPU row groups per
 * face. The four output faces therefore require eight MVMULs per K face, giving
 * replay_start_idx = 0 and replay_len = 8 * 2 - 1 = 15. Each output tile occupies
 * 4 * 16 = 64 destination rows.
 *
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param src_b_shape: Input 0/SrcB tile shape.
 * @param src_a_shape: Input 1/SrcA tile shape.
 * @note @ref _llk_math_matmul_execution_geometry_t documents which axis each slot carries
 * at a given shape.
 */
inline _llk_math_matmul_execution_geometry_t _llk_math_matmul_execution_geometry_(
    const std::uint8_t ct_dim, const std::uint8_t rt_dim, const TensorShape src_b_shape, const TensorShape src_a_shape)
{
    LLK_ASSERT(validate_matmul_tensor_shapes_(src_b_shape, src_a_shape), "unsupported SrcB/SrcA TensorShape pair for matmul");

    // output_shape looks redundant, because only its face counts are read. Keep it: it
    // consolidates the three counts into one packed word, and reading them from the two
    // operand structs instead costs 128 bytes per math kernel (measured).
    const TensorShape output_shape =
        make_tensor_shape(src_b_shape.face_r_dim, src_a_shape.face_c_dim, src_b_shape.num_faces_r_dim, src_a_shape.num_faces_c_dim);
    const std::uint8_t face_rows          = src_b_shape.face_r_dim < MAX_FPU_ROWS ? MAX_FPU_ROWS : src_b_shape.face_r_dim;
    const std::uint16_t dst_rows_per_tile = static_cast<std::uint16_t>(output_shape.total_num_faces()) * face_rows;
    const std::uint8_t num_k_faces        = src_b_shape.num_faces_c_dim;
    const std::uint8_t face_row_passes    = src_b_shape.face_r_dim > MAX_FPU_ROWS ? 2 : 1;
    const std::uint8_t mvmuls_per_k_face  = output_shape.total_num_faces() * face_row_passes;

    // Which traversal axes this shape leaves active. A face holds one or two FPU row
    // groups, and has_row_faces implies two of them: a sub-16 face_r_dim forces
    // num_faces_r_dim == 1, so no shape has row faces and a single row group. The slot
    // predicates below lean on that implication.
    //
    // Keep these predicates named. Each is read two to five times, and substituting them
    // at their use sites costs 204 bytes per math kernel (measured): the compiler reloads
    // and re-compares the shape word at every site instead of once here.
    const bool face_has_two_row_groups  = face_rows > MAX_FPU_ROWS;
    const bool has_column_faces         = output_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM;
    const bool has_row_faces            = output_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM;
    const std::int32_t src_b_row_stride = face_rows * num_k_faces;

    // A single-row-group face has no SrcB row group to step, so SrcA takes the column face
    // instead. Dest advances one row group either way: every MVMUL writes eight rows.
    const std::int32_t fpu_rows_src_a = !face_has_two_row_groups && has_column_faces ? MAX_FACE_C_DIM : 0;
    const std::int32_t fpu_rows_src_b = face_has_two_row_groups ? MAX_FPU_ROWS : 0;
    const std::int32_t fpu_rows_dest  = face_has_two_row_groups || has_column_faces ? MAX_FPU_ROWS : 0;

    // Slot assignment, innermost first. The column face outranks the face row. Because
    // has_row_faces implies face_has_two_row_groups, the face row is active exactly when
    // there are row faces and no column face.
    const bool next_face_is_column = face_has_two_row_groups && has_column_faces;
    const bool next_face_is_row    = has_row_faces && !has_column_faces;
    const bool has_next_face       = next_face_is_column || next_face_is_row;
    const std::int32_t face_src_a  = next_face_is_column ? MAX_FACE_C_DIM : 0;
    const std::int32_t face_src_b  = next_face_is_row ? src_b_row_stride : 0;
    const std::int32_t face_dest   = has_next_face ? MAX_FPU_ROWS : 0;

    const bool has_next_face_row      = has_column_faces && has_row_faces;
    const std::int32_t face_row_src_b = has_next_face_row ? src_b_row_stride : 0;
    const std::int32_t face_row_dest  = has_next_face_row ? MAX_FPU_ROWS : 0;

    const bool has_next_k_face      = num_k_faces > 1;
    const std::int32_t k_face_src_a = has_next_k_face ? MAX_FACE_C_DIM * output_shape.num_faces_c_dim : 0;
    // The SrcB counter wraps modulo 64. Encode negative increments as 6-bit two's complement.
    const std::int32_t k_face_src_b = has_next_k_face ? face_rows - (has_row_faces ? src_b_row_stride : 0) : 0;

    return _llk_math_matmul_execution_geometry_t {
        .replay_start_idx       = static_cast<std::uint8_t>(FULL_TILE_MVMULS_PER_K_FACE - mvmuls_per_k_face),
        .replay_len             = static_cast<std::uint8_t>(mvmuls_per_k_face * num_k_faces - 1),
        .dst_rows_per_tile      = dst_rows_per_tile,
        .dst_tile_row_increment = static_cast<std::uint16_t>((ct_dim >= rt_dim ? 1 : ct_dim) * dst_rows_per_tile),
        .next_fpu_rows =
            {
                .src_a = static_cast<std::uint8_t>(fpu_rows_src_a),
                .src_b = static_cast<std::uint8_t>(0x3F & fpu_rows_src_b),
                .dest  = static_cast<std::uint16_t>(fpu_rows_dest),
            },
        .next_face =
            {
                .src_a = static_cast<std::uint8_t>(face_src_a),
                .src_b = static_cast<std::uint8_t>(0x3F & face_src_b),
                .dest  = static_cast<std::uint16_t>(face_dest),
            },
        .next_face_row_incr =
            {
                .src_a = 0,
                .src_b = static_cast<std::uint8_t>(0x3F & face_row_src_b),
                .dest  = static_cast<std::uint16_t>(face_row_dest),
            },
        .next_k_face_src =
            {
                .src_a = static_cast<std::uint8_t>(k_face_src_a),
                .src_b = static_cast<std::uint8_t>(0x3F & k_face_src_b),
                .dest  = 0,
            },
        .has_next_face     = has_next_face,
        .has_next_face_row = has_next_face_row,
        .has_next_k_face   = has_next_k_face,
    };
}

/**
 * @brief Initializes addrmod for matrix multiply operation.
 *
 * Non-2x matmul derives its four per-slot steps from geometry. The 2x path retains its fixed layout.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, programs addr_mods for the MXFP4_2x non-DI MOP variant.
 * The variant uses 8 MVMULs for A0/A1 and B0/B1. SrcA uses MxFp4_2x_A/B for the 2x sub-element expansion.
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @param geometry: Replay and address-modifier geometry from @ref _llk_math_matmul_execution_geometry_.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_addrmod_(const std::uint8_t ct_dim, const std::uint8_t rt_dim, const _llk_math_matmul_execution_geometry_t& geometry)
{
    constexpr bool high_fidelity     = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT = high_fidelity ? 1 : 0;

    if constexpr (ENABLE_2X_FORMAT)
    {
        const std::uint16_t num_tile_incr = (ct_dim >= rt_dim) ? 64 : ct_dim * 64;
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

    // Advance Dest by one FPU row group. This slot never uses a carriage return.
    addr_mod_t {
        .srca = {.incr = geometry.next_fpu_rows.src_a, .clr = 0, .cr = 0},
        .srcb = {.incr = geometry.next_fpu_rows.src_b, .clr = 0, .cr = 0},
        .dest = {.incr = geometry.next_fpu_rows.dest, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // Advance to the next available face, the column face when present and the row face
    // otherwise. SrcB CR discards the row-group step taken by ADDR_MOD_0.
    addr_mod_t {
        .srca = {.incr = geometry.next_face.src_a, .clr = 0, .cr = 0},
        .srcb = {.incr = geometry.next_face.src_b, .clr = 0, .cr = geometry.has_next_face},
        .dest = {.incr = geometry.next_face.dest, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    // Advance to the next row of the output face grid: rewind the SrcA column face and
    // rebase SrcB one row of faces down.
    addr_mod_t {
        .srca = {.incr = geometry.next_face_row_incr.src_a, .clr = 0, .cr = geometry.has_next_face_row},
        .srcb = {.incr = geometry.next_face_row_incr.src_b, .clr = 0, .cr = geometry.has_next_face_row},
        .dest = {.incr = geometry.next_face_row_incr.dest, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);

    // Advance both sources one K face and rewind Dest to the start of the tile.
    addr_mod_t {
        .srca = {.incr = geometry.next_k_face_src.src_a, .clr = 0, .cr = geometry.has_next_k_face},
        .srcb = {.incr = geometry.next_k_face_src.src_b, .clr = 0, .cr = geometry.has_next_k_face},
        .dest = {.incr = geometry.next_k_face_src.dest, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_3);

    // This address modifier ends a fidelity phase.
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_4);

    // This address modifier ends the tile.
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = geometry.dst_tile_row_increment, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_5);
}

/**
 * @brief Programs the full-tile geometry addr-mod layout used by no-MOP matmul.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_addrmod_(const std::uint8_t ct_dim, const std::uint8_t rt_dim)
{
    const _llk_math_matmul_execution_geometry_t geometry = _llk_math_matmul_execution_geometry_(ct_dim, rt_dim, DEFAULT_TENSOR_SHAPE, DEFAULT_TENSOR_SHAPE);
    _llk_math_matmul_addrmod_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim, geometry);
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
 * @brief Returns the number of MVMULs recorded into the matmul replay image.
 *
 * One less than the total MVMUL count. The closing MVMUL of the Tile x Tile operation is issued from
 * outside the replay buffer by the MOP in @ref _llk_math_matmul_mop_config_, or directly from the
 * RISC core in the experimental no-MOP path.
 *
 * @tparam ENABLE_2X_FORMAT: Select the MXFP4_2x replay image.
 */
template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint32_t _llk_math_matmul_replay_buf_len_()
{
    return (ENABLE_2X_FORMAT ? FULL_TILE_MVMULS_PER_K_FACE : MAX_NUM_FACES_C_DIM * FULL_TILE_MVMULS_PER_K_FACE) - 1;
}

/**
 * @brief Records the standard or 2x MVMUL image into replay buffer slot 0.
 *
 * Standard matmul always records the full 15-entry K-outer image. MOP-based tiny matmul selects a
 * window from that image.
 *
 * @tparam ENABLE_2X_FORMAT: When true, records the non-DI MXFP4_2x variant.
 * The variant uses a 7-MVMUL replay for A0/A1 and B0/B1. SrcA uses MxFp4_2x_A/B for the 2x sub-element expansion.
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
            []
            {
                // Tiny tiles select a window and derive each transition from their geometry.
                // For full-tile input (32x32 x 32x32), the transitions are as follows:
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A0 // srca=srca,  srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B0A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B0A1 // srca=srca, srcb+=8, dest+=8 // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B0A1 // srca=0, srcb=32, dest+=8 // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A0 // srca=srca, srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B2A0 // srca+=16/32, srcb=0, dest+=8 // srca+=32 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B2A1 // srca=srca, srcb+=8, dest+=8 // A1 -> A2 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // B2A1 // srca=32/16,srcb=16, dest=0 // A1 -> A2 && srca=16 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A2 // srca=srca, srcb+=8, dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B1A2 // srca+=16,  srcb=16, dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B1A3 // srca=srca, srcb+=8, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // B1A3 // srca=32, srcb=48, dest+=8
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A2 // srca=srca, srcb+=8, dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // B3A2 // srca+=16, srcb=0, dest+=8 // A2 -> A1 if transposed
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // B3A3 // srca=srca, srcb+=8, dest+=8
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
 * Runs one replay plus one completion MVMUL per fidelity phase, replacing the final phase completion
 * with the tile completion. Standard matmul selects a geometry-dependent window from the K-outer image.
 * MXFP4_2x replays its full image.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, emits the non-DI MXFP4_2x variant.
 * The variant uses a 7-MVMUL replay for A0/A1 and B0/B1. SrcA uses MxFp4_2x_A/B for the 2x sub-element expansion.
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @param geometry: Replay window and destination strides from @ref _llk_math_matmul_execution_geometry_.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_mop_config_(const std::uint8_t ct_dim, const std::uint8_t rt_dim, const _llk_math_matmul_execution_geometry_t& geometry)
{
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);

    const bool reuse_a = ct_dim >= rt_dim;

    constexpr std::uint32_t replay_buf_len = _llk_math_matmul_replay_buf_len_<ENABLE_2X_FORMAT>();

    _llk_math_matmul_load_replay_<ENABLE_2X_FORMAT>();

    const std::uint32_t replay_start_idx = ENABLE_2X_FORMAT ? 0 : geometry.replay_start_idx;
    const std::uint32_t replay_len       = ENABLE_2X_FORMAT ? replay_buf_len : geometry.replay_len;
    LLK_ASSERT(replay_len > 0 && replay_start_idx + replay_len <= replay_buf_len, "matmul replay range exceeds the replay image");
    constexpr std::uint32_t phase_done_mvmul = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
    const std::uint32_t tile_done_mvmul      = reuse_a ? TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0) : TT_OP_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);

    ckernel_template temp(1, FIDELITY_PHASES, TT_OP_REPLAY(replay_start_idx, replay_len, 0, 0, 0, 0), phase_done_mvmul);
    temp.set_last_outer_loop_instr(tile_done_mvmul);
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
 * Standard MOP matmul supports the validated tiny-tile pairs. Direct-indexing and 2x remain full-tile only.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_DIRECT_INDEXING: Enable direct indexing matrix multiplication
 * @tparam ENABLE_2X_FORMAT: Enable matrix multiplication with MXFP_2X mode (double the performance)
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @param src_b_shape: Input 0/SrcB tile shape.
 * @param src_a_shape: Input 1/SrcA tile shape.
 * @note On the unpack thread, pair with @ref _llk_unpack_matmul_init_ (T0); on the pack thread, pair with @ref _llk_pack_init_ (T2).
 * @note @ref _llk_math_matmul_tile_ or @ref _llk_math_matmul_block_ runs the configured matmul with matching template args.
 */

template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_DIRECT_INDEXING = false, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_init_(
    const std::uint8_t ct_dim,
    const std::uint8_t rt_dim,
    const TensorShape src_b_shape = DEFAULT_TENSOR_SHAPE,
    const TensorShape src_a_shape = DEFAULT_TENSOR_SHAPE)
{
    if constexpr (ENABLE_DIRECT_INDEXING || ENABLE_2X_FORMAT)
    {
        LLK_ASSERT(
            src_b_shape.face_r_dim == MAX_FACE_R_DIM && src_b_shape.total_num_faces() == NUM_FACES && src_a_shape.face_r_dim == MAX_FACE_R_DIM &&
                src_a_shape.total_num_faces() == NUM_FACES,
            "direct-indexing and 2x matmul support exact 16x16-face, 2x2 operand shapes only");
    }

    if constexpr (ENABLE_DIRECT_INDEXING)
    {
        // Direct-indexing path. Supports plain DI and DI+X2 (DI+X2 is the original
        // MXFP4_2x matmul implementation).
        _llk_math_matmul_di_addrmod_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim);
        _llk_math_matmul_di_mop_config_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
        _set_tile_shape_idx_gpr_(NUM_FACES * MAX_FACE_R_DIM);
    }
    else
    {
        const _llk_math_matmul_execution_geometry_t geometry = _llk_math_matmul_execution_geometry_(ct_dim, rt_dim, src_b_shape, src_a_shape);
        _llk_math_matmul_addrmod_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim, geometry);
        _llk_math_matmul_mop_config_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim, geometry);
        _set_tile_shape_idx_gpr_(geometry.dst_rows_per_tile);
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
    _set_dst_write_addr_by_rows_(dst_index);
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
    // Tile index zero has the same address for every destination shape.
    _set_dst_write_addr_<DstTileShape::Tile32x32>(0);

    const bool reuse_a                    = ct_dim >= rt_dim;
    const std::uint32_t t_dim             = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim           = reuse_a ? ct_dim : rt_dim; // reuse-dim
    const bool strided_dest               = !reuse_a && ct_dim >= 2;
    const std::uint32_t dst_rows_per_tile = strided_dest ? 1U << ckernel::regfile[p_gpr_math::TILE_SHAPE_IDX] : 0;

    for (std::uint32_t t = 0; t < t_dim; t++)
    {
        for (std::uint32_t rut = 0; rut < rut_dim; rut++)
        {
            ckernel_template::run_bank0_sw_cntl(instrn_buffer);

            // Clear srcB or srcA at end of reuse (once per u block row)
            if (rut == (rut_dim - 1))
            {
                if (reuse_a)
                {
                    TTI_SETRWC(p_setrwc::CLR_B, 0, 0, p_setrwc::SET_AB_F);
                }
                else
                {
                    TTI_SETRWC(p_setrwc::CLR_A, 0, 0, p_setrwc::SET_AB_F);
                }
            }
        }

        //  When rt_dim > ct_dim, the matmul block dest tile indices are not equal to 0,1,2,3..7
        //  Instead they have a ct_dim stride, for instance:
        //  If rt_dim = 4, ct_dim = 2, dest tile indices = 0,2,4,6,  1,3,5,7
        //  If rt_dim = 4, ct_dim = 3, dest tile indices = 0,3,6,9,  1,4,7,10,  2,5,8,11
        //  Below offsets by 1 tile * (t+1), for every subsequence above to start from the next dest_idx
        if (strided_dest)
        {
            TT_SETRWC(p_setrwc::CLR_NONE, 0, dst_rows_per_tile * (t + 1), p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::C_TO_CR_MODE, 0, p_setrwc::SET_D);
        }
    }
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
