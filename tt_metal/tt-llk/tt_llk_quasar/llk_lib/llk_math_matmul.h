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

static_assert(MAX_FACE_R_DIM % ELTWISE_MATH_ROWS == 0, "an FPU row band must divide the 16-row face");

// Granularity of the dest row layout. A sub-eight-row face still occupies a whole group, whatever the
// FPU width, so this is the clamp for tiny-tile geometry — not ELTWISE_MATH_ROWS.
constexpr std::uint8_t DEST_ROW_GROUP = ckernel::arch::dest_row_group;

// True when one dest row group takes more than one MVMUL, i.e. the FPU is narrower than the layout
// granularity. Tiny tiles then need their own replay image: the full-tile image walks a 16-row face,
// and no window of it walks an eight-row one.
constexpr bool FPU_SPLITS_DEST_ROW_GROUP = DEST_ROW_GROUP > ELTWISE_MATH_ROWS;

// MVMULs one K face costs on a full tile: every face row band takes one issue.
constexpr std::uint8_t FULL_TILE_MVMULS_PER_K_FACE = NUM_FACES * (MAX_FACE_R_DIM / ELTWISE_MATH_ROWS);

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
    // Set when the shape needs the half-face replay image instead of a window of the full-tile one.
    // Always false where one MVMUL covers a whole dest row group.
    bool use_half_face_replay;
    // Output face grid, which fixes the transition addrmod of each face pair. Only read on the
    // half-face path, so on a part where one MVMUL covers a dest row group nothing loads these.
    std::uint8_t output_num_faces_c;
    std::uint8_t output_num_faces_r;
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
    const std::uint8_t face_rows          = src_b_shape.face_r_dim < DEST_ROW_GROUP ? DEST_ROW_GROUP : src_b_shape.face_r_dim;
    const std::uint16_t dst_rows_per_tile = static_cast<std::uint16_t>(output_shape.total_num_faces()) * face_rows;
    const std::uint8_t num_k_faces        = src_b_shape.num_faces_c_dim;
    const std::uint8_t face_row_passes    = face_rows / ELTWISE_MATH_ROWS;
    const std::uint8_t mvmuls_per_k_face  = output_shape.total_num_faces() * face_row_passes;

    // Which traversal axes this shape leaves active. A face holds one or two FPU row
    // groups, and has_row_faces implies two of them: a sub-16 face_r_dim forces
    // num_faces_r_dim == 1, so no shape has row faces and a single row group. The slot
    // predicates below lean on that implication.
    //
    // Keep these predicates named. Each is read two to five times, and substituting them
    // at their use sites costs 204 bytes per math kernel (measured): the compiler reloads
    // and re-compares the shape word at every site instead of once here.
    const bool face_has_row_groups      = face_rows > ELTWISE_MATH_ROWS;
    const bool has_column_faces         = output_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM;
    const bool has_row_faces            = output_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM;
    const std::int32_t src_b_row_stride = face_rows * num_k_faces;

    // A single-row-group face has no SrcB row group to step, so SrcA takes the column face
    // instead. Dest advances one row group either way: every MVMUL writes ELTWISE_MATH_ROWS rows.
    const std::int32_t fpu_rows_src_a = !face_has_row_groups && has_column_faces ? MAX_FACE_C_DIM : 0;
    const std::int32_t fpu_rows_src_b = face_has_row_groups ? ELTWISE_MATH_ROWS : 0;
    const std::int32_t fpu_rows_dest  = face_has_row_groups || has_column_faces ? ELTWISE_MATH_ROWS : 0;

    // Slot assignment, innermost first. The column face outranks the face row. Because
    // has_row_faces implies face_has_two_row_groups, the face row is active exactly when
    // there are row faces and no column face.
    const bool next_face_is_column = face_has_row_groups && has_column_faces;
    const bool next_face_is_row    = has_row_faces && !has_column_faces;
    const bool has_next_face       = next_face_is_column || next_face_is_row;
    const std::int32_t face_src_a  = next_face_is_column ? MAX_FACE_C_DIM : 0;
    const std::int32_t face_src_b  = next_face_is_row ? src_b_row_stride : 0;
    const std::int32_t face_dest   = has_next_face ? ELTWISE_MATH_ROWS : 0;

    const bool has_next_face_row      = has_column_faces && has_row_faces;
    const std::int32_t face_row_src_b = has_next_face_row ? src_b_row_stride : 0;
    const std::int32_t face_row_dest  = has_next_face_row ? ELTWISE_MATH_ROWS : 0;

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
        // A face clamped to one dest row group needs the half-face image, but only where the FPU is
        // narrower than that group; otherwise one MVMUL covers the face and the window suffices.
        .use_half_face_replay = FPU_SPLITS_DEST_ROW_GROUP && face_rows == DEST_ROW_GROUP,
        .output_num_faces_c   = output_shape.num_faces_c_dim,
        .output_num_faces_r   = output_shape.num_faces_r_dim,
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

// Face pairs a full-tile K-outer traversal visits, and the subset the MXFP4_2x traversal visits.
constexpr std::uint32_t MATMUL_FULL_FACE_PAIRS = MAX_NUM_FACES_C_DIM * NUM_FACES;
constexpr std::uint32_t MATMUL_2X_FACE_PAIRS   = NUM_FACES;

// Row bands one face costs: one MVMUL issue each.
constexpr std::uint32_t MATMUL_BANDS_PER_FACE = MAX_FACE_R_DIM / ELTWISE_MATH_ROWS;

/**
 * @brief Addrmod that closes the face pair at traversal index @p idx.
 *
 * The traversal is k_face { face_row { column_face } } and the addrmod slots follow that nesting
 * (see @ref _llk_math_matmul_addrmod_): ADDR_MOD_1 steps a column face, ADDR_MOD_2 a face row,
 * ADDR_MOD_3 a K face. So the transition is the step of the innermost axis with iterations left.
 *
 * Which counters a transition rewinds depends only on where the pair sits in the face grid, never on
 * how many MVMULs walked the face. The sequence is therefore the same at any FPU width; only the
 * number of ADDR_MOD_0 steps between transitions changes.
 */
constexpr std::uint8_t _llk_math_matmul_face_transition_(const std::uint32_t idx, const std::uint32_t num_faces_c, const std::uint32_t num_faces_r)
{
    const std::uint32_t column_face = idx % num_faces_c;
    const std::uint32_t face_row    = (idx / num_faces_c) % num_faces_r;

    return (column_face + 1 < num_faces_c) ? ADDR_MOD_1 : (face_row + 1 < num_faces_r) ? ADDR_MOD_2 : ADDR_MOD_3;
}

/*
 * The MXFP4_2x addrmod slots do not follow the face-grid nesting above: they encode a bespoke SrcB
 * carriage-return sequence that lifts RWC_SrcB_Cr from 0 to 16 mid-traversal (see the 2x branch of
 * _llk_math_matmul_addrmod_). Its transitions are therefore listed rather than derived.
 */
constexpr std::uint8_t MATMUL_2X_FACE_TRANSITIONS[MATMUL_2X_FACE_PAIRS] = {ADDR_MOD_1, ADDR_MOD_2, ADDR_MOD_3, ADDR_MOD_3 /* unused: supplied by the MOP */};

/**
 * @brief Records a K-outer MVMUL traversal of a full tile's face grid.
 *
 * Each pair walks its face one FPU row band at a time under ADDR_MOD_0, then takes the pair's
 * transition addrmod. The transition of the last pair is left to the MOP, so the image is one MVMUL
 * short of the traversal.
 *
 * @tparam IS_2X: Select the MXFP4_2x traversal, which visits half the face pairs on its own addrmods.
 */
template <bool IS_2X>
inline void _llk_math_matmul_emit_traversal_()
{
    constexpr std::uint32_t FACE_PAIRS = IS_2X ? MATMUL_2X_FACE_PAIRS : MATMUL_FULL_FACE_PAIRS;

#pragma GCC unroll 8
    for (std::uint32_t pair = 0; pair < FACE_PAIRS; ++pair)
    {
        // Walk the face; the last band is replaced by the transition below.
#pragma GCC unroll 4
        for (std::uint32_t band = 0; band + 1 < MATMUL_BANDS_PER_FACE; ++band)
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
        }

        if (pair + 1 < FACE_PAIRS)
        {
            // The addrmod is an "n" immediate, so each arm carries a literal rather than the indexed value.
            const std::uint8_t transition =
                IS_2X ? MATMUL_2X_FACE_TRANSITIONS[pair] : _llk_math_matmul_face_transition_(pair, MAX_NUM_FACES_C_DIM, MAX_NUM_FACES_R_DIM);
            switch (transition)
            {
                case ADDR_MOD_1:
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                    break;
                case ADDR_MOD_2:
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                    break;
                default:
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
                    break;
            }
        }
    }
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
/**
 * @brief Records the half-face MVMUL image over the window a tiny-tile shape selects.
 *
 * Only reachable where the FPU is narrower than a dest row group. Such a shape occupies
 * DEST_ROW_GROUP dest rows per face but only needs DEST_ROW_GROUP / ELTWISE_MATH_ROWS MVMULs to walk
 * one, and no window of the full-tile image (which walks a 16-row face) has that cadence. Recording
 * over the window the geometry already selected keeps @ref _llk_math_matmul_mop_config_ unchanged.
 *
 * @param geometry: Replay window from @ref _llk_math_matmul_execution_geometry_.
 */
inline void _llk_math_matmul_load_half_face_replay_(const _llk_math_matmul_execution_geometry_t& geometry)
{
    constexpr std::uint32_t BANDS_PER_GROUP = DEST_ROW_GROUP / ELTWISE_MATH_ROWS;

    // The window is one block of BANDS_PER_GROUP MVMULs per face pair visited, less the closing
    // MVMUL the MOP issues.
    const std::uint32_t traversal_len = geometry.replay_len + 1U;
    const std::uint32_t face_pairs    = traversal_len / BANDS_PER_GROUP;
    const std::uint32_t num_faces_c   = geometry.output_num_faces_c;
    const std::uint32_t num_faces_r   = geometry.output_num_faces_r;

    LLK_ASSERT(traversal_len % BANDS_PER_GROUP == 0, "half-face replay window must be a whole number of dest row groups");
    LLK_ASSERT(face_pairs <= MATMUL_FULL_FACE_PAIRS, "half-face replay window exceeds the full-tile face grid");

    load_replay_buf(
        geometry.replay_start_idx,
        geometry.replay_len,
        false,
        0,
        0,
        [face_pairs, num_faces_c, num_faces_r]
        {
            for (std::uint32_t pair = 0; pair < face_pairs; pair++)
            {
            // Walk the face's single dest row group, then take this pair's transition. The last
            // pair's transition comes from the MOP.
#pragma GCC unroll 4
                for (std::uint32_t band = 0; band + 1 < BANDS_PER_GROUP; ++band)
                {
                    TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0);
                }

                if (pair + 1 < face_pairs)
                {
                    // Branch rather than index: the addrmod reaches TTI_MVMUL through the "n" asm
                    // constraint and so has to be a compile-time constant at each call.
                    switch (_llk_math_matmul_face_transition_(pair, num_faces_c, num_faces_r))
                    {
                        case ADDR_MOD_1:
                            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0);
                            break;
                        case ADDR_MOD_2:
                            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0);
                            break;
                        default:
                            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0);
                            break;
                    }
                }
            }
        });
}

template <bool ENABLE_2X_FORMAT>
inline void _llk_math_matmul_load_replay_()
{
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    // Unpacker accounts for transposing if required.
    // if TRANSPOSE_EN = true, unpacker loads f0/f2/f1/f3
    // else, unpacker loads f0/f1/f2/f3
    // Math LLKs do not need any transpose handling for Quasar
    constexpr std::uint32_t replay_buf_len = _llk_math_matmul_replay_buf_len_<ENABLE_2X_FORMAT>();

    if constexpr (ENABLE_2X_FORMAT)
    {
        static_assert(
            ckernel::arch::has_mxfp4_2x_replay,
            "this part has no non-DI MXFP4_2x replay image; reach 2x through direct indexing instead (ENABLE_DIRECT_INDEXING)");

        // Non-DI MXFP4_2x: half the face pairs of the plain image (A0/A1 against B0/B1 only), so
        // 8 MVMULs per tile against 16. (srca,srcb,dest) mirrors the DI X2 path:
        //   #0 (0,  0,  0)  B0[0:7]*A0
        //   #1 (0,  8,  8)  B0[8:15]*A0
        //   #2 (16, 0, 16)  B0[0:7]*A1     <- ADDR_MOD_1 (srca+=16, srcb cr->0)
        //   #3 (16, 8, 24)  B0[8:15]*A1
        //   #4 (0, 16, 32)  B1[0:7]*A0     <- ADDR_MOD_2 (srca cr->0, srcb cr+=16 lifts RWC_SrcB_Cr to 16)
        //   #5 (0, 24, 40)  B1[8:15]*A0
        //   #6 (16,16, 48)  B1[0:7]*A1     <- ADDR_MOD_3 (srca+=16, srcb cr->16)
        //   #7 (16,24, 56)  B1[8:15]*A1    <- matmul_op (ADDR_MOD_4) / matmul_op_last (ADDR_MOD_5)
        load_replay_buf<0, replay_buf_len>([] { _llk_math_matmul_emit_traversal_<true>(); });
    }
    else
    {
        // Tiny tiles select a window from this image and derive each transition from their geometry.
        // Transitions for full-tile input (32x32 x 32x32), one per face pair:
        //   B0A0 -> ADDR_MOD_1  srca+=16/32, srcb=0    (srca+=32 if transposed)
        //   B0A1 -> ADDR_MOD_2  srca=0,      srcb=32
        //   B2A0 -> ADDR_MOD_1  srca+=16/32, srcb=0
        //   B2A1 -> ADDR_MOD_3  srca=32/16,  srcb=16, dest=0
        //   B1A2 -> ADDR_MOD_1  srca+=16,    srcb=16
        //   B1A3 -> ADDR_MOD_2  srca=32,     srcb=48
        //   B3A2 -> ADDR_MOD_1  srca+=16,    srcb=0
        //   B3A3 -> matmul_op / matmul_op_last from the MOP
        // Within a pair the FPU walks the face one row band at a time under ADDR_MOD_0
        // (srca held, srcb+=band, dest+=band).
        load_replay_buf<0, replay_buf_len>([] { _llk_math_matmul_emit_traversal_<false>(); });
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

    // A face clamped to one dest row group has no matching window in the image just recorded, so
    // overwrite that window with a traversal at the group's cadence. Compiled away entirely where one
    // MVMUL covers a whole group.
    if constexpr (FPU_SPLITS_DEST_ROW_GROUP && !ENABLE_2X_FORMAT)
    {
        if (geometry.use_half_face_replay)
        {
            _llk_math_matmul_load_half_face_replay_(geometry);
        }
    }

    const std::uint32_t replay_start_idx = ENABLE_2X_FORMAT ? 0 : geometry.replay_start_idx;
    const std::uint32_t replay_len       = ENABLE_2X_FORMAT ? replay_buf_len : geometry.replay_len;
    LLK_ASSERT(replay_len > 0 && replay_start_idx + replay_len <= replay_buf_len, "matmul replay range exceeds the replay image");
    constexpr std::uint32_t phase_done_mvmul = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
    const std::uint32_t tile_done_mvmul      = reuse_a ? TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0) : TT_OP_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);

    ckernel_template temp(1, FIDELITY_PHASES, TT_OP_REPLAY(replay_start_idx, replay_len, 0, 0, 0, 0), phase_done_mvmul);
    temp.set_last_outer_loop_instr(tile_done_mvmul);
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/*
 * Direct-indexing operand addresses are in units of four dest rows, so one FPU row band steps the
 * SrcB and dest fields by this much. SrcA holds still across a face: the whole band walk multiplies
 * the same SrcA face.
 */
constexpr std::uint32_t MATMUL_DI_ADDR_UNIT_ROWS = 4;
constexpr std::uint32_t MATMUL_DI_BAND_STEP      = ELTWISE_MATH_ROWS / MATMUL_DI_ADDR_UNIT_ROWS;

static_assert(ELTWISE_MATH_ROWS % MATMUL_DI_ADDR_UNIT_ROWS == 0, "the direct-indexing address unit must divide the FPU row band");

// Base (srcb, srca, dest) of each face pair, in direct-indexing address units.
struct _llk_math_matmul_di_block_t
{
    std::uint8_t src_b;
    std::uint8_t src_a;
    std::uint8_t dest;
};

// K-outer traversal: [B0 B2 B1 B3] x [A0 A1 A2 A3], accumulating the second K face onto the first.
constexpr _llk_math_matmul_di_block_t MATMUL_DI_FULL_BLOCKS[MATMUL_FULL_FACE_PAIRS] = {
    {0x0, 0x0, 0x0}, // B0 x A0
    {0x0, 0x4, 0x4}, // B0 x A1
    {0x8, 0x0, 0x8}, // B2 x A0
    {0x8, 0x4, 0xC}, // B2 x A1
    {0x4, 0x8, 0x0}, // B1 x A2, accumulates onto B0 x A0
    {0x4, 0xC, 0x4}, // B1 x A3, accumulates onto B0 x A1
    {0xC, 0x8, 0x8}, // B3 x A2, accumulates onto B2 x A0
    {0xC, 0xC, 0xC}, // B3 x A3, accumulates onto B2 x A1
};

// 2x traversal: SrcA carries both sub-elements, so only B0/B1 against A0/A1 are visited.
constexpr _llk_math_matmul_di_block_t MATMUL_DI_2X_BLOCKS[MATMUL_2X_FACE_PAIRS] = {
    {0x0, 0x0, 0x0}, // B0 x A0
    {0x0, 0x4, 0x4}, // B0 x A1
    {0x4, 0x0, 0x8}, // B1 x A0
    {0x4, 0x4, 0xC}, // B1 x A1
};

template <bool ENABLE_2X_FORMAT>
inline constexpr const _llk_math_matmul_di_block_t* _llk_math_matmul_di_blocks_()
{
    return ENABLE_2X_FORMAT ? MATMUL_DI_2X_BLOCKS : MATMUL_DI_FULL_BLOCKS;
}

template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint32_t _llk_math_matmul_di_block_count_()
{
    return ENABLE_2X_FORMAT ? MATMUL_2X_FACE_PAIRS : MATMUL_FULL_FACE_PAIRS;
}

/**
 * @brief Returns the number of MVMULDIs recorded into the direct-indexing replay image.
 *
 * One short of the traversal: the closing MVMULDI is issued by the MOP.
 */
template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint32_t _llk_math_matmul_di_replay_buf_len_()
{
    return _llk_math_matmul_di_block_count_<ENABLE_2X_FORMAT>() * MATMUL_BANDS_PER_FACE - 1;
}

/**
 * @brief Records the direct-indexing MVMULDI image into replay buffer slot 0.
 *
 * Every face pair walks its face one FPU row band at a time, stepping SrcB and dest and holding
 * SrcA. The final band of the final pair is issued by the MOP as matmul_op / matmul_op_last.
 *
 * @tparam ENABLE_2X_FORMAT: Select the 2x traversal (half the face pairs) instead of the plain one.
 */
template <bool ENABLE_2X_FORMAT>
inline void _llk_math_matmul_di_load_replay_()
{
    constexpr std::uint32_t replay_buf_len = _llk_math_matmul_di_replay_buf_len_<ENABLE_2X_FORMAT>();
    constexpr std::uint32_t BLOCKS         = _llk_math_matmul_di_block_count_<ENABLE_2X_FORMAT>();

    load_replay_buf<0, replay_buf_len>(
        []
        {
#pragma GCC unroll 8
            for (std::uint32_t block = 0; block < BLOCKS; ++block)
            {
                const std::uint32_t src_b = _llk_math_matmul_di_blocks_<ENABLE_2X_FORMAT>()[block].src_b;
                const std::uint32_t src_a = _llk_math_matmul_di_blocks_<ENABLE_2X_FORMAT>()[block].src_a;
                const std::uint32_t dest  = _llk_math_matmul_di_blocks_<ENABLE_2X_FORMAT>()[block].dest;

#pragma GCC unroll 4
                for (const auto row : fpu_row_offsets<MAX_FACE_R_DIM>())
                {
                    // The traversal's final MVMULDI comes from the MOP, so stop one short.
                    if (block + 1 < BLOCKS || row + ELTWISE_MATH_ROWS < MAX_FACE_R_DIM)
                    {
                        const std::uint32_t step = row / MATMUL_DI_ADDR_UNIT_ROWS;
                        TTI_MVMULDI(p_setrwc::CLR_NONE, 0x0, src_b + step, src_a, 0x0, dest + step);
                    }
                }
            }
        });
}

/**
 * @brief Returns the operand addresses of the final MVMULDI, which the MOP issues outside the image.
 */
template <bool ENABLE_2X_FORMAT>
inline constexpr _llk_math_matmul_di_block_t _llk_math_matmul_di_final_()
{
    constexpr _llk_math_matmul_di_block_t BASE = _llk_math_matmul_di_blocks_<ENABLE_2X_FORMAT>()[_llk_math_matmul_di_block_count_<ENABLE_2X_FORMAT>() - 1];
    constexpr std::uint8_t LAST_BAND           = (MATMUL_BANDS_PER_FACE - 1) * MATMUL_DI_BAND_STEP;
    return {static_cast<std::uint8_t>(BASE.src_b + LAST_BAND), BASE.src_a, static_cast<std::uint8_t>(BASE.dest + LAST_BAND)};
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
    // Unpacker accounts for transposing if required.
    // if TRANSPOSE_EN = true, unpacker loads f0/f2/f1/f3
    // else, unpacker loads f0/f1/f2/f3
    // Math LLKs do not need any transpose handling for Quasar
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);
    const bool reuse_a                      = ct_dim >= rt_dim;

    constexpr std::uint32_t replay_buf_len = _llk_math_matmul_di_replay_buf_len_<ENABLE_2X_FORMAT>();

    _llk_math_matmul_di_load_replay_<ENABLE_2X_FORMAT>();

    // Closing MVMULDI of the traversal: the last row band of the last face pair.
    constexpr _llk_math_matmul_di_block_t FINAL = _llk_math_matmul_di_final_<ENABLE_2X_FORMAT>();

    constexpr static std::uint32_t matmul_op = TT_OP_MVMULDI(p_setrwc::CLR_NONE, 0x0, FINAL.src_b, FINAL.src_a, ADDR_MOD_1, FINAL.dest);
    const std::uint32_t matmul_op_last       = reuse_a ? TT_OP_MVMULDI(p_setrwc::CLR_A, 0x0, FINAL.src_b, FINAL.src_a, ADDR_MOD_2, FINAL.dest)
                                                       : TT_OP_MVMULDI(p_setrwc::CLR_B, 0x0, FINAL.src_b, FINAL.src_a, ADDR_MOD_2, FINAL.dest);

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

    // 2x has two implementations: a non-DI replay image, and direct indexing. Parts without the
    // non-DI image reach 2x through direct indexing, which every part has.
    constexpr bool USE_DIRECT_INDEXING = ENABLE_DIRECT_INDEXING || (ENABLE_2X_FORMAT && !ckernel::arch::has_mxfp4_2x_replay);

    if constexpr (USE_DIRECT_INDEXING)
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
