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

struct _llk_math_matmul_geometry_t
{
    TensorShape output_shape;
    std::uint8_t replay_start;
    std::uint8_t replay_length;
    std::uint16_t dst_rows_per_tile;
    std::uint8_t face_rows;
    std::uint8_t k_faces;
    std::uint16_t next_tile_rows;
};

/**
 * @brief Derives output layout, replay window, and destination strides for MOP-based matmul.
 *
 * @param ct_dim: Number of column tiles in the output block.
 * @param rt_dim: Number of row tiles in the output block.
 * @param src_b_shape: Input 0/SrcB tile shape.
 * @param src_a_shape: Input 1/SrcA tile shape.
 */
inline _llk_math_matmul_geometry_t _llk_math_matmul_geometry_(
    const std::uint8_t ct_dim, const std::uint8_t rt_dim, const TensorShape& src_b_shape, const TensorShape& src_a_shape)
{
    LLK_ASSERT(validate_matmul_tensor_shapes_(src_b_shape, src_a_shape), "unsupported SrcB/SrcA TensorShape pair for matmul");

    const TensorShape output_shape =
        make_tensor_shape(src_b_shape.face_r_dim, src_a_shape.face_c_dim, src_b_shape.num_faces_r_dim, src_a_shape.num_faces_c_dim);
    const std::uint8_t face_rows          = src_b_shape.face_r_dim < MAX_FPU_ROWS ? MAX_FPU_ROWS : src_b_shape.face_r_dim;
    const std::uint16_t dst_rows_per_tile = static_cast<std::uint16_t>(output_shape.total_num_faces()) * face_rows;
    const std::uint8_t k_faces            = src_b_shape.num_faces_c_dim;
    const std::uint8_t face_row_passes    = src_b_shape.face_r_dim > MAX_FPU_ROWS ? 2 : 1;
    const std::uint8_t mvmuls_per_face    = output_shape.total_num_faces() * face_row_passes;

    return _llk_math_matmul_geometry_t {
        .output_shape      = output_shape,
        .replay_start      = static_cast<std::uint8_t>(8 - mvmuls_per_face),
        .replay_length     = static_cast<std::uint8_t>(mvmuls_per_face * k_faces - 1),
        .dst_rows_per_tile = dst_rows_per_tile,
        .face_rows         = face_rows,
        .k_faces           = k_faces,
        .next_tile_rows    = static_cast<std::uint16_t>((ct_dim >= rt_dim ? 1 : ct_dim) * dst_rows_per_tile),
    };
}

/**
 * @brief Initializes addrmod for matrix multiply operation.
 *
 * Non-2x matmul derives compacted X/Y/Z/K transitions from geometry; 2x retains its fixed layout.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, programs addr_mods for the MXFP4_2x non-DI MOP variant (8 MVMULs covering only A0/A1 and B0/B1; SrcA in MxFp4_2x_A/B
 * drives the 2x sub-element expansion).
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @param geometry: MOP replay and destination geometry from @ref _llk_math_matmul_geometry_.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_addrmod_(const std::uint8_t ct_dim, const std::uint8_t rt_dim, const _llk_math_matmul_geometry_t& geometry)
{
    constexpr bool high_fidelity      = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT  = high_fidelity ? 1 : 0;

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

    const bool row_pass_axis            = geometry.face_rows > MAX_FPU_ROWS;
    const bool column_axis              = geometry.output_shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM;
    const bool row_axis                 = geometry.output_shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM;
    const std::int32_t src_b_row_stride = geometry.face_rows * geometry.k_faces;

    // Compact the active output axes in 8-row-pass, column, row order.
    const std::int32_t x_src_a = row_pass_axis ? 0 : (column_axis ? MAX_FACE_C_DIM : 0);
    const std::int32_t x_src_b = row_pass_axis ? MAX_FPU_ROWS : (column_axis ? 0 : (row_axis ? src_b_row_stride : 0));
    const std::int32_t x_dest =
        row_pass_axis ? MAX_FPU_ROWS : (column_axis ? geometry.face_rows : (row_axis ? geometry.face_rows * geometry.output_shape.num_faces_c_dim : 0));

    const bool y_is_column          = row_pass_axis && column_axis;
    const bool y_is_row             = !y_is_column && row_axis && (row_pass_axis || column_axis);
    const bool has_y                = y_is_column || y_is_row;
    const std::int32_t y_axis_src_a = y_is_column ? MAX_FACE_C_DIM : 0;
    const std::int32_t y_axis_src_b = y_is_row ? src_b_row_stride : 0;
    const std::int32_t y_axis_dest  = y_is_column ? geometry.face_rows : (y_is_row ? geometry.face_rows * geometry.output_shape.num_faces_c_dim : 0);
    const std::int32_t y_dest       = has_y ? y_axis_dest - x_dest : 0;

    const bool has_z          = row_pass_axis && column_axis && row_axis;
    const std::int32_t z_dest = has_z ? geometry.face_rows * geometry.output_shape.num_faces_c_dim - MAX_FPU_ROWS - geometry.face_rows : 0;

    const bool has_k           = geometry.k_faces > 1;
    const std::int32_t k_src_a = has_k ? MAX_FACE_C_DIM * geometry.output_shape.num_faces_c_dim : 0;
    const std::int32_t k_src_b = has_k ? geometry.face_rows - (row_axis ? src_b_row_stride : 0) : 0;

    // advance the first in-replay output axis.
    addr_mod_t {
        .srca = {.incr = static_cast<std::uint8_t>(x_src_a), .clr = 0, .cr = 0},
        .srcb = {.incr = static_cast<std::uint8_t>(x_src_b), .clr = 0, .cr = 0},
        .dest = {.incr = static_cast<std::uint16_t>(x_dest), .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0);

    // advance the second output axis (column if present, otherwise row); SrcB CR discards the X step.
    addr_mod_t {
        .srca = {.incr = static_cast<std::uint8_t>(y_axis_src_a), .clr = 0, .cr = 0},
        .srcb = {.incr = static_cast<std::uint8_t>(y_axis_src_b), .clr = 0, .cr = has_y},
        .dest = {.incr = static_cast<std::uint16_t>(y_dest), .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1);

    // rewind SrcA's column face and advance SrcB's CR to the next row face.
    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = has_z},
        .srcb = {.incr = static_cast<std::uint8_t>(has_z ? src_b_row_stride : 0), .clr = 0, .cr = has_z},
        .dest = {.incr = static_cast<std::uint16_t>(z_dest), .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_2);

    // rewind output axes through CR and advance both sources to the next K face.
    addr_mod_t {
        .srca = {.incr = static_cast<std::uint8_t>(k_src_a), .clr = 0, .cr = has_k},
        .srcb = {.incr = static_cast<std::uint8_t>(k_src_b), .clr = 0, .cr = has_k},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_3);

    // End a fidelity phase.
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = 0, .clr = 0, .cr = 1},
        .fidelity = {.incr = FIDELITY_INCREMENT, .clr = 0},
    }
        .set(ADDR_MOD_4);

    // End the tile.
    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = geometry.next_tile_rows, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_5);
}

/**
 * @brief Programs the legacy full-tile addr-mod layout used by no-MOP matmul.
 *
 * The 2x path delegates to the equivalent full-tile geometry setup.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_addrmod_(const std::uint8_t ct_dim, const std::uint8_t rt_dim)
{
    if constexpr (ENABLE_2X_FORMAT)
    {
        const _llk_math_matmul_geometry_t geometry = _llk_math_matmul_geometry_(ct_dim, rt_dim, DEFAULT_TENSOR_SHAPE, DEFAULT_TENSOR_SHAPE);
        _llk_math_matmul_addrmod_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim, geometry);
        return;
    }

    constexpr bool high_fidelity      = MATH_FIDELITY_TYPE != ckernel::MathFidelity::LoFi;
    constexpr int FIDELITY_INCREMENT  = high_fidelity ? 1 : 0;
    const std::uint16_t num_tile_incr = (ct_dim >= rt_dim) ? 64 : ct_dim * 64;

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

    addr_mod_t {
        .srca     = {.incr = 0, .clr = 1, .cr = 0},
        .srcb     = {.incr = 0, .clr = 1, .cr = 0},
        .dest     = {.incr = num_tile_incr, .clr = 0, .cr = 1},
        .fidelity = {.incr = 0, .clr = 1},
    }
        .set(ADDR_MOD_3);

    addr_mod_t {
        .srca = {.incr = 32, .clr = 0, .cr = 1},
        .srcb = {.incr = 48, .clr = 0, .cr = 1},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_4);

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
 * @brief Number of MVMULs recorded into the matmul replay image.
 *
 * One less than the total MVMUL count: the closing MVMUL of the Tile x Tile operation is issued from
 * outside the replay buffer by the MOP in @ref _llk_math_matmul_mop_config_, or directly from the
 * RISC core in the experimental no-MOP path.
 *
 * @tparam ENABLE_2X_FORMAT: Select the MXFP4_2x replay image.
 */
template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint32_t _llk_math_matmul_replay_buf_len_()
{
    return ENABLE_2X_FORMAT ? 7 : 15;
}

// Addr-mod slot used by the no-MOP fidelity-phase closing MVMUL.
template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint8_t _llk_math_matmul_op_addr_mod_()
{
    return ENABLE_2X_FORMAT ? ADDR_MOD_4 : ADDR_MOD_5;
}

// Addr-mod slot used by the no-MOP tile-closing MVMUL.
template <bool ENABLE_2X_FORMAT>
inline constexpr std::uint8_t _llk_math_matmul_op_last_addr_mod_()
{
    return ENABLE_2X_FORMAT ? ADDR_MOD_5 : ADDR_MOD_3;
}

/**
 * @brief Records the standard or 2x MVMUL image into replay buffer slot 0.
 *
 * Standard matmul always records the full 15-entry K-outer image. MOP-based tiny matmul selects a
 * window from that image.
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
            []
            {
                // Tiny tiles select a window and derive each transition from their geometry.
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #0: next 8-row destination block
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // #1: next column/row; reset SrcB
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #2: next 8-row destination block
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // #3: next row; reset SrcA column, advance SrcB row
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #4: next 8-row destination block
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // #5: next column/row; reset SrcB
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #6: next 8-row destination block
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_3, 0); // #7: restart output; advance both sources
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #8: next 8-row destination block
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // #9: next column/row; reset SrcB
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #10: next 8-row destination block
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_2, 0); // #11: next row; reset SrcA column, advance SrcB row
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #12: next 8-row destination block
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // #13: next column/row; reset SrcB
                TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // #14: final 8-row destination block
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
 * with the tile completion. Standard matmul selects a geometry-dependent window from the K-outer image;
 * MXFP4_2x replays its full image.
 *
 * @tparam MATH_FIDELITY_TYPE: Controls multiplication precision via the number of FPU fidelity phases; higher values use more of the input mantissa bits,
 * values = <LoFi/HiFi2/HiFi3/HiFi4>
 * @tparam ENABLE_2X_FORMAT: When true, emits the non-DI MXFP4_2x variant (7-MVMUL replay traversing only A0/A1 and B0/B1; relies on SrcA being unpacked as
 * MxFp4_2x_A/B for the 2x sub-element expansion).
 * @param ct_dim: Number of tiles in the column dimension for a matrix multiply
 * @param rt_dim: Number of tiles in the row dimension for a matrix multiply
 * @param geometry: Replay window and destination strides from @ref _llk_math_matmul_geometry_.
 */
template <ckernel::MathFidelity MATH_FIDELITY_TYPE, bool ENABLE_2X_FORMAT = false>
inline void _llk_math_matmul_mop_config_(const std::uint8_t ct_dim, const std::uint8_t rt_dim, const _llk_math_matmul_geometry_t& geometry)
{
    constexpr std::uint32_t FIDELITY_PHASES = MATH_FIDELITY_TYPE == ckernel::MathFidelity::LoFi ? 1 : to_underlying(MATH_FIDELITY_TYPE);

    const bool reuse_a = ct_dim >= rt_dim;

    constexpr std::uint32_t replay_buf_len = _llk_math_matmul_replay_buf_len_<ENABLE_2X_FORMAT>();

    _llk_math_matmul_load_replay_<ENABLE_2X_FORMAT>();

    const std::uint32_t replay_start  = ENABLE_2X_FORMAT ? 0 : geometry.replay_start;
    const std::uint32_t replay_length = ENABLE_2X_FORMAT ? replay_buf_len : geometry.replay_length;
    LLK_ASSERT(replay_length > 0 && replay_start + replay_length <= replay_buf_len, "matmul replay range exceeds the replay image");
    constexpr std::uint32_t phase_done_mvmul = TT_OP_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0);
    const std::uint32_t tile_done_mvmul      = reuse_a ? TT_OP_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_5, 0) : TT_OP_MVMUL(p_setrwc::CLR_B, 0, ADDR_MOD_5, 0);

    ckernel_template temp(1, FIDELITY_PHASES, TT_OP_REPLAY(replay_start, replay_length, 0, 0, 0, 0), phase_done_mvmul);
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
 * Standard MOP matmul supports the validated tiny-tile pairs; direct-indexing and 2x remain full-tile only.
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
    const TensorShape& src_b_shape = DEFAULT_TENSOR_SHAPE,
    const TensorShape& src_a_shape = DEFAULT_TENSOR_SHAPE)
{
    if constexpr (ENABLE_DIRECT_INDEXING || ENABLE_2X_FORMAT)
    {
        LLK_ASSERT(
            src_b_shape.face_r_dim == MAX_FACE_R_DIM && src_b_shape.total_num_faces() == MAX_NUM_FACES && src_a_shape.face_r_dim == MAX_FACE_R_DIM &&
                src_a_shape.total_num_faces() == MAX_NUM_FACES,
            "direct-indexing and 2x matmul support exact 16x16-face, 2x2 operand shapes only");
    }

    if constexpr (ENABLE_DIRECT_INDEXING)
    {
        // Direct-indexing path. Supports plain DI and DI+X2 (DI+X2 is the original
        // MXFP4_2x matmul implementation).
        _llk_math_matmul_di_addrmod_<MATH_FIDELITY_TYPE>(ct_dim, rt_dim);
        _llk_math_matmul_di_mop_config_<MATH_FIDELITY_TYPE, ENABLE_2X_FORMAT>(ct_dim, rt_dim);
        _set_tile_shape_idx_gpr_(MAX_NUM_FACES * MAX_FACE_R_DIM);
    }
    else
    {
        const _llk_math_matmul_geometry_t geometry = _llk_math_matmul_geometry_(ct_dim, rt_dim, src_b_shape, src_a_shape);
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

    const bool reuse_a          = ct_dim >= rt_dim;
    const std::uint32_t t_dim   = reuse_a ? rt_dim : ct_dim;
    const std::uint32_t rut_dim = reuse_a ? ct_dim : rt_dim; // reuse-dim
    const bool strided_dest           = !reuse_a && ct_dim >= 2;
    const std::uint32_t dst_tile_rows = strided_dest ? 1U << ckernel::regfile[p_gpr_math::TILE_SHAPE_IDX] : 0;

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
            TT_SETRWC(p_setrwc::CLR_NONE, 0, dst_tile_rows * (t + 1), p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::C_TO_CR_MODE, 0, p_setrwc::SET_D);
        }
    }
    _reset_counters_<p_setrwc::SET_ABD_F>();
}
