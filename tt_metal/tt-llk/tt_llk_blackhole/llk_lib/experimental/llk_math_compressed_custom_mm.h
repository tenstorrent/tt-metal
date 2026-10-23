// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_assert.h"

using namespace ckernel;
using namespace ckernel::math;

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32]
// rt_dim: 1
// ct_dim: any integer from 1 to 16
// kt_dim: even number from 2 to 256 (inclusive)
// fidelity: LoFi only
// throttle: not supported

// DEST rows from one output tile to the next: 32 with dense_packing (the two faces of a partial tile), 64 otherwise.
template <bool dense_packing>
constexpr std::uint32_t COMPRESSED_CUSTOM_MM_DEST_TILE_ROWS = dense_packing ? 32 : 64;

// The stride the init programmed into ADDR_MOD_2, so the execute can check its dense_packing matches (LLK_ASSERT, compiled
// out in release). 0 until an init has run.
static std::uint32_t compressed_custom_mm_dest_tile_rows = 0;

template <bool transpose, bool split_acc, bool dense_packing>
inline void _llk_math_compressed_custom_mm_addrmod_config_()
{
    constexpr std::uint8_t ADDR_MOD_0_SRCA_INCR  = transpose ? 32 : 16;
    constexpr std::uint8_t ADDR_MOD_1_SRCA_INCR  = transpose ? (64 - 16) : 16;
    constexpr std::uint16_t ADDR_MOD_1_DEST_INCR = split_acc ? (1024 - 8) : (1024 - 16);
    constexpr std::uint8_t ADDR_MOD_2_DEST_INCR  = COMPRESSED_CUSTOM_MM_DEST_TILE_ROWS<dense_packing>;

    addr_mod_t {
        .srca = {.incr = ADDR_MOD_0_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_0); // Move to the next face in width dimension, if transpose that is face 2 instead of face 1

    addr_mod_t {
        .srca = {.incr = ADDR_MOD_1_SRCA_INCR, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = ADDR_MOD_1_DEST_INCR, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_1); // Move to the next face in inner dimension, if transpose that is face 1 instead of face 2,
                          // if split_acc next inner dim is stored at rows 8 and 24 instead of 0 and 16

    addr_mod_t {
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = ADDR_MOD_2_DEST_INCR, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_2); // Move to the next tile in width dimension,
                          // if dense_packing next tile starts at row 32 instead of 64

    addr_mod_t {
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 1, .cr = 0},
    }
        .set(ADDR_MOD_3); // Move to the next inner dimension, resets everything

    addr_mod_t {
        .srca = {.incr = 0, .clr = 1, .cr = 0},
        .srcb = {.incr = 0, .clr = 1, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 1},
    }
        .set(ADDR_MOD_4); // Last MVMUL if finalization enabled, resets SrcA and SrcB,
                          // moves Dest to the beginning of the just multiplied tile

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 32, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_5); // Move SrcB to 32 in preparation for finalization ELWADD

    addr_mod_t {
        .srca = {.incr = 16, .clr = 0, .cr = 0},
        .srcb = {.incr = 16, .clr = 0, .cr = 0},
        .dest = {.incr = 16, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_6); // Move to next face in all three regs, used for final accumulation

    addr_mod_t {
        .srca = {.incr = 0, .clr = 0, .cr = 0},
        .srcb = {.incr = 0, .clr = 0, .cr = 0},
        .dest = {.incr = 0, .clr = 0, .cr = 0},
    }
        .set(ADDR_MOD_7); // NOP, has to be ADDR_MOD_7 to match with address mods configured in silu
                          // for fusing matmul and activation
}

// Zero-tile skip (see _llk_math_compressed_custom_mm_): step the DEST counter and its CR over the tile's slot, the rows
// ADDR_MOD_2 would add, without any Dst or Src access. ZEROACC is the one instruction that applies an address mod
// without writing data, but on Blackhole its 16-row mode is not a no-op for the out-of-range 0xff row select: measured
// on p100a, a skip at the tenth or later tile of a dense k-row (DEST counter >= 288) zeroed the last face of the current
// DEST half, with 16 densely packed tiles tile 15's second face, while earlier columns left DEST intact. That is
// consistent with the 16-row-clear bank-select rule documented in llk_math_eltwise_binary.h (the select flips once dest
// row offset + block index reaches 512; 32 * c + 0xff first does at c = 9). So the step is INCRWC on the DEST counter
// alone; its row increment is 4 bits, hence INCRWC_MAX_DEST_STEP-row steps and a remainder.
constexpr std::uint32_t INCRWC_MAX_DEST_STEP = 15;

template <std::uint32_t rows>
inline void _llk_math_compressed_custom_mm_skip_dest_()
{
    constexpr std::uint32_t step = rows < INCRWC_MAX_DEST_STEP ? rows : INCRWC_MAX_DEST_STEP;
    TTI_INCRWC(p_setrwc::CR_D, step, 0 /*rwc_b*/, 0 /*rwc_a*/);
    if constexpr (rows > step)
    {
        _llk_math_compressed_custom_mm_skip_dest_<rows - step>();
    }
}

inline void _llk_math_compressed_custom_mm_mop_config_(const std::uint32_t operandB_face_r_dim)
{
    const std::uint32_t replay_buf_len = operandB_face_r_dim == 8 ? 11 : 9;

    load_replay_buf(
        ckernel::math::replay_buf_offset,
        replay_buf_len,
        [operandB_face_r_dim]
        {
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F0 @ F0 => 0
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_1, 0); // F0 @ F1 (F2 if transpose) => 16
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_0, 0); // F1 @ F2 (F1 if transpose) => 0 (8 if split_acc)

            // Finalization phase
            TTI_MVMUL(p_setrwc::CLR_NONE, 0, ADDR_MOD_4, 0); // F1 @ F3 => 16 (24 if split_acc) (clear none,
                                                             // as both SrcA and SrcB banks are required for finalization)

            // We will exploit AddDst for finalization to calculate:
            // Dst = partial (in dest) + (split partial (moved to SrcB) + 0 (zeroed SrcA))
            // This way we avoid additional set of moves and move only one set of partials to SrcB
            // We have to move them to SrcB as we have to zero out the other Src register
            // and that can only be SrcA as it is not reused
            // Since we are going to keep reusing SrcB,
            // we have to move partials in lower half of SrcB as upper half contains inputs that are being reused

            // Move the 4 rows of the F0 split partial to rows 32+ of SrcB for finalization
            TTI_MOVD2B(0, 32, ADDR_MOD_5, p_movd2a::MOV_4_ROWS, 0 + 8);
            // Move the 4 rows of the F1 split partial to rows 48+ of SrcB for finalization
            TTI_MOVD2B(0, 16, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8);

            // Move lower 4 rows if they exist
            if (operandB_face_r_dim == 8)
            {
                TTI_MOVD2B(0, 0 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 0 + 8 + 4);
                TTI_MOVD2B(0, 16 + 4, ADDR_MOD_7, p_movd2a::MOV_4_ROWS, 16 + 8 + 4);
            }

            // Zero SrcA as it is not reused and we need to add zero to split partials in SrcB
            TTI_ZEROSRC(0, 1, 0, 1);

            // F0 partial in dest + F0 split partial in SrcB (row 32) => final F0 in dest
            TTI_ELWADD(0, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_6, 0);
            // F1 partial in dest + F1 split partial in SrcB (row 48) => final F1 in dest (clear only SrcA as B is reused)
            TTI_ELWADD(1, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_2, 0);
        });
}

template <bool transpose = false, bool split_acc = false, bool dense_packing = false>
inline void _llk_math_compressed_custom_mm_init_(const std::uint32_t operandB_face_r_dim)
{
    _llk_math_compressed_custom_mm_addrmod_config_<transpose, split_acc, dense_packing>();
    _llk_math_compressed_custom_mm_mop_config_(operandB_face_r_dim);
    compressed_custom_mm_dest_tile_rows = COMPRESSED_CUSTOM_MM_DEST_TILE_ROWS<dense_packing>;

    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Multiply a kt_dim x ct_dim block of compressed weight tiles (SrcA) by the activation tile (SrcB) into DEST,
 *        skipping the zero-code weight tiles the metadata marks.
 *
 * Per weight tile: a non-zero code replays the MVMUL sequence and steps DEST to the next tile slot (ADDR_MOD_2); a
 * zero code steps DEST the same distance with INCRWC and touches nothing else, since the unpacker emitted no SrcA for
 * it. The last tile of a k-row resets the counters instead (ADDR_MOD_3, or SETRWC after a zero code), and with
 * finalize the last k-row folds the split-accumulation partials.
 *
 * @tparam finalize: Run the split-accumulation finalize pass on the last k-row, values = <true/false>. Requires
 *                   split_acc at init.
 * @tparam dense_packing: The DEST tile stride a zero-tile skip steps over, values = <true/false>. Must equal the init's
 *                        dense_packing (asserted).
 * @param base_address_meta: Byte address of the per-tile metadata, read by the RISC-V (not L1_ADDRESS()); see
 *                           @ref _llk_unpack_AB_compressed_custom_mm_ for the layout.
 * @param operandB_face_r_dim: Activation rows per face, 1, 2, 4 or 8.
 * @param dst_index: DEST tile index of the first output tile.
 * @param kt_dim: Inner dimension in tiles, an even number from 2 to 256.
 * @param ct_dim: Output width in tiles, 1 to 16.
 * @note Call @ref _llk_math_compressed_custom_mm_init_ first, with the same dense_packing.
 * @note On the unpack thread, pair with @ref _llk_unpack_AB_compressed_custom_mm_, which reads the same metadata.
 */
template <bool finalize, bool dense_packing>
inline void _llk_math_compressed_custom_mm_(
    const std::uint32_t base_address_meta,
    const std::uint32_t operandB_face_r_dim,
    const std::uint32_t dst_index,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim = 1)
{
    // The multiplied tiles step DEST by the init's ADDR_MOD_2 increment and the skipped ones by this dense_packing;
    // the two have to agree or the tiles after a skip land one slot off.
    LLK_ASSERT(
        compressed_custom_mm_dest_tile_rows == COMPRESSED_CUSTOM_MM_DEST_TILE_ROWS<dense_packing>,
        "compressed_custom_mm: execute dense_packing differs from the init's");

    const std::uint32_t replay_buf_len = operandB_face_r_dim == 8 ? 11 : 9;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);

    const std::uint32_t iterations = finalize ? kt_dim - 1 : kt_dim;
    const std::uint32_t* meta_ptr  = reinterpret_cast<const std::uint32_t*>(base_address_meta);
    std::uint32_t index            = 0;
    std::uint32_t meta_index       = 0;
    std::uint32_t meta             = meta_ptr[meta_index] >> 3;

    for (std::uint32_t i = 0; i < iterations; i++)
    {
        for (std::uint32_t j = 0; j < ct_dim - 1; j++)
        {
            std::uint32_t fmt = meta & 0x3;
            if (fmt != 0)
            {
                lltt::replay(ckernel::math::replay_buf_offset, 3);
                TTI_MVMUL(p_setrwc::CLR_A, 0, ADDR_MOD_2, 0);
            }
            else
            {
                // Zero tile: nothing was unpacked for it, so only step DEST onto the next tile. That is the DEST part
                // of ADDR_MOD_2 alone, because the SrcA/SrcB counters are already 0 at every skip: whatever came
                // before left them so -- a multiplied tile (its last MVMUL applies ADDR_MOD_2, clr), another skip
                // (INCRWC leaves them untouched), a multiplied last-column tile (ADDR_MOD_3, clr), a zero last-column
                // tile (SETRWC SET_ABD) or, for the first tile, the init's reset_counters.
                _llk_math_compressed_custom_mm_skip_dest_<COMPRESSED_CUSTOM_MM_DEST_TILE_ROWS<dense_packing>>();
            }
            index++;
            if (index == 10)
            {
                index = 0;
                meta_index++;
                meta = meta_ptr[meta_index] >> 3;
            }
            else
            {
                meta >>= 3;
            }
        }
        std::uint32_t fmt = meta & 0x3;
        if (fmt != 0)
        {
            lltt::replay(ckernel::math::replay_buf_offset, 3);
            TTI_MVMUL(p_setrwc::CLR_AB, 0, ADDR_MOD_3, 0);
        }
        else
        {
            // Wait for math to drain first, so SRCB_VLD is tested on the bank the previous MVMUL's CLR_AB flipped to
            TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::MATH | ckernel::p_stall::SRCB_VLD);
            TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_ABD);
        }
        index++;
        if (index == 10)
        {
            index = 0;
            meta_index++;
            meta = meta_ptr[meta_index] >> 3;
        }
        else
        {
            meta >>= 3;
        }
    }

    if constexpr (finalize)
    {
        for (std::uint32_t j = 0; j < ct_dim - 1; j++)
        {
            std::uint32_t fmt = meta & 0x3;
            if (fmt != 0)
            {
                lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len);
            }
            else
            {
                TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::SRCA_VLD | 0);
                lltt::replay(ckernel::math::replay_buf_offset + 4, replay_buf_len - 4);
            }
            index++;
            if (index == 10)
            {
                index = 0;
                meta_index++;
                meta = meta_ptr[meta_index] >> 3;
            }
            else
            {
                meta >>= 3;
            }
        }
        std::uint32_t fmt = meta & 0x3;
        if (fmt != 0)
        {
            lltt::replay(ckernel::math::replay_buf_offset, replay_buf_len - 1);
            TTI_ELWADD(3, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
        }
        else
        {
            TTI_STALLWAIT(ckernel::p_stall::STALL_MATH, ckernel::p_stall::SRCA_VLD | ckernel::p_stall::SRCB_VLD);
            lltt::replay(ckernel::math::replay_buf_offset + 4, replay_buf_len - 5);
            TTI_ELWADD(3, 1, p_elwise::SRCB_NO_BCAST, ADDR_MOD_3, 0);
        }
    }
}
