// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "ckernel_vector.h"
#include "experimental/llk_compressed_custom_mm_common.h"
#include "experimental/llk_unpack_compressed_custom_mm.h"
#include "llk_unpack_common.h"
#include "llk_unpack_matmul.h"
#include "tt_t6_trisc_map.h"

#ifndef __riscv_vector
#error "llk_unpack_face_compressed_custom_mm.h pushes the ARG_FIFO with vector stores: build the unpack thread (TRISC0) with the vector extension"
#endif

using namespace ckernel;

/*************************************************************************
 * LLK UNPACK FACE COMPRESSED CUSTOM MM
 *************************************************************************/

// The face-granular compressed matmul: every 16x16 face of a weight tile has its own MX format, and all-zero faces
// are not stored (contract in llk_compressed_custom_mm_common.h). Same unpacker roles as the tile version in
// llk_unpack_compressed_custom_mm.h: Input0 (activation) -> SrcB through UNPACR1 and the MOP, Input1 (weights) ->
// SrcA through UNPACR0 with its ARG_FIFO in mode 1, one {format, address} entry per UNPACR0.
//
// The in1 buffer descriptor describes a single 16x16 face, so one UNPACR0_TILE moves one face. A weight tile is
// four of them into SrcA face slots 0..3 (Dst_Tile_Idx), the last one setting dvalid, and math sees an ordinary
// four-face tile: the face version shares the tile version's math thread. A zero face is unpacked like any other,
// from an all-zero face the host keeps in the stream, so the loop never looks at the metadata.
//
// Blackhole keeps one address counter per format (three, borrowed from the math and pack threads), sorts the faces
// into per-format piles and lets the counters walk them. Quasar has no spare unpacker counters, but needs none: the
// host places every face, writes its address and format into the metadata, and the ARG_FIFO hands them to each
// UNPACR0. Faces stay in matrix order at their own size, and any number of formats costs the same.
//
// The ARG_FIFO is 128 bits wide, eight lines of four 32-bit entries, and UNPACR0 pops them one at a time. A 16 B
// store at the base of the mailbox window pushes one full line, so a tile's four entries go in with one vector
// store from TRISC0's vector unit (the vle32/vadd/vse32 below; TRISC0 is the only TRISC with one). A group of
// eight tiles is then one vector load of its 32 metadata words, one vector add of the stream base and eight
// stores, which keeps the RISC-V, the bottleneck of the tile version, well under the unpacker's time per tile.
//
// Replay buffer layout on this thread:
//   [0, 4)       UNPACR0_TILE x4: the four faces of one weight tile -> SrcA, address and format from the ARG_FIFO
//   [4, 4 + len) one Input0 tile -> SrcB, recorded by _llk_unpack_matmul_load_tile_replay_ and run by the MOP

constexpr std::uint32_t COMPRESSED_FACE_MM_UNPACK_SRC_A_REPLAY_IDX = 0;
constexpr std::uint32_t COMPRESSED_FACE_MM_UNPACK_SRC_B_REPLAY_IDX = NUM_FACES;

// Tiles per group: one FIFO line per tile, so the FIFO depth in lines.
constexpr std::uint32_t COMPRESSED_FACE_MM_GROUP = COMPRESSED_MM_ARG_FIFO_DEPTH;

// A 16 B store here pushes one line of four entries into the UNPACR0 argument FIFO (the tile version's
// unpack0_arg_fifo_push is the one-entry line at +0x30 of the same window).
std::uint32_t* const unpack0_arg_fifo_push_line = reinterpret_cast<std::uint32_t*>(UNPACKER_ARG_MAILBOX_BASE0);

/**
 * @brief Initializes the unpacker for the face compressed matrix multiply: Input0 -> SrcB, compressed Input1 -> SrcA.
 *
 * Records the replay buffer described above and programs the MOP to unpack one Input0 tile into SrcB.
 *
 * @param buf_desc_id_0: Buffer descriptor ID for Input 0/SrcB, values = 0 - 16.
 * @param buf_desc_id_1: Buffer descriptor ID for Input 1/SrcA, values = 0 - 16. Only its tile dimensions are
 *        used and they must describe a single 16x16 face; the ARG_FIFO supplies each face's address and format.
 * @param src_b_shape: Input 0/SrcB tile shape, [{1, 2, 4, 8}, 32].
 * @note On the math thread, pair with @ref _llk_math_compressed_custom_mm_init_ (T1); on the pack thread, pair
 *       with @ref _llk_pack_matmul_init_ (T2).
 * @note @ref _llk_unpack_face_compressed_custom_mm_ is the matching execute call on this thread; call
 *       @ref _llk_unpack_face_compressed_custom_mm_uninit_ after it to switch the ARG_FIFO back off.
 */
inline void _llk_unpack_face_compressed_custom_mm_init_(const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape src_b_shape)
{
    LLK_ASSERT(_llk_compressed_custom_mm_is_valid_src_b_shape_(src_b_shape), "compressed_custom_mm supports in0 tile shapes [{1, 2, 4, 8}, 32] only");

    _llk_unpack_compressed_custom_mm_arg_fifo_cfg_(true);
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);

    load_replay_buf<COMPRESSED_FACE_MM_UNPACK_SRC_A_REPLAY_IDX, NUM_FACES>(
        [buf_desc_id_1]
        {
            // Src_Tile_Idx is ignored in ARG_FIFO mode 1: the entry holds the absolute face address. The
            // descriptor is one face, so Dst_Tile_Idx counts SrcA face slots.
            for (std::uint32_t face = 0; face < NUM_FACES; face++)
            {
                TT_UNPACR0_TILE(face /*Dst_Tile_Idx*/, 0 /*Src_Tile_Idx*/, buf_desc_id_1, face == NUM_FACES - 1 /*SetDatValid*/);
            }
        });

    // Advance the SrcB source counter past the tile, so consecutive runs walk the Input0 k-tiles.
    const std::uint32_t src_b_replay_len = _llk_unpack_matmul_load_tile_replay_<p_unpacr::UNP_B>(
        COMPRESSED_FACE_MM_UNPACK_SRC_B_REPLAY_IDX, buf_desc_id_0, src_b_shape, true /*advance_to_next_tile*/);

    ckernel_template temp(1 /*outer loop*/, 1 /*inner loop*/, TT_OP_REPLAY(COMPRESSED_FACE_MM_UNPACK_SRC_B_REPLAY_IDX, src_b_replay_len, 0, 0, 0, 0));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Unpacks one weight tile whose four entries sit in vector register V: the SrcB unpack of its Input0 tile
 *        first when the tile starts a k-row, then the four UNPACR0s and the push of their FIFO line.
 *
 * The row counter travels by value: anything the loop keeps in memory is reloaded after every vector store (its
 * asm clobbers memory), which is why this is not a lambda capturing it by reference.
 *
 * @tparam V: Vector register holding the tile's four metadata entries, already offset by the stream base.
 * @param row_left: Weight tiles left in the current k-row before this one; 0 means this tile starts a row.
 * @param ct_dim: Number of tiles in the column dimension of Input 1.
 * @return row_left for the next tile.
 */
template <vreg V>
__attribute__((always_inline)) inline std::uint32_t _llk_unpack_face_compressed_custom_mm_tile_(std::uint32_t row_left, const std::uint32_t ct_dim)
{
    if (row_left == 0)
    {
        ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        row_left = ct_dim;
    }
    // The four UNPACR0s are issued before their line is pushed: they block in the unpacker until it arrives.
    TTI_REPLAY(COMPRESSED_FACE_MM_UNPACK_SRC_A_REPLAY_IDX, NUM_FACES, 0, 0, 0, 0);
    vector_store<V>(unpack0_arg_fifo_push_line);
    return row_left - 1;
}

/**
 * @brief Unpacks a [1, kt_dim] row of Input0 tiles into SrcB and a [kt_dim, ct_dim] block of face-compressed
 *        Input1 tiles into SrcA.
 *
 * For each k, unpacks Input0 tile k into SrcB once, then every weight tile (k, 0..ct_dim-1) into SrcA, face by
 * face from the addresses in its metadata.
 *
 * @param base_address_in1: L1 address of the compressed weight stream, in 16 B units (L1_ADDRESS()).
 * @param base_address_meta: Byte address of the kt_dim * ct_dim * 4 metadata words. Not L1_ADDRESS(): the
 *        RISC-V dereferences it.
 * @param kt_dim: Number of tiles in the common dimension, values >= 1.
 * @param ct_dim: Number of tiles in the column dimension of Input 1, values = 1 - 16.
 * @note Call @ref _llk_unpack_face_compressed_custom_mm_init_ before this function.
 * @note On the math thread, pair with @ref _llk_math_compressed_custom_mm_ with the same kt_dim and ct_dim.
 */
inline void _llk_unpack_face_compressed_custom_mm_(
    const std::uint32_t base_address_in1, const std::uint32_t base_address_meta, const std::uint32_t kt_dim, const std::uint32_t ct_dim)
{
    const std::uint32_t* meta     = reinterpret_cast<const std::uint32_t*>(base_address_meta);
    const std::uint32_t num_tiles = kt_dim * ct_dim;
    const std::int32_t base       = static_cast<std::int32_t>(base_address_in1);

    // Input0 tiles start at the base of their buffer descriptor; the SrcA counters stay at 0.
    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

    // Tiles go in groups of COMPRESSED_FACE_MM_GROUP, one FIFO line each, so one wait for an empty FIFO at the
    // start of a group replaces per-push accounting and also drains whatever a previous call left behind. The
    // group's 32 metadata words are one vector load into v0-v7 (LMUL 8) issued before the wait, so its latency
    // overlaps the poll; after the add of the stream base, each v register holds one tile's line.
    constexpr std::uint32_t GROUP       = COMPRESSED_FACE_MM_GROUP;
    constexpr std::uint32_t GROUP_WORDS = GROUP * COMPRESSED_FACE_MM_META_WORDS_PER_TILE;
    std::uint32_t row_left              = 0; // the first tile starts a k-row
    std::uint32_t i                     = 0;
    for (; i + GROUP <= num_tiles; i += GROUP)
    {
        (void)vsetvl<E32, M8>(GROUP_WORDS);
        vector_load<V0>(meta + i * COMPRESSED_FACE_MM_META_WORDS_PER_TILE);
        _llk_unpack_compressed_custom_mm_wait_arg_fifo_empty_();
        // The relative offsets in [19:0] cannot carry into the format field: L1 is 4 MB, 2^18 16 B units.
        vector_add<V0, V0>(base);
        (void)vsetvl<E32, M1, COMPRESSED_FACE_MM_META_WORDS_PER_TILE>();
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V0>(row_left, ct_dim);
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V1>(row_left, ct_dim);
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V2>(row_left, ct_dim);
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V3>(row_left, ct_dim);
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V4>(row_left, ct_dim);
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V5>(row_left, ct_dim);
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V6>(row_left, ct_dim);
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V7>(row_left, ct_dim);
    }
    if (i < num_tiles)
    {
        // The tail is a short group: one vector load of its tail * 4 words, so no tile waits on its own load.
        const std::uint32_t tail = num_tiles - i;
        (void)vsetvl<E32, M8>(tail * COMPRESSED_FACE_MM_META_WORDS_PER_TILE);
        vector_load<V0>(meta + i * COMPRESSED_FACE_MM_META_WORDS_PER_TILE);
        _llk_unpack_compressed_custom_mm_wait_arg_fifo_empty_();
        vector_add<V0, V0>(base);
        (void)vsetvl<E32, M1, COMPRESSED_FACE_MM_META_WORDS_PER_TILE>();
        row_left = _llk_unpack_face_compressed_custom_mm_tile_<V0>(row_left, ct_dim);
        if (tail > 1)
        {
            row_left = _llk_unpack_face_compressed_custom_mm_tile_<V1>(row_left, ct_dim);
        }
        if (tail > 2)
        {
            row_left = _llk_unpack_face_compressed_custom_mm_tile_<V2>(row_left, ct_dim);
        }
        if (tail > 3)
        {
            row_left = _llk_unpack_face_compressed_custom_mm_tile_<V3>(row_left, ct_dim);
        }
        if (tail > 4)
        {
            row_left = _llk_unpack_face_compressed_custom_mm_tile_<V4>(row_left, ct_dim);
        }
        if (tail > 5)
        {
            row_left = _llk_unpack_face_compressed_custom_mm_tile_<V5>(row_left, ct_dim);
        }
        if (tail > 6)
        {
            row_left = _llk_unpack_face_compressed_custom_mm_tile_<V6>(row_left, ct_dim);
        }
    }
}

/**
 * @brief Switches the UNPACR0 argument FIFO back off, so later ops unpack from their buffer descriptors again.
 *
 * @note Call after the last @ref _llk_unpack_face_compressed_custom_mm_ of the op.
 */
inline void _llk_unpack_face_compressed_custom_mm_uninit_()
{
    _llk_unpack_compressed_custom_mm_uninit_();
}
