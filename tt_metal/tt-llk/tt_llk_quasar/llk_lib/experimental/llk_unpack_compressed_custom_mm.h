// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_template.h"
#include "ckernel_trisc_common.h"
#include "experimental/llk_compressed_custom_mm_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_matmul.h"
#include "tt_t6_trisc_map.h"

using namespace ckernel;

/*************************************************************************
 * LLK UNPACK COMPRESSED CUSTOM MM
 *************************************************************************/

// Input0 (activation) -> SrcB through UNPACR1, Input1 (weights) -> SrcA through UNPACR0, as in llk_unpack_matmul.h.
//
// The weight tiles change format from tile to tile, so UNPACR0 runs with its argument FIFO (ARG_FIFO) in mode 1:
// before each UNPACR0_TILE the RISC-V pushes one entry {tile_fmt[27:20], tile_addr[19:0]} and the unpacker takes
// that tile's L1 address and format from the entry instead of from the buffer descriptor. The descriptor still
// supplies the tile dimensions, and the unpacker blocks until an entry is available. Zero tiles push nothing and
// read nothing from L1: an UNPACR_NOP clears a SrcA bank to zero for them instead, so the math thread multiplies
// every tile and never has to read the metadata.
//
// The unpack thread's FIFO is the vector-width one: 8 lines of up to four 32-bit entries, and every UNPACR0_TILE
// pops one entry. A store at byte offset 16 * n of the mailbox window pushes a line of 4 - n entries, all holding
// the stored word, so a plain 32-bit store must go to offset 0x30 to push exactly one. Stores past offset 0x40
// clear the FIFO, and loads return its empty flag.
//
// Replay buffer layout on this thread:
//   [0]          UNPACR0_TILE: one weight tile -> SrcA, address and format from the ARG_FIFO
//   [1, 1 + len) one Input0 tile -> SrcB, recorded by _llk_unpack_matmul_load_tile_replay_ and run by the MOP

constexpr std::uint32_t COMPRESSED_MM_UNPACK_SRC_A_REPLAY_IDX = 0;
constexpr std::uint32_t COMPRESSED_MM_UNPACK_SRC_B_REPLAY_IDX = 1;

// FIFO lines. Pushing into a full FIFO is not safe to rely on, so the execute loop never has more in flight.
constexpr std::uint32_t COMPRESSED_MM_ARG_FIFO_DEPTH = 8;

// Pushes a single-entry line into the UNPACR0 argument FIFO. Each thread has its own FIFO behind the same window.
volatile std::uint32_t* const unpack0_arg_fifo_push = reinterpret_cast<volatile std::uint32_t*>(UNPACKER_ARG_MAILBOX_BASE0 + 0x30);
// Reads 1 once the UNPACR0 argument FIFO is empty, that is, once the unpacker has taken every pushed entry.
volatile std::uint32_t* const unpack0_arg_fifo_status = reinterpret_cast<volatile std::uint32_t*>(UNPACKER_ARG_MAILBOX_BASE0 + 0x80);

/**
 * @brief Spins on the RISC-V until the unpacker has popped every entry pushed into the UNPACR0 argument FIFO.
 *
 * @note Only pushed entries whose UNPACR0_TILE has already been issued get popped, so issue those first.
 */
inline void _llk_unpack_compressed_custom_mm_wait_arg_fifo_empty_()
{
    while (*unpack0_arg_fifo_status == 0)
    {
    }
}

/**
 * @brief Switches the UNPACR0 argument FIFO on in mode 1 (per-instruction L1 address and format), or back off.
 *
 * @param enable: true for the compressed matmul (init); false restores descriptor-driven unpacking (uninit).
 */
inline void _llk_unpack_compressed_custom_mm_arg_fifo_cfg_(const bool enable)
{
    // UNPACR0 instructions issued before this point must finish under the mode they were issued in.
    TTI_STALLWAIT(p_stall::STALL_CFG, 0, 0, p_stall::UNPACK0);
    cfg_rmw(THCON_UNPACKER0_REG0_ENABLE_ARG_FIFO_RMW, enable ? 1 : 0);
    cfg_rmw(THCON_UNPACKER0_REG0_ARG_FIFO_MODE_RMW, enable ? 1 : 0);
}

/**
 * @brief Initializes the unpacker for compressed matrix multiply: Input0 -> SrcB, compressed Input1 -> SrcA.
 *
 * Records the replay buffer described above and programs the MOP to unpack one Input0 tile into SrcB.
 *
 * @param buf_desc_id_0: Buffer descriptor ID for Input 0/SrcB, values = 0 - 16.
 * @param buf_desc_id_1: Buffer descriptor ID for Input 1/SrcA, values = 0 - 16. Only its tile dimensions are
 *        used and they must describe a full 32x32 tile; the ARG_FIFO supplies each tile's address and format.
 * @param src_b_shape: Input 0/SrcB tile shape, [{1, 2, 4, 8}, 32].
 * @note On the math thread, pair with @ref _llk_math_compressed_custom_mm_init_ (T1); on the pack thread, pair
 *       with @ref _llk_pack_matmul_init_ (T2).
 * @note @ref _llk_unpack_compressed_custom_mm_ is the matching execute call on this thread; call
 *       @ref _llk_unpack_compressed_custom_mm_uninit_ after it to switch the ARG_FIFO back off.
 */
inline void _llk_unpack_compressed_custom_mm_init_(const std::uint32_t buf_desc_id_0, const std::uint32_t buf_desc_id_1, const TensorShape src_b_shape)
{
    LLK_ASSERT(_llk_compressed_custom_mm_is_valid_src_b_shape_(src_b_shape), "compressed_custom_mm supports in0 tile shapes [{1, 2, 4, 8}, 32] only");

    _llk_unpack_compressed_custom_mm_arg_fifo_cfg_(true);
    cfg_rmw(THCON_UNPACKER0_REG0_TRANSPOSE_RMW, 0);
    cfg_rmw(THCON_UNPACKER1_REG0_TRANSPOSE_RMW, 0);

    load_replay_buf<COMPRESSED_MM_UNPACK_SRC_A_REPLAY_IDX, 1>(
        [buf_desc_id_1]
        {
            // Src_Tile_Idx is ignored in ARG_FIFO mode 1: the entry holds the absolute tile address.
            TT_UNPACR0_TILE(0 /*Dst_Tile_Idx*/, 0 /*Src_Tile_Idx*/, buf_desc_id_1, 1 /*SetDatValid*/);
        });

    // Advance the SrcB source counter past the tile, so consecutive runs walk the Input0 k-tiles.
    const std::uint32_t src_b_replay_len =
        _llk_unpack_matmul_load_tile_replay_<p_unpacr::UNP_B>(COMPRESSED_MM_UNPACK_SRC_B_REPLAY_IDX, buf_desc_id_0, src_b_shape, true /*advance_to_next_tile*/);

    ckernel_template temp(1 /*outer loop*/, 1 /*inner loop*/, TT_OP_REPLAY(COMPRESSED_MM_UNPACK_SRC_B_REPLAY_IDX, src_b_replay_len, 0, 0, 0, 0));
    temp.program_bank0_sw_cntl(instrn_buffer);
}

/**
 * @brief Unpacks a [1, kt_dim] row of Input0 tiles into SrcB and a [kt_dim, ct_dim] block of compressed Input1
 *        tiles into SrcA, zeroing SrcA for the zero tiles instead of reading them.
 *
 * For each k, unpacks Input0 tile k into SrcB once, then every weight tile (k, 0..ct_dim-1) into SrcA; a zero
 * tile becomes a zeroed SrcA bank.
 *
 * @param base_address_in1: L1 address of the compressed weight stream, in 16 B units (L1_ADDRESS()).
 * @param base_address_meta: Byte address of the kt_dim * ct_dim metadata words. Not L1_ADDRESS(): the RISC-V
 *        dereferences it.
 * @param kt_dim: Number of tiles in the common dimension, values >= 1.
 * @param ct_dim: Number of tiles in the column dimension of Input 1, values = 1 - 16.
 * @note Call @ref _llk_unpack_compressed_custom_mm_init_ before this function.
 * @note On the math thread, pair with @ref _llk_math_compressed_custom_mm_ with the same metadata.
 */
inline void _llk_unpack_compressed_custom_mm_(
    const std::uint32_t base_address_in1, const std::uint32_t base_address_meta, const std::uint32_t kt_dim, const std::uint32_t ct_dim)
{
    const std::uint32_t* meta     = reinterpret_cast<const std::uint32_t*>(base_address_meta);
    const std::uint32_t num_tiles = kt_dim * ct_dim;

    // Input0 tiles start at the base of their buffer descriptor; the SrcA counters stay at 0.
    TTI_SET_SRC_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_B, 0);
    TTI_SET_DST_TILE_FACE_ROW_IDX(p_set_inc_sel::TILE_SEL, p_unpacr::UNP_A, 0);

    const auto unpack_src_a = [&](const std::uint32_t entry) __attribute__((always_inline))
    {
        // Zero tiles are the minority; without the hint the compiler lays them out inline and jumps away for every
        // stored tile.
        if (__builtin_expect(_llk_compressed_custom_mm_is_zero_tile_(entry), 0))
        {
            // No data in L1: hand math a zeroed SrcA bank, so it can run the same matmul for every tile.
            TTI_UNPACR_NOP(p_unpacr::UNP_A, 1 /*Set_Dvalid*/, 0, 0, p_unpacr::UNP_CLRSRC_ZERO, p_unpacr::UNP_CLRSRC);
        }
        else
        {
            TTI_REPLAY(COMPRESSED_MM_UNPACK_SRC_A_REPLAY_IDX, 1, 0, 0, 0, 0);
            // The relative offset in [19:0] cannot carry into the format field: L1 is 4 MB, 2^18 16 B units.
            *unpack0_arg_fifo_push = entry + base_address_in1;
        }
    };

    // The RISC-V, not the unpacker, sets the pace here, so the loop is built to issue as few instructions per tile
    // as possible:
    //   - Tiles are taken in groups of COMPRESSED_MM_ARG_FIFO_DEPTH metadata words. A group pushes at most that many
    //     entries, so one wait for an empty FIFO at its start replaces per-push accounting. The wait also drains
    //     whatever a previous call left behind.
    //   - The group's words are loaded before any is used. The core stalls on a load until its data arrives, so a
    //     word used right after its load costs the full L1 latency on every tile; loads issued back to back overlap.
    //   - Each UNPACR0 is issued before its entry is pushed: it blocks in the unpacker until the entry arrives.
    //   - The SrcB unpack of Input0 tile k goes in front of the first weight tile of row k. With ct_dim == 1 that is
    //     every tile, which needs no row counter.
    constexpr std::uint32_t GROUP = COMPRESSED_MM_ARG_FIFO_DEPTH;
    std::uint32_t c               = 0;
    const auto unpack_tile        = [&](const std::uint32_t entry) __attribute__((always_inline))
    {
        if (ct_dim == 1)
        {
            ckernel_template::run_bank0_sw_cntl(instrn_buffer);
        }
        else
        {
            if (c == 0)
            {
                ckernel_template::run_bank0_sw_cntl(instrn_buffer);
            }
            if (++c == ct_dim)
            {
                c = 0;
            }
        }
        unpack_src_a(entry);
    };

    std::uint32_t i = 0;
    for (; i + GROUP <= num_tiles; i += GROUP)
    {
        std::uint32_t entries[GROUP];
        for (std::uint32_t j = 0; j < GROUP; j++)
        {
            entries[j] = meta[i + j];
        }
        _llk_unpack_compressed_custom_mm_wait_arg_fifo_empty_();
        for (std::uint32_t j = 0; j < GROUP; j++)
        {
            unpack_tile(entries[j]);
        }
    }
    if (i < num_tiles)
    {
        _llk_unpack_compressed_custom_mm_wait_arg_fifo_empty_();
        for (; i < num_tiles; i++)
        {
            unpack_tile(meta[i]);
        }
    }
}

/**
 * @brief Switches the UNPACR0 argument FIFO back off, so later ops unpack from their buffer descriptors again.
 *
 * @note Call after the last @ref _llk_unpack_compressed_custom_mm_ of the op.
 */
inline void _llk_unpack_compressed_custom_mm_uninit_()
{
    // A leftover entry would feed the next ARG_FIFO user the wrong tile, so drain the FIFO first.
    _llk_unpack_compressed_custom_mm_wait_arg_fifo_empty_();
    _llk_unpack_compressed_custom_mm_arg_fifo_cfg_(false);
}
