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
#include "cunpack_common.h"
#include "experimental/llk_custom_mm_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32]
// in1 tile shape: [32, 32] or [32, 16] ([16, 32] with transpose), giving [{1, 2, 4, 8}, 32] or [{1, 2, 4, 8}, 16] output tiles
// rt_dim: 1
// ct_dim: any integer from 1 to 34 (DEST capacity also bounds it, see llk_custom_mm_common.h)
// kt_dim: any integer of 1 or more
// fidelity: LoFi only
// throttle: not supported

inline void _llk_unpack_AB_custom_mm_iter_insns(const bool post1)
{
    // Unpack SrcA (in1, full tile, uses CFGSHIFTMASK to manipulate L1 addr and keeps SrcA addr fixed at 0)
    TTI_UNPACR_COMMON(SrcA, 0b00000000, 1); // Also set dvalid

    // Unpack SrcB (in0, one instruction per face, uses counters to manipulate addresses for both L1 and SrcB)
    // Counters of interest are CH0 Y and Z (both with a stride of a single face in L1
    // (datum_size * 16 * face_r_dim) and CH1 Z with a stride of a face in SrcB (16 rows)
    // Each unpack instruction moves to the next face in L1, split among the Y and Z counters.
    // The L1 face index is (Y & 0xff) + (Z & 0xff), so one base address covers CUSTOM_MM_MAX_KT_PER_WALK k-tiles.
    // For SrcB the first unpack lands at row 0 and the second at row 16; CH1 Z is then reset to 0
    // explicitly (SETADCZW below) so no SrcB row past 63 is ever addressed.
    TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
    if (!post1)
    {
        // Unpack second face of SrcB before CFGSHIFTMASK in post0 tuning
        TTI_UNPACR_COMMON(SrcB, 0b00000100, 1); // Also set dvalid
    }

    // Increment UnpA L1 address
    TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
    if (post1)
    {
        // Unpack second face of SrcB after CFGSHIFTMASK in post1 tuning
        TTI_UNPACR_COMMON(SrcB, 0b00000100, 1); // Also set dvalid;
    }
    // Rewind the SrcB row counter (CH1 Z) to row 0 for the next k-tile. In post0 this also fills the slot after
    // CFGSHIFTMASK (a 2 cycle instruction) before the next SrcA unpack, which reads the address it updates.
    TTI_SETADCZW(p_setadc::UNP_B, 0, 0, 0, 0, 0b0100);
}

// TT_MOP has a 7-bit loop count.
constexpr std::uint32_t CUSTOM_MM_MAX_MOP_ITERATIONS = 128;

inline void _llk_unpack_AB_custom_mm_mop_config_(const std::uint32_t ct_dim, const bool post1)
{
    LLK_ASSERT(ct_dim >= 1 && ct_dim <= CUSTOM_MM_MAX_CT_DIM, "custom_mm: ct_dim must be 1 to 34");

    load_replay_buf(
        0,
        32,
        [post1]
        {
            // Full unpack (both SrcA and SrcB)
            _llk_unpack_AB_custom_mm_iter_insns(post1);

            // Loop 8 times to fill up the replay buffer
            for (std::uint32_t i = 0; i < 8; i++)
            {
                // Reuse unpack (unpacks only SrcA, SrcB is reused across width dim)
                TTI_UNPACR_COMMON(SrcA, 0b00000000, 1); // Also set dvalid
                TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
                TTI_NOP;
            }

            // Reuse unpack (unpacks only SrcA, SrcB is reused across width dim)
            TTI_UNPACR_COMMON(SrcA, 0b00000000, 1); // Also set dvalid
            // This last iteration uses inner_increment instead of block_increment
            TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 1, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
        });

    // Each mop iteration covers one kt_dim loop: the ct_dim output tiles of one k-tile, all sharing its SrcB tile.
    // To usefully issue up to 128 mop iterations we're limited to only using 0s in zmask
    // (not using SKIP_A/B instructions) since iterations beyond 32 always use 0s for zmask
    //
    // Replay buffer layout (always 32 instructions):
    // post0: [0-4]: full unpack (ends in the SrcB row reset), [5-28]: 8 reuse blocks (3 insns each), [29-31]: final reuse w/ inner_increment
    // post1: same layout; its full unpack only orders CFGSHIFTMASK before the second SrcB unpack
    // Where:
    //   - full unpack = SrcA + SrcB + address increment (new k-tile)
    //   - reuse = SrcA only + address increment (same k-tile, next c-tile)
    //   - final reuse = uses inner_increment to jump to next k-row instead of block_increment
    //
    // Sequence per mop iteration: [full, reuse x (ct_dim - 2), final reuse], or [full] for ct_dim == 1
    // (block_increment == inner_increment then, so no final reuse is needed).
    //
    // The ct_dim - 1 reuses split over the template slots, each slot a replay range:
    //   the first slot starts at the buffer beginning and holds the full unpack plus some reuses,
    //   the last slot counts backwards from the buffer end so it always ends in the final reuse,
    //   any slots between replay reuse blocks from the start of the reuse range, since all reuses but the last
    //   are the same instructions.
    // Every slot the template issues must replay something, so the configuration follows the reuse count:
    //   ct_dim == 1:       A0 = full                                       (no reuse for a B slot)
    //   ct_dim 2 to 3:     A0 = full + a reuses, B = d reuses              (too few reuses for A1..A3)
    //   ct_dim 4 to 34:    A0 = full + a reuses, A1 = b, A2 = c, A3 = d    (a, b, c <= 8, d <= 9)
    // The reuses split evenly; the remainder goes to A3, A2 and A1 before A0, which already holds the full unpack,
    // so A1..A3 get a reuse each from ct_dim 4 on.
    const std::uint32_t reuses          = ct_dim - 1;
    const auto full_and                 = [](const std::uint32_t n) { return lltt::replay_insn(0, 2 + (1 + n) * 3); };
    const auto final_reuses             = [](const std::uint32_t n) { return lltt::replay_insn(32 - n * 3, n * 3); };
    constexpr std::uint32_t reuse_start = 5;

    std::uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b = 0;
    bool use_b = false, use_halo = false;
    if (reuses == 0)
    {
        a0 = full_and(0);
    }
    else if (reuses < 3)
    {
        use_b = true;
        a0    = full_and(reuses >> 1);
        b     = final_reuses(reuses - (reuses >> 1));
    }
    else
    {
        use_halo                 = true;
        const std::uint32_t base = reuses >> 2;
        const std::uint32_t rem  = reuses & 3;
        a0                       = full_and(base);
        a1                       = lltt::replay_insn(reuse_start, (base + (rem > 2 ? 1 : 0)) * 3);
        a2                       = lltt::replay_insn(reuse_start, (base + (rem > 1 ? 1 : 0)) * 3);
        a3                       = final_reuses(base + (rem > 0 ? 1 : 0));
    }

    ckernel_unpack_template tmp = ckernel_unpack_template(
        use_b,    // Use UNPACR_B and SKIP_B instructions?
        use_halo, // Use UNPACR_A1/2/3 instructions?
        a0,       // A0
        a1,       // A1
        a2,       // A2
        a3,       // A3
        0,        // Skip A
        b,        // B
        0         // Skip B
    );

    tmp.program();
    TTI_MOP_CFG(0);
}

/**
 * @brief Configure the unpack thread for a custom_mm block matmul.
 *
 * Sets the unpacker X ends, records the MOP and replay buffer the execute runs, and resets the counters.
 *
 * @tparam transpose: Transpose the SrcA read, values = <true/false>
 * @tparam clear_src: Zero both SrcB banks once here, values = <true/false>. Only unpB_face_r_dim rows of each
 *                    SrcB face are unpacked, so zeroing the rest saves FPU power.
 * @param unpB_face_r_dim: Activation rows per face, 1, 2, 4 or 8. Sets unpacker 1's X end.
 * @param unpA_dst_format: Unpack destination format of the weights (SrcA); Bfp4_b selects a tuned sequence.
 * @param ct_dim: Output width in tiles, 1 to 34.
 * @param unpA_num_faces: Faces in a weight tile: 4 for [32, 32], 2 for [32, 16] ([16, 32] with transpose). Sets
 *                        unpacker 0's X end.
 * @note Call this before @ref _llk_unpack_AB_custom_mm_, and again after any other op has run, in particular one
 *       that writes SrcB: the execute does not reprogram the MOP, and the SrcB clear happens only here.
 * @note On the math thread, pair with @ref _llk_math_custom_mm_init_, giving it the same face count.
 */
template <bool transpose = false, bool clear_src = true>
inline void _llk_unpack_AB_custom_mm_init_(
    const std::uint32_t unpB_face_r_dim,
    const std::uint32_t unpA_dst_format,
    const std::uint32_t ct_dim         = 1,
    const std::uint32_t unpA_num_faces = TILE_NUM_FACES)
{
    LLK_ASSERT(custom_mm_legal_in1_num_faces(unpA_num_faces), "custom_mm: in1 must have 4 faces, or 2 for a [32, 16] tile");
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    // UnpA unpacks full tiles, of 4 faces or 2 for a [32, 16] tile
    const std::uint32_t unpA_x_end = unpA_num_faces * FACE_R_DIM * FACE_C_DIM - 1;
    // UnpB unpacks [{1, 2, 4, 8}, 32] tiles which only have top two faces and only a single face per instruction
    // so a single instruction only unpacks unpB_face_r_dim rows
    const std::uint32_t unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1;
    TT_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

    // This is a profiling guided heuristic that slightly tunes the instruction sequence
    // More details under tt-metal#38518
    const bool post1 = unpA_dst_format == to_underlying(DataFormat::Bfp4_b);

    _llk_unpack_AB_custom_mm_mop_config_(ct_dim, post1);

    if constexpr (clear_src)
    {
        // Clear SrcB as we only unpack into 1/8 FPU rows so zeroing them gives power savings
        // This particular instruction clears both banks after waiting for both of them to be free
        // It must run alone: a both-bank SrcB clear drops SrcA writes that unpacker 0 makes while it runs
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK1);
    }

    // Reset counters here since reset in the execute API is at the end
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
}

// Most k-tiles one SrcB base address covers: k-tile k reads faces k and k + 1 of an 8-bit Y + 8-bit Z face index,
// which reaches face 255 + 255 = 510, so k-tile 255 would need face 511 and alias.
constexpr std::uint32_t CUSTOM_MM_MAX_KT_PER_WALK = 255;

/**
 * @brief Issue the SrcA/SrcB unpack of a kt_dim x ct_dim block with the MOP that @ref _llk_unpack_AB_custom_mm_init_
 *        recorded. Shared by @ref _llk_unpack_AB_custom_mm_ and @ref _llk_unpack_AB_sdpa_custom_mm_.
 *
 * Splits kt_dim into ceil(kt_dim / CUSTOM_MM_MAX_KT_PER_WALK) SrcB face walks whose lengths differ by at most one.
 * Each walk sets the SrcB base, waits for its UNPACRs to issue and resets the counters. The SrcA base carries over.
 *
 * @param cfg: Config register pointer of the unpack thread.
 * @param address_a: L1 address of the first weight tile (SrcA), in 16-byte words.
 * @param address_b: L1 address of the first activation tile (SrcB), in 16-byte words.
 * @param tile_size_b: L1 stride between activation k-tiles, in 16-byte words. Above CUSTOM_MM_MAX_KT_PER_WALK k-tiles
 *                     it must equal the dense stride the counters step, two in0 faces (4 * face_r_dim words for
 *                     16-bit in0), since later walks are placed with it.
 * @param block_increment: SrcA address step to the next tile of a k-row, in 16-byte words.
 * @param inner_increment: SrcA address step from the last tile of a k-row to the next k-row, in 16-byte words.
 * @param kt_dim: Inner dimension in tiles, 1 or more.
 */
inline void _llk_unpack_AB_custom_mm_run_(
    volatile std::uint32_t* cfg,
    std::uint32_t address_a,
    const std::uint32_t address_b,
    const std::uint32_t tile_size_b,
    const std::uint32_t block_increment,
    const std::uint32_t inner_increment,
    const std::uint32_t kt_dim)
{
    LLK_ASSERT(kt_dim >= 1, "custom_mm: kt_dim must be at least 1");
    // Program the SrcA base once; CFGSHIFTMASK advances it across all walks
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
    // Setup CFGSHIFTMASK increments
    // block_increment moves to next tile in tile_row
    cfg[SCRATCH_SEC0_val_ADDR32] = block_increment;
    // inner_increment moves to next tile_row (next inner_dim)
    cfg[SCRATCH_SEC1_val_ADDR32] = inner_increment;

    constexpr std::uint32_t mop_template_0 = 0;
    constexpr std::uint32_t no_skip_zmask  = 0;

    // Split kt_dim into near-equal walks; the first kt_dim % num_walks take one extra k-tile
    const std::uint32_t num_walks  = (kt_dim + CUSTOM_MM_MAX_KT_PER_WALK - 1) / CUSTOM_MM_MAX_KT_PER_WALK;
    const std::uint32_t base_kt    = kt_dim / num_walks;
    const std::uint32_t long_walks = kt_dim % num_walks;
    std::uint32_t walk_kt          = 0;
    std::uint32_t walk             = 0;
    for (std::uint32_t walk_start = 0; walk_start < kt_dim; walk_start += walk_kt, walk++)
    {
        walk_kt = base_kt + (walk < long_walks ? 1 : 0);

        // Program the SrcB base for this walk; the counters advance it within the walk
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b + walk_start * tile_size_b;

        semaphore_post(semaphore::UNPACK_SYNC);

        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        // One mop iteration per kt_dim loop, at most CUSTOM_MM_MAX_MOP_ITERATIONS per TT_MOP
        for (std::uint32_t kt_done = 0; kt_done < walk_kt; kt_done += CUSTOM_MM_MAX_MOP_ITERATIONS)
        {
            const std::uint32_t kt_left    = walk_kt - kt_done;
            const std::uint32_t iterations = kt_left < CUSTOM_MM_MAX_MOP_ITERATIONS ? kt_left : CUSTOM_MM_MAX_MOP_ITERATIONS;
            TT_MOP(mop_template_0, iterations - 1, no_skip_zmask);
        }

        // Wait until every UNPACR of this walk has issued
        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // Wait for all contexts to be free
        wait_for_next_context(1);
        reset_config_context();

        // Reset counters at the end of each walk
        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
    }
}

/**
 * @brief Unpack a kt_dim x ct_dim block of weight tiles into SrcA and the matching activation tiles into SrcB.
 *
 * @tparam read_transposed: Walk the weight tiles column by column (ct_dim tiles with a stride of kt_dim, then the
 *                          next tile) instead of row by row, values = <true/false>
 * @param base_address_a: L1 address of the weights (SrcA), in the 16-byte-word encoding of L1_ADDRESS().
 * @param base_address_b: L1 address of the activations (SrcB), in the same encoding.
 * @param tile_index_a: First weight tile to read.
 * @param tile_index_b: First activation tile to read.
 * @param tile_size_a: Weight tile size, in 16-byte words.
 * @param tile_size_b: L1 stride between activation k-tiles, in 16-byte words; above CUSTOM_MM_MAX_KT_PER_WALK k-tiles
 *                     the dense two-face in0 stride, see @ref _llk_unpack_AB_custom_mm_run_.
 * @param kt_dim: Inner dimension in tiles, 1 or more.
 * @param ct_dim: Output width in tiles; must match the init, whose MOP bakes it in.
 * @note Call @ref _llk_unpack_AB_custom_mm_init_ first.
 * @note On the math thread, pair with @ref _llk_math_custom_mm_.
 */
template <bool read_transposed = false>
inline void _llk_unpack_AB_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const std::uint32_t tile_size_a,
    const std::uint32_t tile_size_b,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim)
{
    volatile std::uint32_t* cfg = get_cfg_pointer();

    const std::uint32_t block_increment = read_transposed ? kt_dim * tile_size_a : tile_size_a;
    const std::uint32_t inner_increment = read_transposed ? -(((ct_dim - 1) * kt_dim) - 1) * tile_size_a : tile_size_a;

    const std::uint32_t address_a = base_address_a + tile_size_a * tile_index_a;
    const std::uint32_t address_b = base_address_b + tile_size_b * tile_index_b;

    // Wait for all contexts to be free
    wait_for_next_context(1);
    reset_config_context();

    _llk_unpack_AB_custom_mm_run_(cfg, address_a, address_b, tile_size_b, block_increment, inner_increment, kt_dim);
}
