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
#include "llk_assert.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// CUSTOM_MM
// Custom version of matmul that performs a full matrix multiplication more optimally but has the following limitations:
// in0 tile shape: [{1, 2, 4, 8}, 32] or [32, 32]
// in1 tile shape: [32, 32]
// rt_dim: any integer of 1 or more, as long as the rt_dim * ct_dim output tiles fit in the DEST sync region
// ct_dim: any integer from 1 to 64, with the same DEST bound (custom_mm_tile_rows of the CustomMmDestLayout per
//         output tile); nothing else bounds it
// kt_dim: even number from 2 to 254 (inclusive) for a short in0 tile and from 2 to 128 for a [32, 32] one; one SrcB
//         face walk covers CUSTOM_MM_MAX_FACES_PER_WALK faces
// fidelity: LoFi only
// throttle: not supported
//
// in0 is [rt_dim, kt_dim] tiles in row-major order and the output [rt_dim, ct_dim] tiles. The block is done one in0
// tile-row at a time: each row is a pass over the whole weight stream, which unpacks every in0 tile once and every
// weight tile rt_dim times. A [32, 32] in0 tile puts 32 activation rows behind one weight unpack, four times what
// the short tiles reach, so prefer it to stacking short tile-rows.

// Records the SrcB unpack of one in0 tile, two replay slots in either case.
inline void _llk_unpack_AB_compressed_custom_mm_in0_insns_(const bool full_in0)
{
    constexpr std::uint8_t set_dvalid = 1;
    if (full_in0)
    {
        // The whole [32, 32] tile in one instruction: four faces from L1 into SrcB rows 0 to 63. CH0 Y += 2 and
        // CH0 Z += 2 step L1 to the next tile, and SrcB starts at row 0 again without a counter to rewind.
        TTI_UNPACR_COMMON(SrcB, 0b00001010, set_dvalid);
        TTI_NOP;
    }
    else
    {
        // One face per instruction, at SrcB rows 0 and 16; CH1 Z += 3 after the second wraps SrcB back to row 0.
        TTI_UNPACR_COMMON(SrcB, 0b00010001, 0);
        TTI_UNPACR_COMMON(SrcB, 0b00110100, set_dvalid);
    }
}

inline void _llk_unpack_AB_compressed_custom_mm_mop_config_(const bool full_in0)
{
    constexpr std::uint8_t set_dvalid = 1;
    load_replay_buf(
        0,
        25,
        [full_in0]
        {
            // Bfp8
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
            TTI_WRCFG(p_gpr_unpack::PERF_UNPACK_NUM_TILES_3, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32);
            TTI_NOP;
            TTI_UNPACR_COMMON(SrcA, 0b00000000, set_dvalid);
            TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 0, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
            _llk_unpack_AB_compressed_custom_mm_in0_insns_(full_in0);
            // Bfp4
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
            TTI_WRCFG(p_gpr_unpack::PERF_UNPACK_NUM_TILES_2, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32);
            TTI_NOP;
            TTI_UNPACR_COMMON(SrcA, 0b00000000, set_dvalid);
            TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 1, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
            _llk_unpack_AB_compressed_custom_mm_in0_insns_(full_in0);
            // Bfp2
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
            TTI_WRCFG(p_gpr_unpack::PERF_UNPACK_NUM_TILES_1, p_cfg::WRCFG_32b, THCON_SEC0_REG0_TileDescriptor_ADDR32);
            TTI_NOP;
            TTI_UNPACR_COMMON(SrcA, 0b00000000, set_dvalid);
            TTI_CFGSHIFTMASK(1, 3, 32 - 1, 0, 2, THCON_SEC0_REG3_Base_address_ADDR32);
            TTI_NOP;
            _llk_unpack_AB_compressed_custom_mm_in0_insns_(full_in0);
            TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
        });
}

/**
 * @brief Configure the unpack thread for a compressed custom_mm block matmul, whose weight tiles each carry their own
 *        BFP format.
 *
 * Records the per-format unpack sequences into the replay buffer, sets the unpacker X ends, and loads the per-format
 * SrcA tile-size increments (SCRATCH_SEC0-2) and tile descriptors (GPRs PERF_UNPACK_NUM_TILES_1-3) the execute uses.
 *
 * @tparam transpose: Transpose the SrcA read, values = <true/false>
 * @tparam clear_src: Zero both SrcB banks once here, values = <true/false>. Only unpB_face_r_dim rows of each
 *                    SrcB face are unpacked, so zeroing the rest saves FPU power.
 * @param unpB_face_r_dim: Activation rows per face: 1, 2, 4 or 8 for a short tile, 16 for a [32, 32] one. Sets
 *                         unpacker 1's X end.
 * @param unpB_num_faces: Faces in an activation tile: 2 for [{1, 2, 4, 8}, 32], 4 for [32, 32].
 * @note Call this before @ref _llk_unpack_AB_compressed_custom_mm_, and again after any other op has run, in
 *       particular one that writes SrcB, SCRATCH_SEC0-2 or those GPRs (a custom_mm execute rewrites SCRATCH_SEC0/1):
 *       the execute relies on all of them and sets none.
 * @note On the math thread, pair with @ref _llk_math_compressed_custom_mm_init_, giving it the same face count.
 */
template <bool transpose = false, bool clear_src = true>
inline void _llk_unpack_AB_compressed_custom_mm_init_(const std::uint32_t unpB_face_r_dim, const std::uint32_t unpB_num_faces = CUSTOM_MM_SHORT_IN0_NUM_FACES)
{
    LLK_ASSERT(custom_mm_legal_in0(unpB_face_r_dim, unpB_num_faces), "compressed custom_mm: in0 must be a [{1, 2, 4, 8}, 32] or a [32, 32] tile");
    volatile std::uint32_t* cfg = get_cfg_pointer();
    const bool full_in0         = custom_mm_full_in0(unpB_num_faces);

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    constexpr std::uint32_t unpA_x_end = TILE_NUM_FACES * FACE_R_DIM * FACE_C_DIM - 1;
    // A short in0 tile goes in one face per instruction, a [32, 32] one whole
    const std::uint32_t unpB_x_end = (full_in0 ? TILE_NUM_FACES : 1) * unpB_face_r_dim * FACE_C_DIM - 1;
    TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);

    _llk_unpack_AB_compressed_custom_mm_mop_config_(full_in0);

    if constexpr (clear_src)
    {
        // Clear SrcB as we only unpack into 1/8 FPU rows so zeroing them gives power savings
        // This particular instruction clears both banks after waiting for both of them to be free
        // It must run alone: a both-bank SrcB clear drops SrcA writes that unpacker 0 makes while it runs
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK1);
    }

    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
    TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);

    // These are RISC-V stores, so they can overtake unpacks still queued from the previous op (e.g. a
    // tilize whose CFGSHIFTMASK reads SCRATCH_SEC0); wait for its contexts to retire first.
    wait_for_next_context(1);

    cfg[SCRATCH_SEC0_val_ADDR32]                   = 68;
    cfg[SCRATCH_SEC1_val_ADDR32]                   = 36;
    cfg[SCRATCH_SEC2_val_ADDR32]                   = 20;
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_3] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp8_b);
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_2] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp4_b);
    regfile[p_gpr_unpack::PERF_UNPACK_NUM_TILES_1] = 0x10 | static_cast<std::uint32_t>(DataFormat::Bfp2_b);
}

constexpr std::uint32_t get_replay_insn_for_combo(const std::uint8_t combo)
{
    std::uint8_t prev = combo & 0b11;
    std::uint8_t curr = (combo >> 3) & 0b11;

    bool use_b         = (combo >> 2) & 0b1;
    bool need_reconfig = prev != curr;
    bool need_stall    = need_reconfig && (prev == 1 || curr == 1);

    std::uint32_t start_idx    = curr == 3 ? 0 : curr == 2 ? 8 : 16;
    std::uint32_t start_offset = 3;
    std::uint32_t replay_len   = 3;

    if (use_b)
    {
        replay_len += 2;
    }
    if (need_reconfig)
    {
        start_offset -= 2;
        replay_len += 2;
    }
    if (need_stall)
    {
        start_offset -= 1;
        replay_len += 1;
    }

    if (curr == 0)
    {
        if (use_b)
        {
            return lltt::replay_insn(22, need_stall ? 3 : 2);
        }
        else
        {
            return need_stall ? lltt::replay_insn(24, 1) : TT_OP_NOP;
        }
    }

    return lltt::replay_insn(start_idx + start_offset, replay_len);
}

/**
 * @brief Unpack an rt_dim x kt_dim block of activation tiles into SrcB and, once per activation tile-row, a
 *        kt_dim x ct_dim block of compressed weight tiles into SrcA, switching the SrcA format per tile as the
 *        metadata says.
 *
 * @param base_address_a: L1 address of the compressed weight stream (SrcA), in the 16-byte-word encoding of
 *                        L1_ADDRESS(). Tiles are packed back to back at their own format's size; zero tiles take no
 *                        space.
 * @param base_address_b: L1 address of the activations (SrcB), in the same encoding. Tiles are in row-major
 *                        (r, k) order.
 * @param base_address_meta: Byte address of the per-tile metadata, read by the RISC-V (so not L1_ADDRESS()). Each
 *                           word holds 10 tiles: the previous tile's format in bits 1:0, then per tile a use_b bit
 *                           and a 2-bit format (0 zero, 1 bfp2, 2 bfp4, 3 bfp8).
 * @param kt_dim: Inner dimension in tiles, an even number from 2 to 254, or to 128 for a [32, 32] activation tile.
 * @param ct_dim: Output width in tiles, 1 to 64; DEST capacity bounds it, see @ref _llk_math_compressed_custom_mm_.
 * @param rt_dim: Activation tile-rows, 1 or more; each one is a pass over the weight stream and its metadata.
 * @param tile_size_b: L1 stride between activation tiles, in 16-byte words. Only read for rt_dim > 1, to place the
 *                     tile-rows after the first.
 * @note Call @ref _llk_unpack_AB_compressed_custom_mm_init_ first.
 * @note Every pass ends on the to/from-bfp2 format stall, so the next pass or call can start its metadata from the
 *       previous-format sentinel (0) without encoding one.
 * @note On the math thread, pair with @ref _llk_math_compressed_custom_mm_ with the same rt_dim.
 */
inline void _llk_unpack_AB_compressed_custom_mm_(
    const std::uint32_t base_address_a,
    const std::uint32_t base_address_b,
    const std::uint32_t base_address_meta,
    const std::uint32_t kt_dim,
    const std::uint32_t ct_dim      = 1,
    const std::uint32_t rt_dim      = 1,
    const std::uint32_t tile_size_b = 0)
{
    static constexpr std::uint32_t FMTABLE[32] = {
        get_replay_insn_for_combo(0b000'00), // 0b000'00 zero to zero
        get_replay_insn_for_combo(0b000'01), // 0b000'01 bfp2 to zero
        get_replay_insn_for_combo(0b000'10), // 0b000'10 bfp4 to zero
        get_replay_insn_for_combo(0b000'11), // 0b000'11 bfp8 to zero

        get_replay_insn_for_combo(0b001'00), // 0b001'00 zero to zero with b
        get_replay_insn_for_combo(0b001'01), // 0b001'01 bfp2 to zero with b
        get_replay_insn_for_combo(0b001'10), // 0b001'10 bfp4 to zero with b
        get_replay_insn_for_combo(0b001'11), // 0b001'11 bfp8 to zero with b

        get_replay_insn_for_combo(0b010'00), // 0b010'00 zero to bfp2
        get_replay_insn_for_combo(0b010'01), // 0b010'01 bfp2 to bfp2
        get_replay_insn_for_combo(0b010'10), // 0b010'10 bfp4 to bfp2
        get_replay_insn_for_combo(0b010'11), // 0b010'11 bfp8 to bfp2

        get_replay_insn_for_combo(0b011'00), // 0b011'00 zero to bfp2 with b
        get_replay_insn_for_combo(0b011'01), // 0b011'01 bfp2 to bfp2 with b
        get_replay_insn_for_combo(0b011'10), // 0b011'10 bfp4 to bfp2 with b
        get_replay_insn_for_combo(0b011'11), // 0b011'11 bfp8 to bfp2 with b

        get_replay_insn_for_combo(0b100'00), // 0b100'00 zero to bfp4
        get_replay_insn_for_combo(0b100'01), // 0b100'01 bfp2 to bfp4
        get_replay_insn_for_combo(0b100'10), // 0b100'10 bfp4 to bfp4
        get_replay_insn_for_combo(0b100'11), // 0b100'11 bfp8 to bfp4

        get_replay_insn_for_combo(0b101'00), // 0b101'00 zero to bfp4 with b
        get_replay_insn_for_combo(0b101'01), // 0b101'01 bfp2 to bfp4 with b
        get_replay_insn_for_combo(0b101'10), // 0b101'10 bfp4 to bfp4 with b
        get_replay_insn_for_combo(0b101'11), // 0b101'11 bfp8 to bfp4 with b

        get_replay_insn_for_combo(0b110'00), // 0b110'00 zero to bfp8
        get_replay_insn_for_combo(0b110'01), // 0b110'01 bfp2 to bfp8
        get_replay_insn_for_combo(0b110'10), // 0b110'10 bfp4 to bfp8
        get_replay_insn_for_combo(0b110'11), // 0b110'11 bfp8 to bfp8

        get_replay_insn_for_combo(0b111'00), // 0b111'00 zero to bfp8 with b
        get_replay_insn_for_combo(0b111'01), // 0b111'01 bfp2 to bfp8 with b
        get_replay_insn_for_combo(0b111'10), // 0b111'10 bfp4 to bfp8 with b
        get_replay_insn_for_combo(0b111'11), // 0b111'11 bfp8 to bfp8 with b
    };

    LLK_ASSERT(kt_dim <= 254, "compressed custom_mm: kt_dim above 254 needs the SrcB walk split");
    LLK_ASSERT(rt_dim >= 1, "compressed custom_mm: rt_dim must be at least 1");

    volatile std::uint32_t* cfg = get_cfg_pointer();

    const std::uint32_t address_a  = base_address_a;
    const std::uint32_t full_iters = (kt_dim * ct_dim) / 10;
    const std::uint32_t rem_iters  = (kt_dim * ct_dim) % 10;
    const std::uint32_t* meta_ptr  = reinterpret_cast<const std::uint32_t*>(base_address_meta);

    // One pass over the weights per activation tile-row: each restarts the weight stream and its metadata, and
    // takes its SrcB base from the row's first tile, so the face walk never grows with rt_dim.
    for (std::uint32_t r = 0; r < rt_dim; ++r)
    {
        const std::uint32_t address_b = base_address_b + r * kt_dim * tile_size_b;

        wait_for_next_context(1);
        reset_config_context();

        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_a;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_b;

        semaphore_post(semaphore::UNPACK_SYNC);

        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

        for (std::uint32_t i = 0; i < full_iters; ++i)
        {
            std::uint32_t meta = meta_ptr[i];

            std::uint32_t idx0 = (meta >> 0) & 0b11111;
            std::uint32_t idx1 = (meta >> 3) & 0b11111;
            std::uint32_t idx2 = (meta >> 6) & 0b11111;
            std::uint32_t idx3 = (meta >> 9) & 0b11111;
            std::uint32_t idx4 = (meta >> 12) & 0b11111;
            std::uint32_t idx5 = (meta >> 15) & 0b11111;
            std::uint32_t idx6 = (meta >> 18) & 0b11111;
            std::uint32_t idx7 = (meta >> 21) & 0b11111;
            std::uint32_t idx8 = (meta >> 24) & 0b11111;
            std::uint32_t idx9 = (meta >> 27) & 0b11111;

            TT_INSN(FMTABLE[idx0]);
            TT_INSN(FMTABLE[idx1]);
            TT_INSN(FMTABLE[idx2]);
            TT_INSN(FMTABLE[idx3]);
            TT_INSN(FMTABLE[idx4]);
            TT_INSN(FMTABLE[idx5]);
            TT_INSN(FMTABLE[idx6]);
            TT_INSN(FMTABLE[idx7]);
            TT_INSN(FMTABLE[idx8]);
            TT_INSN(FMTABLE[idx9]);
        }
        std::uint32_t meta = meta_ptr[full_iters];
        for (std::uint32_t i = 0; i < rem_iters; ++i)
        {
            std::uint32_t idx0 = (meta >> 0) & 0b11111;
            TT_INSN(FMTABLE[idx0]);
            meta >>= 3;
        }

        t6_semaphore_get(semaphore::UNPACK_SYNC);

        // The to/from bfp2 stall is encoded per tile only within a pass; the first tile's prev format is a
        // metadata sentinel, not the last format of the previous pass or call, so always end on the stall. It only
        // has to precede the next unpacks, so it goes after the SEMGET to keep it out of the context handshake.
        TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1 /* Stall_Clr_Cntrl */, 0, 0, p_unpacr_nop::CLR_SRC);

        wait_for_next_context(1);
        reset_config_context();

        TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);
        TTI_SETADCXY(0b011, 0, 0, 0, 0, 0b1010);
    }
}
