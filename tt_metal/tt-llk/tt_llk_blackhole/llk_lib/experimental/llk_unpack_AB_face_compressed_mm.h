// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// Unpack meta layout, in the index-word section of the meta buffer.
//
// One 32-bit index word packs six 6-bit metas at a stride of five bits, so adjacent windows overlap by one
// bit: meta k is (word >> 5k) & 0b111111, and its bit 0 is the very bit that held meta k-1's last face.
// That shared bit is what buys the sixth meta per word. Within a window:
//
//   bit 0     format of the preceding face; in word 0 the producer replicates this meta's own first face,
//             so the first meta never opens with a format switch
//   bit 1     whether this meta starts an activation block, in which case it also loads the group's SrcB;
//             a meta holds either one first-in-block face or none, never two
//   bits 2-5  formats of this meta's four faces, 0 bfp2 and 1 bfp4
//
// The shared bit is a producer invariant nothing on this side can check: word w bit 0 must equal word w-1
// bit 30, the two spellings of that one face format. encode_meta in test_matmul_face_compressed.py upholds
// it; a hand-built meta that does not corrupts silently.

// The two BFP precisions need their own row counters so one instruction stream can alternate format face to
// face, and the unpacker has only one set of its own. The RISC keeps both row positions and, whenever the
// format changes, writes the new format's row into the single Ch0 Y counter before the face is unpacked.

/**
 * @brief Configure the unpack thread for a face-granular compressed matmul.
 *
 * @tparam transpose: Haloize the SrcA read, values = <true/false>
 * @param unpB_face_r_dim: Activation rows per face, 1 or 8. Sets unpacker 1's X end.
 * @note Call this before @ref _llk_unpack_AB_face_compressed_mm_, and
 *       @ref _llk_unpack_AB_face_compressed_mm_uninit_ after the last one, to put back the tile descriptor
 *       num_faces this forces to a single face.
 * @note On the math thread, pair with @ref _llk_math_face_compressed_mm_init_.
 */
template <bool transpose = false>
inline void _llk_unpack_AB_face_compressed_mm_init_(const std::uint32_t unpB_face_r_dim)
{
    LLK_ASSERT(unpB_face_r_dim == 1 || unpB_face_r_dim == 8, "face_compressed_mm (unpack): unsupported activation face_r_dim (expected 1 or 8)");

    // The config writes below need no preceding stall. The unpacker samples these fields when it issues an
    // UNPACR, not while the unpack runs, so they are free to change as soon as the previous UNPACR has been
    // issued -- which program order on this thread already guarantees.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Haloize_mode_RMW>(transpose ? 1 : 0);

    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Ovrd_data_format_RMW>(1); // read dataformat from per cntx registers
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx0_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx0_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx1_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx1_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx2_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx2_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp2_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_data_format_cntx3_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));
    cfg_reg_rmw_tensix<THCON_SEC0_REG7_Unpack_out_data_format_cntx3_RMW>(static_cast<std::uint32_t>(DataFormat::Bfp4_b));

    // Replicate onto cntx2/3 what hw_configure programs for cntx0/1, the only contexts used before. Both
    // values come from there: the x_dim through the same canonical helper, the dest address as a literal.
    // Each word packs the cntx2 value low and the cntx3 value high.
    constexpr std::uint32_t canonical_x_dim_cntx = canonical_unpA_tile_x_dim_cntx(FACE_R_DIM);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx2_ADDR32, 0, 0xffffffff>(canonical_x_dim_cntx);
    cfg_reg_rmw_tensix<THCON_SEC0_REG5_Dest_cntx2_address_ADDR32, 0, 0xffffffff>(((4 * 16) << 16) | (4 * 16));

    // override z dim
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xFF0000>(1);

    constexpr std::uint32_t unpA_x_end = FACE_R_DIM * FACE_C_DIM - 1;
    const std::uint32_t unpB_x_end     = 4 * unpB_face_r_dim * FACE_C_DIM - 1;

    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
    TTI_SETADCXX(p_setadc::UNP_A, unpA_x_end, 0x0);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);
}

/**
 * @brief Restore the unpacker state the init changed.
 *
 * @param unpA_num_faces: Faces per tile to put back in unpacker 0's tile descriptor, which
 *                        @ref _llk_unpack_AB_face_compressed_mm_init_ forced to a single face.
 * @note Call after the last @ref _llk_unpack_AB_face_compressed_mm_.
 */
inline void _llk_unpack_AB_face_compressed_mm_uninit_(const std::uint32_t unpA_num_faces)
{
    // No stall needed here either, same as the init: the last UNPACR sampled these fields when it issued.
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Ovrd_data_format_RMW>(0);
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xFF0000>(unpA_num_faces);

    // reset CH0 XY counters for both unpackers to leave them in a clean state
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
}

/**
 * @brief Unpack the activation block into SrcB and the compressed weight faces into SrcA.
 *
 * Streams the meta buffer's index words, unpacking each meta's faces, and reloads the
 * double-buffered unpacker contexts with the next chunk's weight base addresses as it goes.
 *
 * @tparam ct_dim: Output width in tiles, 1 to 16.
 * @tparam clear_src: Clear SrcB before the first unpack, values = <true/false>
 * @tparam finalize: For ct_dim == 1, leave both sources zeroed and valid so the math thread can merge its
 *                   split-accumulation partials, values = <true/false>
 * @param base_address_b: SrcB base address, which is the activation CB's read pointer.
 * @param base_address_meta: L1 address of the meta buffer. This thread reads the weight base addresses and
 *                           the index words that follow the math metas.
 * @param kt_dim: Inner dimension in tiles, an even number in [2, 256].
 * @note Call @ref _llk_unpack_AB_face_compressed_mm_init_ first.
 * @note On the math thread, pair with @ref _llk_math_face_compressed_mm_.
 */
template <std::uint32_t ct_dim = 1, bool clear_src = true, bool finalize = true>
inline void _llk_unpack_AB_face_compressed_mm_(const std::uint32_t base_address_b, const std::uint32_t base_address_meta, const std::uint32_t kt_dim)
{
    static_assert(ct_dim >= 1 && ct_dim <= 16, "face_compressed_mm (unpack): ct_dim must be in [1, 16]");
    LLK_ASSERT(kt_dim >= 2 && kt_dim <= 256 && kt_dim % 2 == 0, "face_compressed_mm (unpack): kt_dim must be an even number in [2, 256]");

    // A meta address word packs two fields: the low 24 bits are a 16B-word base address, written to one
    // unpacker context's Base_address, and the top byte is a Y offset, the first row of that format in the chunk.
    constexpr std::uint32_t meta_addr_base_mask   = 0x00FFFFFF;
    constexpr std::uint32_t meta_addr_y_off_shift = 24;

    // Geometry of the index words, per the layout block above. The stride is narrower
    // than the meta because consecutive metas share a bit, which is what fits six of them in one word.
    constexpr std::uint32_t meta_index_bits      = 6;
    constexpr std::uint32_t meta_stride_bits     = 5;
    constexpr std::uint32_t meta_index_mask      = (1u << meta_index_bits) - 1;
    constexpr std::uint32_t metas_per_index_word = 6;
    static_assert((metas_per_index_word - 1) * meta_stride_bits + meta_index_bits <= 32, "six 6-bit metas at a five-bit stride must fit one 32-bit index word");

    // Written to UNPACK_MISC_CFG_CfgContextOffset_0 to flip which context pair the UNPACRs take their base
    // address and format from. The pairs are double buffered: while the unpacker streams from one, the RISC
    // writes the next chunk's base addresses into the other, then switches.
    constexpr std::uint32_t unp_cfg_ctxt_offset_base = 0x0000; // cntx0 / cntx1
    constexpr std::uint32_t unp_cfg_ctxt_offset_alt  = 0x0002; // cntx2 / cntx3

    // Row position of the next face of each format, in the Ch0 Y counter's units.
    std::uint32_t row[2];

    // Take the Y offsets of a consecutive bfp2/bfp4 pair of meta address words as the two row positions, and
    // point the Y counter at the one of the face format the stream is in, which is bit 0 of the next index word.
    auto set_y_off = [&row](const std::uint32_t tt_l1_ptr* addr_pair, const std::uint32_t next_index_word)
    {
        row[0] = addr_pair[0] >> meta_addr_y_off_shift;
        row[1] = addr_pair[1] >> meta_addr_y_off_shift;
        TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, row[next_index_word & 1]);
    };

    // Unpack one meta's faces. A face of another format than the one before it needs the CLR_SRC stall ahead
    // of it and its own row in the Y counter, which then advances by one per face of that format.
    auto emit_meta = [&row](std::uint32_t meta)
    {
        const bool block        = (meta >> 1) & 1;
        std::uint32_t prev_face = meta & 1;
#pragma GCC unroll 4
        for (std::uint32_t i = 0; i < 4; ++i)
        {
            const std::uint32_t face = (meta >> (2 + i)) & 1;
            if (face != prev_face)
            {
                TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
                TT_SETADC(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, row[face]);
            }
            if (i == 0 && block)
            {
                TTI_UNPACR_COMMON(SrcB, 0b00'00'10'10, 1); // Ch0Y += 2 Ch0Z += 2
            }
            if (face)
            {
                TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00'00'01'00, 1, 1); // Ch0Y += 1
            }
            else
            {
                TTI_UNPACR_COMMON_EXPLICIT_CONTEXT(SrcA, 0b00'00'01'00, 0, 1); // Ch0Y += 1
            }
            ++row[face];
            prev_face = face;
        }
    };

    auto emit_word = [&emit_meta](std::uint32_t word) __attribute__((noinline))
    {
        for (std::uint32_t k = 0; k < metas_per_index_word; ++k)
        {
            emit_meta((word >> (k * meta_stride_bits)) & meta_index_mask);
        }
    };

    volatile std::uint32_t* cfg = get_cfg_pointer();

    // per tile math meta is 6 bits, 5 entries fit into 32bits, round up
    const std::uint32_t math_meta_size          = (kt_dim * ct_dim + 4) / 5;
    const std::uint32_t tt_l1_ptr* pre_meta_ptr = reinterpret_cast<std::uint32_t tt_l1_ptr*>(base_address_meta) + math_meta_size;
    const std::uint32_t iters                   = pre_meta_ptr[0];
    const std::uint32_t full_iters              = iters / metas_per_index_word; // whole index words
    const std::uint32_t rem_iters               = iters % metas_per_index_word; // metas in the trailing word
    const std::uint32_t full_blocks             = full_iters / 16;
    const bool odd_block                        = (full_iters % 16) >= 8;
    const std::uint32_t tt_l1_ptr* meta_ptr     = pre_meta_ptr + 1 + 2 * (full_iters / 8) + 2;

    wait_for_next_context(1);
    reset_config_context();

    if constexpr (clear_src)
    {
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    }
    TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_address_ADDR32, pre_meta_ptr[1] & meta_addr_base_mask);
    TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_cntx1_address_ADDR32, pre_meta_ptr[2] & meta_addr_base_mask);
    set_y_off(pre_meta_ptr + 1, meta_ptr[0]);
    TT_CFG_STORE(cfg, THCON_SEC1_REG3_Base_address_ADDR32, base_address_b);

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    std::uint32_t c = 0;

    for (std::uint32_t b = 0; b < full_blocks; ++b)
    {
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_cntx2_address_ADDR32, pre_meta_ptr[3 + 4 * b] & meta_addr_base_mask);
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_cntx3_address_ADDR32, pre_meta_ptr[4 + 4 * b] & meta_addr_base_mask);
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, unp_cfg_ctxt_offset_alt);
        set_y_off(pre_meta_ptr + 3 + 4 * b, meta_ptr[c]);
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_address_ADDR32, pre_meta_ptr[5 + 4 * b] & meta_addr_base_mask);
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_cntx1_address_ADDR32, pre_meta_ptr[6 + 4 * b] & meta_addr_base_mask);
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, unp_cfg_ctxt_offset_base);
        set_y_off(pre_meta_ptr + 5 + 4 * b, meta_ptr[c]);
    }

    if (odd_block)
    {
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_cntx2_address_ADDR32, pre_meta_ptr[3 + 4 * full_blocks] & meta_addr_base_mask);
        TT_CFG_STORE(cfg, THCON_SEC0_REG3_Base_cntx3_address_ADDR32, pre_meta_ptr[4 + 4 * full_blocks] & meta_addr_base_mask);
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, unp_cfg_ctxt_offset_alt);
        set_y_off(pre_meta_ptr + 3 + 4 * full_blocks, meta_ptr[c]);
    }

    for (; c < full_iters; ++c)
    {
        emit_word(meta_ptr[c]);
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t j = 0; j < rem_iters; ++j)
    {
        emit_meta(meta & meta_index_mask);
        meta >>= meta_stride_bits;
    }

    if constexpr (ct_dim == 1 && finalize)
    {
        TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
        TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
        TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 1, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    }

    t6_semaphore_get(semaphore::UNPACK_SYNC);

    wait_for_next_context(1);
    reset_config_context();

    // Put the counters back the way init left them, ready for the next call: zero the Y counter the
    // per-face SET_Y writes moved and the ZW pair, but not X, so init's CH1 X ends survive.
    TTI_SETADCXY(p_setadc::UNP_A, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
}
