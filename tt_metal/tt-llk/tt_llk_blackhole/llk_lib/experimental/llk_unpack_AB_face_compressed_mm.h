// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_code_sequence.h"
#include "ckernel_defs.h"
#include "ckernel_globals.h"
#include "ckernel_ops.h"
#include "ckernel_template.h"
#include "cunpack_common.h"

using namespace ckernel;
using namespace ckernel::unpacker;

// The unpack code sequence: one character is one instruction, recorded into the replay buffer at init (see
// instr_for_code in _llk_unpack_AB_face_compressed_mm_mop_config_ for the encodings):
//
//   0   unpack the next weight face through the bfp2 context (UNPACR into SrcA, Ch0Y += 1)
//   1   the same through the bfp4 context
//   S   CLR_SRC, which has to precede any face whose format differs from the one before it
//   B   unpack the next activation block into SrcB (Ch0Y += 2, Ch0Z += 2)
//
// The header table asks for 0, 1, S0, S1, B0 and B1, issuing SB0 and SB1 through the MOP instead; the
// pairs table asks for the sixteen three-transition concatenations its four face bits can produce, from
// 000 up to S1S0S1. All twenty-three fragments are substrings of the thirty-two characters below, which is
// the entire replay buffer -- on this thread nothing else uses it, so it starts at offset 0.
static constexpr auto _llk_unpack_AB_face_compressed_mm_seq_ =
    ckernel::code_seq::make_sequence<0 /* replay_base */, ckernel::REPLAY_BUF_SIZE>("SB0B1S0S1S0S111S00S1S000S11S0S11");

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
// The two tables below decode overlapping slices of it: the header table takes bits 0-2 and issues the
// transition into face 1 plus the optional B, the pairs table takes bits 2-5 as three sliding 2-bit windows
// and issues faces 2-4. Bit 5 therefore belongs to meta k+1 as much as to meta k, so masking the window
// with 0b11111 would silently unpack every fourth face as bfp2.
//
// The shared bit is a producer invariant nothing on this side can check: word w bit 0 must equal word w-1
// bit 30, the two spellings of that one face format. encode_meta in test_matmul_face_compressed.py upholds
// it; a hand-built meta that does not corrupts silently.

static constexpr auto _llk_unpack_AB_face_compressed_mm_header_table_ = ckernel::code_seq::make_table<_llk_unpack_AB_face_compressed_mm_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        // A face's bit is its unpacker context: 0 bfp2, 1 bfp4. The previous face rides along so the
        // two can be compared -- when the format changes, an S (CLR_SRC) has to be issued first to
        // stall the unpacker before the context switches. The header bit adds the group's SrcB unpack.
        const std::uint32_t face0  = m & 0b1;        // previous face
        const std::uint32_t header = (m >> 1) & 0b1; // also unpack SrcB
        const std::uint32_t face1  = (m >> 2) & 0b1; // this face
        switch ((header << 2) | (face1 << 1) | face0)
        {
            case 0b000:
                return find("0"); // bfp2 -> bfp2
            case 0b001:
                return find("S0"); // bfp4 -> bfp2
            case 0b010:
                return find("S1"); // bfp2 -> bfp4
            case 0b011:
                return find("1"); // bfp4 -> bfp4
            case 0b100:
                return find("B0"); // bfp2 -> bfp2, with B
            case 0b101:
                return TT_OP_MOP(p_mop::MASK_LOOP, 0, 0x0); // bfp4 -> bfp2, with B (SB0)
            case 0b110:
                return TT_OP_MOP(p_mop::MASK_LOOP, 0, 0x1); // bfp2 -> bfp4, with B (SB1)
            case 0b111:
                return find("B1"); // bfp4 -> bfp4, with B
            default:
                return 0;
        }
    });

static constexpr auto _llk_unpack_AB_face_compressed_mm_pairs_table_ = ckernel::code_seq::make_table<_llk_unpack_AB_face_compressed_mm_seq_, 64>(
    [](const std::uint32_t m, auto find) -> std::uint32_t
    {
        // faces1234 holds the group's four face formats, same encoding as the header table. That table
        // issued face1, so this one issues faces 2..4 and concatenates their fragments into a single
        // handle. Every transition needs its predecessor to decide the stall, so the 2-bit windows
        // overlap -- low bit the previous face, high bit this face -- and the shift is by one, not two.
        auto pair_frag = [](std::uint32_t transition) -> const char*
        {
            switch (transition)
            {
                case 0b00:
                    return "0"; // bfp2 -> bfp2
                case 0b01:
                    return "S0"; // bfp4 -> bfp2
                case 0b10:
                    return "S1"; // bfp2 -> bfp4
                case 0b11:
                    return "1"; // bfp4 -> bfp4
                default:
                    return "";
            }
        };
        std::uint32_t faces1234 = m >> 2;
        char needle[8]          = {};
        std::uint32_t len       = 0;
        for (std::uint32_t i = 0; i < 3; ++i)
        {
            for (const char* p = pair_frag(faces1234 & 0x3); *p != '\0'; ++p)
            {
                needle[len++] = *p;
            }
            faces1234 >>= 1;
        }
        return find(needle);
    });

/**
 * @brief Record the unpack code sequence into the replay buffer and program the MOP that replays it.
 *
 * One instruction per character: '0' and '1' unpack a bfp2 or bfp4 face into SrcA, 'S' is the CLR_SRC stall
 * a format switch needs first, and 'B' unpacks the activation block into SrcB. The decode tables hand the
 * execute loop replay handles naming runs of these.
 *
 * @note Called from @ref _llk_unpack_AB_face_compressed_mm_init_ rather than directly.
 */
inline void _llk_unpack_AB_face_compressed_mm_mop_config_()
{
    auto instr_for_code = [](char code)
    {
        switch (code)
        {
            case '0':
                TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 0, 1, 1);
                break; // Ch0Y += 1
            case '1':
                TTI_UNPACR_COMMON_EXPLICIT_CONTEXT_AND_COUNTER(SrcA, 0b00'00'01'00, 1, 2, 1);
                break; // Ch0Y += 1
            case 'S':
                TTI_UNPACR_NOP(SrcA, 0, 0, 0, 0, 1, 0, 0, p_unpacr_nop::CLR_SRC);
                break;
            case 'B':
                TTI_UNPACR_COMMON(SrcB, 0b00'00'10'10, 1);
                break; // Ch0Y += 2 Ch0Z += 2
            default:
                LLK_ASSERT(false, "Invalid code for unpack instruction");
                break;
        }
    };

    _llk_unpack_AB_face_compressed_mm_seq_.load(instr_for_code);

    constexpr std::uint32_t SB  = _llk_unpack_AB_face_compressed_mm_seq_.fragment("SB");
    constexpr std::uint32_t op0 = _llk_unpack_AB_face_compressed_mm_seq_.fragment("0");
    constexpr std::uint32_t op1 = _llk_unpack_AB_face_compressed_mm_seq_.fragment("1");

    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,      // unpackB    = true
        false,     // unpackHalo = false
        SB,        // A
        TT_OP_NOP, // A1 (unused)
        TT_OP_NOP, // A2 (unused)
        TT_OP_NOP, // A3 (unused)
        SB,        // skipA
        op0,       // B
        op1        // skipB
    );
    tmp.program();
}

// The two BFP precisions need their own address counters so one instruction stream can alternate format
// face to face, and the unpacker has only one set of its own. It borrows other threads' ADC sets via the
// SETADC thread-id override: bfp2 counts in the math thread's set, bfp4 in the pack thread's. That is why
// every SETADC* site below holds mutex::THREAD2_ADC -- see ckernel_common_ops.h for the override macros.

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

    _llk_unpack_AB_face_compressed_mm_mop_config_();

    constexpr std::uint32_t unpA_x_end = FACE_R_DIM * FACE_C_DIM - 1;
    const std::uint32_t unpB_x_end     = 4 * unpB_face_r_dim * FACE_C_DIM - 1;

    t6_mutex_acquire(mutex::THREAD2_ADC);
    // the thread override covers CH0 XY and CH1 X, CH0 ZW stays shared
    // reset the borrowed counters, bfp2's in the math set and bfp4's in the pack set
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_MATH, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_PACK, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY)); // unp1's own set, unp0's rides along
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW)); // ZW has no override, both formats share it
    // set CH1 X for both unpackers; bfp4 and bfp2 sit in different thread overrides, so both get set
    TTI_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, p_setadc::THREAD_OVRD_MATH, unpA_x_end);
    TTI_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_1, p_setadc::SET_X, p_setadc::THREAD_OVRD_PACK, unpA_x_end);
    TT_SETADCXX(p_setadc::UNP_B, unpB_x_end, 0x0);
    t6_mutex_release(mutex::THREAD2_ADC);
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

    t6_mutex_acquire(mutex::THREAD2_ADC);
    // reset CH0 XY counters for both unpackers to leave them in a clean state
    TTI_SETADCXY(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::XY));
    t6_mutex_release(mutex::THREAD2_ADC);
}

/**
 * @brief Unpack the activation block into SrcB and the compressed weight faces into SrcA.
 *
 * Streams the meta buffer's index words, pushing one replay handle pair per meta, and reloads the
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
    // unpacker context's Base_address, and the top byte is a Y offset, applied as that context's SET_Y.
    constexpr std::uint32_t meta_addr_base_mask   = 0x00FFFFFF;
    constexpr std::uint32_t meta_addr_y_off_shift = 24;

    // Geometry of the index words, per the layout block above the decode tables. The stride is narrower
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

    // Apply the Y offsets of a consecutive bfp2/bfp4 pair of meta address words to the two borrowed Y
    // counters, so each context reads from the right row of its base. bfp2 counts in the math thread's ADC
    // set and bfp4 in the pack thread's, so each needs its own write. Caller holds the mutex.
    auto set_y_off = [](const std::uint32_t* addr_pair)
    {
        const std::uint32_t bfp2_y = addr_pair[0] >> meta_addr_y_off_shift;
        const std::uint32_t bfp4_y = addr_pair[1] >> meta_addr_y_off_shift;
        TT_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, p_setadc::THREAD_OVRD_MATH, bfp2_y);
        TT_SETADC_THREAD_OVERRIDE(p_setadc::UNP0, p_setadc::CH_0, p_setadc::SET_Y, p_setadc::THREAD_OVRD_PACK, bfp4_y);
    };

    auto emit_word = [](std::uint32_t meta)
    {
        std::uint32_t idx0 = (meta >> (0 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx1 = (meta >> (1 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx2 = (meta >> (2 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx3 = (meta >> (3 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx4 = (meta >> (4 * meta_stride_bits)) & meta_index_mask;
        std::uint32_t idx5 = (meta >> (5 * meta_stride_bits)) & meta_index_mask;

        std::uint32_t data0  = _llk_unpack_AB_face_compressed_mm_header_table_[idx0];
        std::uint32_t data1  = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx0];
        std::uint32_t data2  = _llk_unpack_AB_face_compressed_mm_header_table_[idx1];
        std::uint32_t data3  = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx1];
        std::uint32_t data4  = _llk_unpack_AB_face_compressed_mm_header_table_[idx2];
        std::uint32_t data5  = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx2];
        std::uint32_t data6  = _llk_unpack_AB_face_compressed_mm_header_table_[idx3];
        std::uint32_t data7  = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx3];
        std::uint32_t data8  = _llk_unpack_AB_face_compressed_mm_header_table_[idx4];
        std::uint32_t data9  = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx4];
        std::uint32_t data10 = _llk_unpack_AB_face_compressed_mm_header_table_[idx5];
        std::uint32_t data11 = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx5];

        ckernel::instrn_buffer[0] = data0;
        ckernel::instrn_buffer[0] = data1;
        ckernel::instrn_buffer[0] = data2;
        ckernel::instrn_buffer[0] = data3;
        ckernel::instrn_buffer[0] = data4;
        ckernel::instrn_buffer[0] = data5;
        ckernel::instrn_buffer[0] = data6;
        ckernel::instrn_buffer[0] = data7;
        ckernel::instrn_buffer[0] = data8;
        ckernel::instrn_buffer[0] = data9;
        ckernel::instrn_buffer[0] = data10;
        ckernel::instrn_buffer[0] = data11;
    };

    volatile std::uint32_t* cfg = get_cfg_pointer();

    // per tile math meta is 6 bits, 5 entries fit into 32bits, round up
    const std::uint32_t math_meta_size = (kt_dim * ct_dim + 4) / 5;
    const std::uint32_t* pre_meta_ptr  = reinterpret_cast<std::uint32_t*>(base_address_meta) + math_meta_size;
    const std::uint32_t iters          = pre_meta_ptr[0];
    const std::uint32_t full_iters     = iters / metas_per_index_word; // whole index words
    const std::uint32_t rem_iters      = iters % metas_per_index_word; // metas in the trailing word
    const std::uint32_t full_blocks    = full_iters / 16;
    const bool odd_block               = (full_iters % 16) >= 8;
    const std::uint32_t* meta_ptr      = pre_meta_ptr + 1 + 2 * (full_iters / 8) + 2;

    wait_for_next_context(1);
    reset_config_context();

    if constexpr (clear_src)
    {
        TTI_UNPACR_NOP(SrcB, 0, 0, 0, 0, 0, 1, 0, p_unpacr_nop::CLR_SRC);
    }
    cfg[THCON_SEC0_REG3_Base_address_ADDR32]       = pre_meta_ptr[1] & meta_addr_base_mask;
    cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = pre_meta_ptr[2] & meta_addr_base_mask;
    t6_mutex_acquire(mutex::THREAD2_ADC);
    set_y_off(pre_meta_ptr + 1);
    t6_mutex_release(mutex::THREAD2_ADC);
    cfg[THCON_SEC1_REG3_Base_address_ADDR32] = base_address_b;

    semaphore_post(semaphore::UNPACK_SYNC);

    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    std::uint32_t c = 0;

    for (std::uint32_t b = 0; b < full_blocks; ++b)
    {
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        cfg[THCON_SEC0_REG3_Base_cntx2_address_ADDR32] = pre_meta_ptr[3 + 4 * b] & meta_addr_base_mask;
        cfg[THCON_SEC0_REG3_Base_cntx3_address_ADDR32] = pre_meta_ptr[4 + 4 * b] & meta_addr_base_mask;
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, unp_cfg_ctxt_offset_alt);
        t6_mutex_acquire(mutex::THREAD2_ADC);
        set_y_off(pre_meta_ptr + 3 + 4 * b);
        t6_mutex_release(mutex::THREAD2_ADC);
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        cfg[THCON_SEC0_REG3_Base_address_ADDR32]       = pre_meta_ptr[5 + 4 * b] & meta_addr_base_mask;
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = pre_meta_ptr[6 + 4 * b] & meta_addr_base_mask;
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, unp_cfg_ctxt_offset_base);
        t6_mutex_acquire(mutex::THREAD2_ADC);
        set_y_off(pre_meta_ptr + 5 + 4 * b);
        t6_mutex_release(mutex::THREAD2_ADC);
    }

    if (odd_block)
    {
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        cfg[THCON_SEC0_REG3_Base_cntx2_address_ADDR32] = pre_meta_ptr[3 + 4 * full_blocks] & meta_addr_base_mask;
        cfg[THCON_SEC0_REG3_Base_cntx3_address_ADDR32] = pre_meta_ptr[4 + 4 * full_blocks] & meta_addr_base_mask;
        for (std::uint32_t i = 0; i < 4; ++i, ++c)
        {
            emit_word(meta_ptr[c]);
        }
        TTI_SETC16(UNPACK_MISC_CFG_CfgContextOffset_0_ADDR32, unp_cfg_ctxt_offset_alt);
        t6_mutex_acquire(mutex::THREAD2_ADC);
        set_y_off(pre_meta_ptr + 3 + 4 * full_blocks);
        t6_mutex_release(mutex::THREAD2_ADC);
    }

    for (; c < full_iters; ++c)
    {
        emit_word(meta_ptr[c]);
    }
    std::uint32_t meta = meta_ptr[full_iters];
    for (std::uint32_t j = 0; j < rem_iters; ++j)
    {
        std::uint32_t idx0        = meta & meta_index_mask;
        std::uint32_t data0       = _llk_unpack_AB_face_compressed_mm_header_table_[idx0];
        std::uint32_t data1       = _llk_unpack_AB_face_compressed_mm_pairs_table_[idx0];
        ckernel::instrn_buffer[0] = data0;
        ckernel::instrn_buffer[0] = data1;
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

    // Put the counters back the way init left them, ready for the next call: zero the Y counters the
    // per-face SET_Y writes moved and the shared ZW pair, but not X, so init's CH1 X ends survive.
    t6_mutex_acquire(mutex::THREAD2_ADC);
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_MATH, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));
    TTI_SETADCXY_THREAD_OVERRIDE(p_setadc::UNP_AB, p_setadc::THREAD_OVRD_PACK, 0, 0, 0, 0, SETADC_CH01(p_setadc::Y));
    TTI_SETADCZW(p_setadc::UNP_AB, 0, 0, 0, 0, SETADC_CH01(p_setadc::ZW));
    t6_mutex_release(mutex::THREAD2_ADC);
}
