// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SFPU decoders of GGUF weight formats: one page (experimental/ckernel_gguf_page.h) to one Float32 [K][N] tile in DEST,
// scales applied, equal to ggml's dequantization. Needs a 32-bit DEST (fp32_dest_acc_en).
//
// DEST tiles from the call's tile: 0 output, 1 page (unpacked as UInt32), 2 scratch for the per-lane scales S and
// offsets M. Output o of SFPU lane (r, c) is element k = 16h + 4((o / 2) % 4) + r, n = 16a + 2c + b, with h = o / 16,
// a = (o / 8) % 2, b = o % 2. Word-vector v of the page (rows 4 (v / 2) .. + 3, columns of parity v % 2) holds the
// codes of lane (r, c)'s outputs; in the scale rows (from scale_offset / 64), word 2c + p holds the fields of columns
// 16p + 2c and 16p + 2c + 1 (low, high half).
// Phase 1 (params) writes S[a][b][h] and M[a][b][h] to the scratch tile. Phase 2 (body) writes q * S + M (linear
// formats) or v(q) * S (value-table and codebook formats).
// MXFP4 and NVFP4 differ (ckernel_sfpu_gguf_decode_fp4.h): their codes are block-float datums in the 16-bit rows
// that tile 0's rows address, their scratch is in tile 1 past the page's rows.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "experimental/ckernel_gguf_page.h"
#include "sfpi.h"
#include "sfpu/experimental/ckernel_sfpu_gguf_decode_common.h"
#include "sfpu/experimental/ckernel_sfpu_gguf_decode_fp4.h"
#include "sfpu/experimental/ckernel_sfpu_gguf_decode_iq.h"
#include "sfpu/experimental/ckernel_sfpu_gguf_decode_iq4.h"
#include "sfpu/experimental/ckernel_sfpu_gguf_decode_linear.h"

namespace ckernel::sfpu
{
namespace gguf
{

using namespace ckernel::gguf;

template <int FMT>
struct traits
{
    static constexpr int lo      = FMT == Q8_0 ? 8 : (FMT == Q3_K || FMT == Q2_K || FMT == TQ2_0 || FMT == Q2_0) ? 2 : FMT == Q1_0 ? 1 : 4;
    static constexpr int hi      = (FMT == Q5_K || FMT == Q3_K || FMT == Q5_0 || FMT == Q5_1) ? 1 : FMT == Q6_K ? 2 : 0;
    static constexpr bool pair   = FMT == Q5_K || FMT == Q3_K || FMT == Q1_0 || FMT == Q5_0 || FMT == Q5_1;
    static constexpr bool has_h  = FMT == NVFP4 || FMT == Q6_K || FMT == Q3_K || FMT == Q2_K;
    static constexpr bool linear = !(FMT == IQ4_NL || FMT == IQ4_XS || FMT == MXFP4 || FMT == NVFP4);
};

// Phase 1 of one tile. TI = the tile's index in its group of K tiles.
template <int FMT, int TI>
inline void _calculate_gguf_decode_params_()
{
    if constexpr (FMT == IQ2_XXS)
    {
        iq2_xxs_params();
    }
    else if constexpr (FMT == IQ1_S)
    {
        iq1_s_params();
    }
    else if constexpr (FMT == IQ2_S)
    {
        iq2_s_params();
    }
    else if constexpr (FMT == IQ3_S)
    {
        iq3_s_params();
    }
    else if constexpr (FMT == IQ3_XXS)
    {
        iq3_xxs_params();
    }
    else if constexpr (FMT == IQ2_XS)
    {
        iq2_xs_params<TI>();
    }
    else if constexpr (FMT == IQ1_M)
    {
        iq1_m_params<TI>();
    }
    else if constexpr (FMT == MXFP4)
    {
        fp4_bfp_mx_params<TI>();
    }
    else if constexpr (FMT == NVFP4)
    {
        fp4_bfp_params();
    }
    else if constexpr (FMT == IQ4_XS)
    {
        iq4_params<FMT, TI>();
    }
    else
    {
        linear_params<FMT, TI>();
    }
}

// IQ2_XS runs phase 2 as a separate call, one copy shared by every tile index.
template <int FMT>
constexpr bool two_calls = FMT == IQ2_XS;

// The linear formats' macro bodies count cycles (stores arbitrated) where an unpack to DEST may run beside phase 2
// (BESIDE_UNPACK); otherwise they use the shorter instruction-counted streams. Q2_K keeps its instruction-counted
// body beside an unpack too and waits for the copy instead (sfpu_gguf_decode_body_unarbitrated), which costs less.
template <int FMT, bool BESIDE_UNPACK>
constexpr bool cycle_body = BESIDE_UNPACK && FMT != Q2_K;

// Phase 2 of one tile (IQ2_XS: of every tile).
template <int FMT, int TI, bool BESIDE_UNPACK>
inline void _calculate_gguf_decode_body_()
{
    if constexpr (FMT == IQ2_XXS)
    {
        iq2_stage<false>();
    }
    else if constexpr (FMT == IQ1_S)
    {
        iq1_stage<false>();
    }
    else if constexpr (FMT == IQ2_S)
    {
        iq2_stage<true>();
    }
    else if constexpr (FMT == IQ3_S)
    {
        iq3_s_body();
    }
    else if constexpr (FMT == IQ3_XXS)
    {
        iq3_xxs_init();
        iq3_xxs_body();
    }
    else if constexpr (FMT == IQ2_XS)
    {
        iq2_stage<true, true>();
    }
    else if constexpr (FMT == IQ1_M)
    {
        iq1_stage<true>();
    }
    else if constexpr (FMT == TQ1_0)
    {
        tq1_0_body<TI>();
    }
    else if constexpr (macro_body<FMT> && macro_q4<FMT>)
    {
        body_macro_q4<false, cycle_body<FMT, BESIDE_UNPACK>>();
    }
    else if constexpr (FMT == MXFP4)
    {
        body_fp4_bfp_mx();
    }
    else if constexpr (FMT == NVFP4)
    {
        body_fp4_bfp();
    }
    else if constexpr (macro_body<FMT>)
    {
        body_macro<FMT, traits<FMT>::pair ? TI % 2 : 0, cycle_body<FMT, BESIDE_UNPACK>>();
    }
    else if constexpr (iq4_macro_body<FMT> && FMT == IQ4_XS)
    {
        body_macro_iq4_shared();
    }
    else if constexpr (iq4_macro_body<FMT>)
    {
        body_macro_iq4();
    }
    else
    {
        if constexpr (FMT == IQ4_NL || FMT == IQ4_XS)
        {
            iq4_init();
        }
        body<FMT, traits<FMT>::pair ? TI % 2 : 0>();
    }
}

// Load-macro setup of the formats whose phase 2 runs through SFPLOADMACRO.
template <int FMT, bool BESIDE_UNPACK>
inline void macro_init()
{
    if constexpr (FMT == MXFP4)
    {
        macro_init_fp4_bfp_mx();
    }
    if constexpr (FMT == NVFP4)
    {
        macro_init_fp4_bfp();
    }
    if constexpr (iq4_macro_body<FMT>)
    {
        macro_init_iq4();
    }
    if constexpr (macro_body<FMT>)
    {
        macro_init_linear<FMT, cycle_body<FMT, BESIDE_UNPACK>>();
    }
}

} // namespace gguf

// Points the SFPU at DEST tile dst_index plus `rows` rows (a page `rows` rows into a tile of several pages).
inline void sfpu_gguf_decode_start(const std::uint32_t dst_index, const std::uint32_t rows)
{
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (dst_index << 6) + rows + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void sfpu_gguf_decode_done()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <int FMT, int TI, bool PARAMS, bool BODY, bool BESIDE_UNPACK>
inline void sfpu_gguf_decode_tile_ti(const std::uint32_t dst_index, const std::uint32_t rows)
{
    sfpu_gguf_decode_start(dst_index, rows);
    if constexpr (PARAMS && BODY && gguf::k4_regs<FMT>)
    {
        gguf::k4_params_regs<TI>();
        gguf::body_macro_q4<true, gguf::cycle_body<FMT, BESIDE_UNPACK>>();
    }
    else if constexpr (PARAMS && !BODY && gguf::k4_regs<FMT>)
    {
        // Phase 1 alone (e.g. a pack-thread decode that waits between the phases): a = 0's S / M stay in L4..L7 for
        // the body of the next call, so nothing else may run on this SFPU in between.
        gguf::k4_params_regs<TI>();
    }
    else if constexpr (BODY && !PARAMS && gguf::k4_regs<FMT>)
    {
        gguf::body_macro_q4<true, gguf::cycle_body<FMT, BESIDE_UNPACK>>();
    }
    else
    {
        if constexpr (PARAMS)
        {
            gguf::_calculate_gguf_decode_params_<FMT, TI>();
        }
        if constexpr (BODY && !gguf::two_calls<FMT>)
        {
            gguf::_calculate_gguf_decode_body_<FMT, TI, BESIDE_UNPACK>();
        }
    }
    sfpu_gguf_decode_done();
}

/**
 * @brief Set up the SFPU for @ref sfpu_gguf_decode_tile of format FMT.
 *
 * @tparam FMT: ckernel::gguf::Format of the pages.
 * @tparam BESIDE_UNPACK: Whether an unpack to DEST may run beside phase 2; the same value as the tile calls'.
 * @note Call after _llk_math_eltwise_unary_sfpu_init_, and again after other SFPU code that programs the load macros
 *       or the programmable constants (IQ4_NL / IQ4_XS: 0..2; the others: 2).
 */
template <int FMT, bool BESIDE_UNPACK = true>
inline void sfpu_gguf_decode_init()
{
    gguf::macro_init<FMT, BESIDE_UNPACK>();
}

/**
 * @brief Whether phase 2 of format FMT has DEST stores that are not arbitrated against the unpackers' DEST writes
 *        (SFPLOADMACRO with delays that count instructions), so no unpack to DEST may run beside it. No SFPLOADMACRO
 *        body may run beside the FPU's DEST writes.
 *
 * @tparam FMT: ckernel::gguf::Format of the pages.
 * @tparam BESIDE_UNPACK: As @ref sfpu_gguf_decode_tile.
 */
template <int FMT, bool BESIDE_UNPACK = true>
constexpr bool sfpu_gguf_decode_body_unarbitrated()
{
    using namespace gguf;
    return iq4_macro_body<FMT> || (macro_body<FMT> && !cycle_body<FMT, BESIDE_UNPACK>);
}

/**
 * @brief Group step of Q4_K, Q5_K, Q3_K and Q2_K: convert the fp16 d (and dmin) of a group once, into tile
 *        dst_index + 1 past the page, for every tile of it decoded at the same DEST tiles (Q4_K, Q5_K, Q3_K: also
 *        unpack its 6-bit scales, and mins). Nothing for the other formats.
 *
 * @tparam FMT: ckernel::gguf::Format of the page.
 * @param dst_index: As @ref sfpu_gguf_decode_tile; the page in tile dst_index + 1 is any page of the group.
 * @param rows: As @ref sfpu_gguf_decode_tile; the same value as the group's tile calls.
 * @note Call before the first tile of a group that @ref sfpu_gguf_decode_tile decodes at dst_index and rows; nothing
 *       else may write scratch tile dst_index + 2 (entries 0..15) or tile dst_index + 1 past its page until the
 *       group ends.
 */
template <int FMT>
inline void sfpu_gguf_decode_group(const std::uint32_t dst_index, const std::uint32_t rows = 0)
{
    if constexpr (FMT == gguf::Q4_K || FMT == gguf::Q5_K)
    {
        sfpu_gguf_decode_start(dst_index, rows);
        gguf::k4_group<FMT>();
        sfpu_gguf_decode_done();
    }
    else if constexpr (FMT == gguf::Q2_K)
    {
        sfpu_gguf_decode_start(dst_index, rows);
        gguf::q2_group<FMT>();
        sfpu_gguf_decode_done();
    }
    else if constexpr (FMT == gguf::Q3_K)
    {
        sfpu_gguf_decode_start(dst_index, rows);
        gguf::q3_group<FMT>();
        sfpu_gguf_decode_done();
    }
}

/**
 * @brief Decode one GGUF weight tile into a Float32 [K][N] tile in DEST, scales applied (ggml's dequantization).
 *
 * @tparam FMT: ckernel::gguf::Format of the page.
 * @tparam PARAMS, BODY: Run phase 1 (the scales) and phase 2 (the codes). A decoded tile needs both, in one call or in
 *                       two (phase 1, then phase 2 with no other SFPU code in between); one alone is for timing it.
 * @tparam BESIDE_UNPACK: Whether an unpack to DEST may run beside phase 2 (where
 *                        @ref sfpu_gguf_decode_body_unarbitrated is false); false selects shorter phase 2 streams
 *                        for decodes with no DEST writer beside them.
 * @param dst_index: DEST tile of the output. The page must be in tile dst_index + 1 (unpacked to DEST as UInt32);
 *                   tile dst_index + 2 is scratch.
 * @param ti: The tile's index in its group of K tiles (page_layout<FMT>().group_tiles).
 * @param rows: DEST row offset of the page inside tile dst_index + 1 (several pages unpacked as one tile); the output
 *              tile and the scratch tile move with it.
 * @note Needs a 32-bit DEST and @ref sfpu_gguf_decode_init (Q4_K, Q5_K, Q3_K, Q2_K: and @ref sfpu_gguf_decode_group
 *       for the tile's group). Overwrites the programmable constants and may record into the replay buffer.
 */
template <int FMT, bool PARAMS = true, bool BODY = true, bool BESIDE_UNPACK = true>
inline void sfpu_gguf_decode_tile(const std::uint32_t dst_index, const std::uint32_t ti, const std::uint32_t rows = 0)
{
    using namespace gguf;
    if constexpr (FMT == MXFP4 || FMT == Q1_0 || FMT == Q5_0 || FMT == Q5_1)
    {
        if (ti & 1)
        {
            sfpu_gguf_decode_tile_ti<FMT, 1, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
        }
        else
        {
            sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
        }
    }
    else if constexpr (!(FMT == Q6_K || FMT == Q4_K || FMT == Q5_K || FMT == Q3_K || FMT == Q2_K || FMT == IQ4_XS || FMT == IQ2_XS || FMT == IQ1_M ||
                         FMT == TQ1_0))
    {
        sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
    }
    else
    {
        switch (ti)
        {
            case 0:
                sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
                break;
            case 1:
                sfpu_gguf_decode_tile_ti<FMT, 1, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
                break;
            case 2:
                sfpu_gguf_decode_tile_ti<FMT, 2, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
                break;
            case 3:
                sfpu_gguf_decode_tile_ti<FMT, 3, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
                break;
            case 4:
                sfpu_gguf_decode_tile_ti<FMT, 4, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
                break;
            case 5:
                sfpu_gguf_decode_tile_ti<FMT, 5, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
                break;
            case 6:
                sfpu_gguf_decode_tile_ti<FMT, 6, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
                break;
            default:
                sfpu_gguf_decode_tile_ti<FMT, 7, PARAMS, BODY, BESIDE_UNPACK>(dst_index, rows);
                break;
        }
    }
    if constexpr (BODY && gguf::two_calls<FMT>)
    {
        sfpu_gguf_decode_start(dst_index, rows);
        gguf::_calculate_gguf_decode_body_<FMT, 0, BESIDE_UNPACK>();
        sfpu_gguf_decode_done();
    }
}

} // namespace ckernel::sfpu
