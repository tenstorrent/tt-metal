// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SFPU decoders of GGUF weight formats: one page (experimental/ckernel_gguf_page.h) to one Float32 [K][N] tile in DEST
// with the block scales applied. Needs a 32-bit DEST (fp32_dest_acc_en).
//
// DEST, from the tile the call starts at (sfpu_gguf_decode_tile): tile 0 is the output, tile 1 the page (unpacked as
// UInt32), tile 2 scratch for the per-lane scales S and offsets M. Output o of SFPU lane (r, c) is element
// k = 16h + 4((o / 2) % 4) + r, n = 16a + 2c + b of the tile, with h = o / 16, a = (o / 8) % 2, b = o % 2: the K
// row and the N column (the weight's row) of a [K][N] tile, as a matmul's SrcA wants it.
//
// The page's code words sit where those lanes load them: word-vector v (DEST rows 4 (v / 2) .. + 3 of the page tile,
// columns of parity v % 2) holds, in lane (r, c), the codes of that lane's outputs. Its scale fields start at row
// R = scale_offset / 64, regrouped so that word w = 2c + p of a row carries the fields of columns 16p + 2c and
// 16p + 2c + 1 (low half, high half).
//
// Phase 1 (params) reads the scale fields and writes S[a][b][h] and M[a][b][h] per lane to the scratch tile: a 4-row
// block of scale rows is loaded four times and transposed (transp4), which leaves each row replicated over the four
// SFPU row groups; fp16 fields come converted (f16_at) or as the fp32 rows ckernel_gguf_page.h prepare_scales adds.
// Phase 2 (body) extracts each output's code and writes q * S + M (linear formats) or v(q) * S (value-table and
// codebook formats). Results equal ggml's dequantization bit for bit: every product of a code and a scale is exact in
// fp32, and q * S + M rounds once like ggml's d * q - m.

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
    else if constexpr (FMT == MXFP4 || FMT == NVFP4)
    {
        fp4_params<FMT, TI>();
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

// IQ2_XS decodes in two calls, its tile-index phase 1 and the phase 2 below, shared by every tile index: with phase 2
// inlined into each of the eight phase 1 instantiations its decode wrote zeros in a matmul kernel.
template <int FMT>
constexpr bool two_calls = FMT == IQ2_XS;

// Phase 2 of one tile (IQ2_XS: of every tile).
template <int FMT, int TI>
inline void _calculate_gguf_decode_body_()
{
    if constexpr (FMT == IQ2_XXS)
    {
        iq2_xxs_init();
        iq2_xxs_body<false>();
    }
    else if constexpr (FMT == IQ1_S)
    {
        iq1_s_body<false>();
    }
    else if constexpr (FMT == IQ2_S)
    {
        iq2_xxs_init();
        iq2_xxs_body<true>();
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
        iq2_xxs_init();
        iq2_xxs_body<true, true>();
    }
    else if constexpr (FMT == IQ1_M)
    {
        iq1_s_body<true>();
    }
    else if constexpr (FMT == TQ1_0)
    {
        tq1_0_body<TI>();
    }
    else if constexpr (macro_body<FMT> && macro_q4<FMT>)
    {
        body_macro_q4();
    }
    else if constexpr (fp4_macro_body<FMT>)
    {
        body_macro_fp4<FMT>();
    }
    else if constexpr (macro_body<FMT>)
    {
        body_macro<FMT, traits<FMT>::pair ? TI % 2 : 0>();
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
        else if constexpr (FMT == MXFP4 || FMT == NVFP4)
        {
            e2m1_init();
        }
        body<FMT, traits<FMT>::pair ? TI % 2 : 0>();
    }
}

// Load-macro setup of the formats whose phase 2 runs through SFPLOADMACRO.
template <int FMT>
inline void macro_init()
{
    if constexpr (fp4_macro_body<FMT>)
    {
        macro_init_fp4();
    }
    if constexpr (iq4_macro_body<FMT>)
    {
        macro_init_iq4();
    }
    if constexpr (macro_body<FMT>)
    {
        macro_init_linear<FMT>();
    }
}

} // namespace gguf

// The SFPU at DEST tile dst_index, `rows` rows further on (a page tile holding several pages: the page `rows` rows
// into tile dst_index + 1, the output over the rows before it).
inline void sfpu_gguf_decode_start(const std::uint32_t dst_index, const std::uint32_t rows)
{
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, (dst_index << 6) + rows + get_dest_buffer_base());
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void sfpu_gguf_decode_done()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <int FMT, int TI, bool PARAMS, bool BODY>
inline void sfpu_gguf_decode_tile_ti(const std::uint32_t dst_index, const std::uint32_t rows)
{
    sfpu_gguf_decode_start(dst_index, rows);
    if constexpr (PARAMS)
    {
        gguf::_calculate_gguf_decode_params_<FMT, TI>();
    }
    if constexpr (BODY && !gguf::two_calls<FMT>)
    {
        gguf::_calculate_gguf_decode_body_<FMT, TI>();
    }
    sfpu_gguf_decode_done();
}

/**
 * @brief Set up the SFPU for @ref sfpu_gguf_decode_tile of format FMT.
 *
 * @tparam FMT: ckernel::gguf::Format of the pages.
 * @note Call after _llk_math_eltwise_unary_sfpu_init_, and again after other SFPU code that programs the load macros
 *       or programmable constant 2.
 */
template <int FMT>
inline void sfpu_gguf_decode_init()
{
    gguf::macro_init<FMT>();
}

/**
 * @brief Decode one GGUF weight tile into a Float32 [K][N] tile in DEST, scales applied (ggml's dequantization).
 *
 * @tparam FMT: ckernel::gguf::Format of the page.
 * @tparam PARAMS, BODY: Run phase 1 (the scales) and phase 2 (the codes). A decoded tile needs both; one alone is for
 *                       timing it.
 * @param dst_index: DEST tile of the output. The page must be in tile dst_index + 1 (unpacked to DEST as UInt32);
 *                   tile dst_index + 2 is scratch.
 * @param ti: The tile's index in its group of K tiles (page_layout<FMT>().group_tiles): the formats whose scales or
 *            page differ per tile select them by it.
 * @param rows: DEST row offset of the page inside tile dst_index + 1 (several pages unpacked as one tile); the output
 *              tile and the scratch tile move with it.
 * @note Needs a 32-bit DEST and @ref sfpu_gguf_decode_init. Overwrites the programmable constants and may record
 *       into the replay buffer.
 */
template <int FMT, bool PARAMS = true, bool BODY = true>
inline void sfpu_gguf_decode_tile(const std::uint32_t dst_index, const std::uint32_t ti, const std::uint32_t rows = 0)
{
    using namespace gguf;
    if constexpr (FMT == MXFP4 || FMT == Q1_0 || FMT == Q5_0 || FMT == Q5_1)
    {
        if (ti & 1)
        {
            sfpu_gguf_decode_tile_ti<FMT, 1, PARAMS, BODY>(dst_index, rows);
        }
        else
        {
            sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY>(dst_index, rows);
        }
    }
    else if constexpr (!(FMT == Q6_K || FMT == Q4_K || FMT == Q5_K || FMT == Q3_K || FMT == Q2_K || FMT == IQ4_XS || FMT == IQ2_XS || FMT == IQ1_M ||
                         FMT == TQ1_0))
    {
        sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY>(dst_index, rows);
    }
    else
    {
        switch (ti)
        {
            case 0:
                sfpu_gguf_decode_tile_ti<FMT, 0, PARAMS, BODY>(dst_index, rows);
                break;
            case 1:
                sfpu_gguf_decode_tile_ti<FMT, 1, PARAMS, BODY>(dst_index, rows);
                break;
            case 2:
                sfpu_gguf_decode_tile_ti<FMT, 2, PARAMS, BODY>(dst_index, rows);
                break;
            case 3:
                sfpu_gguf_decode_tile_ti<FMT, 3, PARAMS, BODY>(dst_index, rows);
                break;
            case 4:
                sfpu_gguf_decode_tile_ti<FMT, 4, PARAMS, BODY>(dst_index, rows);
                break;
            case 5:
                sfpu_gguf_decode_tile_ti<FMT, 5, PARAMS, BODY>(dst_index, rows);
                break;
            case 6:
                sfpu_gguf_decode_tile_ti<FMT, 6, PARAMS, BODY>(dst_index, rows);
                break;
            default:
                sfpu_gguf_decode_tile_ti<FMT, 7, PARAMS, BODY>(dst_index, rows);
                break;
        }
    }
    if constexpr (BODY && gguf::two_calls<FMT>)
    {
        sfpu_gguf_decode_start(dst_index, rows);
        gguf::_calculate_gguf_decode_body_<FMT, 0>();
        sfpu_gguf_decode_done();
    }
}

} // namespace ckernel::sfpu
