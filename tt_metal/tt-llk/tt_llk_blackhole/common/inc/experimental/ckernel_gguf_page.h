// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF weight pages for the Blackhole SFPU decoders: the format ids and the layout of a page in L1.

#pragma once

#include <cstdint>

namespace ckernel::gguf
{

enum Format : std::uint32_t
{
    Q8_0    = 1,
    Q4_0    = 2,
    IQ4_NL  = 3,
    MXFP4   = 4,
    NVFP4   = 5,
    Q6_K    = 6,
    Q4_K    = 7,
    Q5_K    = 8,
    Q3_K    = 9,
    Q2_K    = 10,
    IQ4_XS  = 11,
    TQ2_0   = 12,
    Q2_0    = 13,
    Q1_0    = 14,
    IQ2_XXS = 15,
    IQ1_S   = 16,
    IQ2_S   = 17,
    IQ3_S   = 18,
    IQ3_XXS = 19,
    IQ2_XS  = 20,
    IQ1_M   = 21,
    Q4_1    = 22,
    Q5_0    = 23,
    Q5_1    = 24,
    TQ1_0   = 25,
};

// A page is one K tile (two for the pair formats, eight for TQ1_0) of a 32-column N tile: a 32-bit tile of 16, 32 or 64
// rows of 16 words, unpacked to DEST as is.
//   [code_offset, code_offset + code_bytes): the code words, regrouped from the GGUF blocks so that SFPU lane (r, c)
//     finds its outputs' codes in one place; codebook formats keep their native index bytes here, with the level codes
//     of their grid entries in front (from byte 0).
//   From scale_offset: the group's scale fields, regrouped the same way: scale_rows native rows (the last fp16_rows of
//     them fp16 fields), then fp32 copies of those fields.
// Code and scale bytes are the weights' GGUF bytes, moved; the writer of the page fills in the level codes and fp32
// copies before it is unpacked.
struct PageLayout
{
    std::uint32_t group_tiles;  // K tiles sharing one set of scale fields (the decoder's tile index runs over them)
    std::uint32_t page_tiles;   // K tiles per page
    std::uint32_t code_bytes;   // native code bytes of a page
    std::uint32_t scale_rows;   // native 64-byte rows of scale fields per group
    std::uint32_t fp16_rows;    // the last scale rows: fp16 fields, each with an fp32 copy after the scale rows
    std::uint32_t code_offset;  // bytes before the code bytes (the codebook formats' level codes)
    std::uint32_t scale_offset; // bytes before the scale fields: the code area rounded up to 256 bytes
    std::uint32_t scale_bytes;  // scale fields with their fp32 copies
    std::uint32_t rows;         // 64-byte rows of the page: 16, 32 or 64
};

constexpr std::uint32_t round_up(const std::uint32_t value, const std::uint32_t step)
{
    return (value + step - 1) / step * step;
}

constexpr PageLayout make_layout(
    const std::uint32_t group_tiles,
    const std::uint32_t page_tiles,
    const std::uint32_t code_bytes,
    const std::uint32_t scale_rows,
    const std::uint32_t fp16_rows,
    const std::uint32_t code_offset = 0)
{
    const std::uint32_t scale_offset = round_up(code_offset + code_bytes, 256);
    const std::uint32_t scale_bytes  = fp16_rows ? (round_up(scale_rows, 4) + 8 * fp16_rows) * 64 : scale_rows * 64;
    std::uint32_t rows               = 16;
    while (rows * 64 < scale_offset + scale_bytes)
    {
        rows *= 2;
    }
    return {group_tiles, page_tiles, code_bytes, scale_rows, fp16_rows, code_offset, scale_offset, scale_bytes, rows};
}

constexpr PageLayout page_layout(const Format fmt)
{
    // clang-format off
    switch (fmt)
    {
        case Q8_0:    return make_layout(1, 1, 1024, 1, 0);
        case Q4_0:    return make_layout(1, 1, 512, 1, 0);
        case IQ4_NL:  return make_layout(1, 1, 512, 1, 0);
        case MXFP4:   return make_layout(2, 1, 512, 1, 0);
        case NVFP4:   return make_layout(1, 1, 512, 1, 0);
        case Q6_K:    return make_layout(8, 1, 768, 9, 1);
        case Q4_K:    return make_layout(8, 1, 512, 8, 0);
        case Q5_K:    return make_layout(8, 2, 1280, 8, 0);
        case Q3_K:    return make_layout(8, 2, 768, 7, 0);
        case Q2_K:    return make_layout(8, 1, 256, 10, 0);
        case IQ4_XS:  return make_layout(8, 1, 512, 4, 1);
        case TQ2_0:   return make_layout(8, 1, 256, 1, 0);
        case Q2_0:    return make_layout(2, 1, 256, 1, 0);
        case Q1_0:    return make_layout(4, 2, 256, 1, 0);
        case IQ2_XXS: return make_layout(8, 1, 256, 1, 1, 256);
        case IQ1_S:   return make_layout(8, 1, 192, 1, 1, 256);
        case IQ2_S:   return make_layout(8, 1, 320, 1, 1, 256);
        case IQ3_S:   return make_layout(8, 1, 432, 1, 1, 512);
        case IQ3_XXS: return make_layout(8, 1, 384, 1, 1, 512);
        case IQ2_XS:  return make_layout(8, 1, 256, 5, 1, 256);
        case IQ1_M:   return make_layout(8, 1, 192, 4, 1, 256);
        case Q4_1:    return make_layout(1, 1, 512, 2, 0);
        case Q5_0:    return make_layout(2, 2, 1280, 2, 0);
        case Q5_1:    return make_layout(2, 2, 1280, 4, 0);
        case TQ1_0:   return make_layout(8, 8, 1536, 3, 0);
    }
    // clang-format on
    return {};
}

template <Format FMT>
constexpr PageLayout page_layout()
{
    return page_layout(FMT);
}

// Linear formats: bits of a code's low plane and of its high plane above it (Q6_K: ql | qh << 4).
constexpr std::uint32_t code_lo_bits(const Format fmt)
{
    return fmt == Q8_0 ? 8 : (fmt == Q3_K || fmt == Q2_K || fmt == TQ2_0 || fmt == Q2_0) ? 2 : fmt == Q1_0 ? 1 : 4;
}

constexpr std::uint32_t code_hi_bits(const Format fmt)
{
    return (fmt == Q5_K || fmt == Q3_K || fmt == Q5_0 || fmt == Q5_1) ? 1 : fmt == Q6_K ? 2 : 0;
}

// The formats whose group step (sfpu_gguf_decode_group) converts the fp16 d and dmin of a group once, past the page.
template <std::uint32_t FMT>
constexpr bool group_scales = FMT == Q4_K || FMT == Q5_K || FMT == Q3_K || FMT == Q2_K;

// MXFP4, NVFP4, IQ4_NL, IQ4_XS: a page's BFP_CODES codes are 4-bit block-float datums, unpacked to DEST with the
// exponent forced to BFP_CODE_EXP, followed by bfp_scale_words<FMT>() words of scale fields at scale_offset
// (ckernel_sfpu_gguf_decode_fp4.h, ckernel_sfpu_gguf_decode_iq4.h).
template <std::uint32_t FMT>
constexpr bool bfp_codes = FMT == NVFP4 || FMT == MXFP4 || FMT == IQ4_NL || FMT == IQ4_XS;

constexpr std::uint32_t BFP_CODES    = 1024;
constexpr std::uint32_t BFP_CODE_EXP = 15;

template <std::uint32_t FMT>
constexpr std::uint32_t bfp_scale_words()
{
    return page_layout(static_cast<Format>(FMT)).scale_bytes / 4;
}

} // namespace ckernel::gguf
