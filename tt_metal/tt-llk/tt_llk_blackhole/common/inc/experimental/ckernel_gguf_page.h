// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GGUF weight pages for the Blackhole SFPU decoders (sfpu/experimental/ckernel_sfpu_gguf_decode.h): the format ids, the
// layout of a page in L1, and the two steps a RISC runs on a page before it is unpacked to DEST:
//   - prepare_scales: fp32 copies of the page's fp16 scale fields, so the SFPU needs no fp16 conversion for them;
//   - lookup_grid:    the codebook formats' grid indices to level codes, through the format's level table.
// Plain C++ over L1 with no Tensix instructions, so any RISC can run them (data movement core or TRISC).
//
// A page is the decoder's view of one K tile (two for the pair formats, eight for TQ1_0) of a 32-column N tile. It is a
// 32-bit tile of 16, 32 or 64 rows of 16 words, unpacked to DEST as is:
//   bytes [code_offset, code_offset + code_bytes): the code words, regrouped from the GGUF blocks so that SFPU lane
//                     (r, c) finds its outputs' codes in one place; codebook formats keep their native index bytes here
//                     and lookup_grid writes the level codes in front of them (from byte 0);
//   from scale_offset: the scale fields of the K tile's group, regrouped the same way: raw_rows native rows, then
//                     fp16_rows rows of fp16 fields, then the fp32 rows prepare_scales adds.
// The bits are the GGUF file's, moved: a page's code and scale bytes are its weights' GGUF bytes. Each format's rows
// are described in the ckernel_sfpu_gguf_decode*.h headers.

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

struct PageLayout
{
    std::uint32_t group_tiles;   // K tiles sharing one set of scale fields (the decoder's tile index runs over them)
    std::uint32_t page_tiles;    // K tiles per page
    std::uint32_t code_bytes;    // native code bytes of a page
    std::uint32_t scale_rows;    // native 64-byte rows of scale fields per group
    std::uint32_t raw_rows;      // scale rows without fp32 copies
    std::uint32_t fp16_rows;     // scale rows of fp16 fields after them, each copied as fp32 by prepare_scales
    std::uint32_t code_offset;   // bytes before the code bytes (the codebook formats' level codes)
    std::uint32_t table_entries; // codebook formats: level table entries
    std::uint32_t scale_offset;  // bytes before the scale fields: the code area rounded up to 256 bytes
    std::uint32_t scale_bytes;   // scale fields with their fp32 copies
    std::uint32_t rows;          // 64-byte rows of the page: 16, 32 or 64
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
    const std::uint32_t code_offset   = 0,
    const std::uint32_t table_entries = 0)
{
    const std::uint32_t raw_rows     = scale_rows - fp16_rows;
    const std::uint32_t scale_offset = round_up(code_offset + code_bytes, 256);
    const std::uint32_t scale_bytes  = fp16_rows ? (round_up(scale_rows, 4) + 8 * fp16_rows) * 64 : scale_rows * 64;
    std::uint32_t rows               = 16;
    while (rows * 64 < scale_offset + scale_bytes)
    {
        rows *= 2;
    }
    return {group_tiles, page_tiles, code_bytes, scale_rows, raw_rows, fp16_rows, code_offset, table_entries, scale_offset, scale_bytes, rows};
}

template <Format FMT>
constexpr PageLayout page_layout()
{
    // clang-format off
    switch (FMT)
    {
        case Q8_0:    return make_layout(1, 1, 1024, 1, 0);
        case Q4_0:    return make_layout(1, 1, 512, 1, 0);
        case IQ4_NL:  return make_layout(1, 1, 512, 1, 0);
        case MXFP4:   return make_layout(2, 1, 512, 1, 0);
        case NVFP4:   return make_layout(1, 1, 512, 1, 0);
        case Q6_K:    return make_layout(8, 1, 768, 9, 1);
        case Q4_K:    return make_layout(8, 1, 512, 8, 2);
        case Q5_K:    return make_layout(8, 2, 1280, 8, 2);
        case Q3_K:    return make_layout(8, 2, 768, 7, 1);
        case Q2_K:    return make_layout(8, 1, 256, 10, 2);
        case IQ4_XS:  return make_layout(8, 1, 512, 4, 1);
        case TQ2_0:   return make_layout(8, 1, 256, 1, 0);
        case Q2_0:    return make_layout(2, 1, 256, 1, 0);
        case Q1_0:    return make_layout(4, 2, 256, 1, 0);
        case IQ2_XXS: return make_layout(8, 1, 256, 1, 1, 256, 256);
        case IQ1_S:   return make_layout(8, 1, 192, 1, 1, 256, 2048);
        case IQ2_S:   return make_layout(8, 1, 320, 1, 1, 256, 1024);
        case IQ3_S:   return make_layout(8, 1, 432, 1, 1, 512, 512);
        case IQ3_XXS: return make_layout(8, 1, 384, 1, 1, 512, 256);
        case IQ2_XS:  return make_layout(8, 1, 256, 5, 1, 256, 512);
        case IQ1_M:   return make_layout(8, 1, 192, 4, 1, 256, 2048);
        case Q4_1:    return make_layout(1, 1, 512, 2, 0);
        case Q5_0:    return make_layout(2, 2, 1280, 2, 0);
        case Q5_1:    return make_layout(2, 2, 1280, 4, 0);
        case TQ1_0:   return make_layout(8, 8, 1536, 3, 0);
    }
    // clang-format on
    return {};
}

template <Format FMT>
constexpr bool is_codebook()
{
    return page_layout<FMT>().table_entries != 0;
}

// fp16 bits to fp32 bits, exact for every value (subnormals normalized, inf and NaN kept).
inline std::uint32_t fp16_to_fp32(const std::uint32_t h)
{
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t mag        = h & 0x7FFFu;
    if (mag - 0x0400u < 0x7800u)
    {
        return sign | ((mag << 13) + 0x38000000u);
    }
    if (mag >= 0x7C00u)
    {
        return sign | 0x7F800000u | ((mag & 0x3FFu) << 13);
    }
    if (mag == 0)
    {
        return sign;
    }
    std::uint32_t exp = 113;
    while ((mag & 0x400u) == 0)
    {
        mag <<= 1;
        exp--;
    }
    return sign | (exp << 23) | ((mag & 0x3FFu) << 13);
}

// The fp32 copies of a page's fp16 scale rows: fp16 row raw_rows + f, half b (column 16p + 2c + b of word 2c + p) goes
// to rows R + 8f + 4b .. R + 8f + 4b + 3, R = raw_rows + fp16_rows rounded up to 4, the same 16 words in each of the
// four rows, so that one SFPU load gives every lane its column's value. `scales` points at the page's scale fields.
template <Format FMT>
inline void prepare_scales(volatile std::uint32_t* scales)
{
    constexpr PageLayout L    = page_layout<FMT>();
    constexpr std::uint32_t R = round_up(L.scale_rows, 4);
#pragma GCC unroll 1
    for (std::uint32_t f = 0; f < L.fp16_rows; f++)
    {
        std::uint32_t lo[16], hi[16];
#pragma GCC unroll 16
        for (std::uint32_t w = 0; w < 16; w++)
        {
            const std::uint32_t x = scales[(L.raw_rows + f) * 16 + w];
            lo[w]                 = fp16_to_fp32(x & 0xFFFFu);
            hi[w]                 = fp16_to_fp32(x >> 16);
        }
#pragma GCC unroll 4
        for (std::uint32_t k = 0; k < 4; k++)
        {
#pragma GCC unroll 16
            for (std::uint32_t w = 0; w < 16; w++)
            {
                scales[(R + 8 * f + k) * 16 + w]     = lo[w];
                scales[(R + 8 * f + 4 + k) * 16 + w] = hi[w];
            }
        }
    }
}

// Codebook formats: the level codes of a page's grid indices, two 16-bit codes per output word, from the format's
// level table (entry -> the level index of each of its elements, element i at bits w i; built from ggml's grid by the
// host). `indices` points at the page's index words, `levels` at the page's first byte.
//   IQ2_XXS, IQ3_XXS: 32 / 64 index words of four 8-bit indices.
//   IQ1_S, IQ1_M, IQ2_S: 32 index words of four low index bytes; 32-bit word j after them holds the high bits of
//     index words 2j (bytes 0..3 at bits 8, 8 + w, 24, 24 + w) and 2j + 1 (at 0, w, 16, 16 + w), w = 3 (IQ1_*) or 2
//     (IQ2_S); its other bits are the decoder's.
//   IQ2_XS: 32 index words of low index bytes; the ninth bit of index word m = 8l + c's byte k is bit 8k + 7 of word
//     16 (l / 2) + 2c + l % 2 of the 32 sign words before them.
//   IQ3_S: 64 index words of low index bytes; word k < 8 of the 12 after them holds the high bits of index words
//     8k + i, bytes 0..3 at bits i, 8 + i, 16 + i, 24 + i; byte 8 (m / 32) + m % 8 of words 8..11 is ORed into bits
//     12..15 of index word m's two codes when (m / 8) % 4 == 0 (low nibble into the first): the columns' scales.
// part / parts: this call looks up part `part` of `parts` equal shares of the index words.
template <Format FMT, typename Out>
inline __attribute__((always_inline)) void lookup_grid(
    const std::uint32_t* indices, const std::uint16_t* table, Out* levels, const std::uint32_t part = 0, const std::uint32_t parts = 1)
{
    static_assert(is_codebook<FMT>(), "lookup_grid: not a codebook format");
    constexpr std::uint32_t words = FMT == IQ3_S || FMT == IQ3_XXS ? 64 : 32;
    const std::uint32_t m_begin   = part * words / parts;
    const std::uint32_t m_end     = (part + 1) * words / parts;
    if constexpr (FMT == IQ3_S || FMT == IQ3_XXS)
    {
        constexpr bool high     = FMT == IQ3_S;
        const std::uint32_t* hi = indices + 64;
        const std::uint32_t* sc = hi + 8;
#pragma GCC unroll 1
        for (std::uint32_t m0 = m_begin; m0 < m_end; m0 += 8)
        {
            // The loads of a batch first: independent L1 loads overlap, dependent ones do not.
            std::uint32_t x[8];
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++)
            {
                x[b] = indices[m0 + b];
            }
            const std::uint32_t hw = high ? hi[m0 >> 3] : 0;
            std::uint32_t s0 = 0, s1 = 0;
            if (high && ((m0 >> 3) & 3) == 0)
            {
                s0 = sc[2 * (m0 >> 5)];
                s1 = sc[2 * (m0 >> 5) + 1];
            }
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++)
            {
                constexpr std::uint32_t lo = 0x00ff00ffu, hi_mask = 0x01000100u;
                const std::uint32_t ac = (x[b] & lo) | ((hw << (8 - b)) & hi_mask);
                const std::uint32_t bd = ((x[b] >> 8) & lo) | ((hw >> b) & hi_mask);
                std::uint32_t g0       = table[ac & 0xffff];
                std::uint32_t g1       = table[bd & 0xffff];
                const std::uint32_t g2 = table[ac >> 16];
                const std::uint32_t g3 = table[bd >> 16];
                if (high && ((m0 >> 3) & 3) == 0)
                {
                    const std::uint32_t s = ((b < 4 ? s0 : s1) >> (8 * (b & 3))) & 0xff;
                    g0 |= (s & 15) << 12;
                    g1 |= (s >> 4) << 12;
                }
                levels[2 * (m0 + b)]     = g0 | (g1 << 16);
                levels[2 * (m0 + b) + 1] = g2 | (g3 << 16);
            }
        }
    }
    else if constexpr (FMT == IQ1_S || FMT == IQ1_M || FMT == IQ2_S || FMT == IQ2_XS)
    {
        constexpr std::uint32_t w_bits  = FMT == IQ2_S ? 2 : 3;
        constexpr std::uint32_t lo      = 0x00ff00ffu;
        constexpr std::uint32_t hi_mask = FMT == IQ2_S ? 0x03000300u : FMT == IQ2_XS ? 0x01000100u : 0x07000700u;
#pragma GCC unroll 1
        for (std::uint32_t m0 = m_begin; m0 < m_end; m0 += 8)
        {
            std::uint32_t x[8], hh[8];
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++)
            {
                x[b] = indices[m0 + b];
            }
            if constexpr (FMT == IQ2_XS)
            {
                const std::uint32_t* s = indices - 32 + 16 * (m0 >> 4) + ((m0 >> 3) & 1);
#pragma GCC unroll 8
                for (std::uint32_t b = 0; b < 8; b++)
                {
                    hh[b] = s[2 * b];
                }
            }
            else
            {
#pragma GCC unroll 4
                for (std::uint32_t b = 0; b < 4; b++)
                {
                    hh[b] = indices[32 + m0 / 2 + b];
                }
            }
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++)
            {
                const std::uint32_t h    = hh[FMT == IQ2_XS ? b : b / 2];
                const std::uint32_t h_ac = (FMT == IQ2_XS ? h << 1 : b % 2 ? h << 8 : h) & hi_mask;
                const std::uint32_t h_bd = (FMT == IQ2_XS ? h >> 7 : b % 2 ? h << (8 - w_bits) : h >> w_bits) & hi_mask;
                const std::uint32_t ac   = (x[b] & lo) | h_ac;
                const std::uint32_t bd   = ((x[b] >> 8) & lo) | h_bd;
                const std::uint32_t g0   = table[ac & 0xffff];
                const std::uint32_t g1   = table[bd & 0xffff];
                const std::uint32_t g2   = table[ac >> 16];
                const std::uint32_t g3   = table[bd >> 16];
                levels[2 * (m0 + b)]     = g0 | (g1 << 16);
                levels[2 * (m0 + b) + 1] = g2 | (g3 << 16);
            }
        }
    }
    else
    {
#pragma GCC unroll 1
        for (std::uint32_t m0 = m_begin; m0 < m_end; m0 += 8)
        {
            std::uint32_t x[8];
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++)
            {
                x[b] = indices[m0 + b];
            }
#pragma GCC unroll 8
            for (std::uint32_t b = 0; b < 8; b++)
            {
                const std::uint32_t g0   = table[x[b] & 0xff];
                const std::uint32_t g1   = table[(x[b] >> 8) & 0xff];
                const std::uint32_t g2   = table[(x[b] >> 16) & 0xff];
                const std::uint32_t g3   = table[x[b] >> 24];
                levels[2 * (m0 + b)]     = g0 | (g1 << 16);
                levels[2 * (m0 + b) + 1] = g2 | (g3 << 16);
            }
        }
    }
}

// Both steps for one page at L1 byte address `page` (table: the level table, codebook formats only).
template <Format FMT>
inline void prepare_page(const std::uint32_t page, const std::uint16_t* table = nullptr)
{
    constexpr PageLayout L = page_layout<FMT>();
    if constexpr (L.fp16_rows > 0)
    {
        prepare_scales<FMT>(reinterpret_cast<volatile std::uint32_t*>(page + L.scale_offset));
    }
    if constexpr (is_codebook<FMT>())
    {
        // The index words follow 128 bytes of sign words, except IQ1_S / IQ1_M's, which come first.
        constexpr std::uint32_t index_offset = L.code_offset + (FMT == IQ1_S || FMT == IQ1_M ? 0 : 128);
        lookup_grid<FMT>(reinterpret_cast<const std::uint32_t*>(page + index_offset), table, reinterpret_cast<volatile std::uint32_t*>(page));
    }
}

} // namespace ckernel::gguf
