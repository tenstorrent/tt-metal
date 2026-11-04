// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The router's rows.
// A fragment the including kernel inlines: it relies on what that kernel declared before it (included once a
// kernel).
#pragma once
#ifndef RT_FOLD
#define RT_FOLD 1
#endif
// (RT_SPLIT: a row tile's experts over two router cores - RT_HALF 0 / 1, its expert tiles from RT_HB, ET of them;
// RT_ETF every expert tile: the bias, the keys' id bits and the joint rows are the whole row tile's)
#ifndef RT_ETF
#define RT_ETF ET
#endif
#ifndef RT_HB
#define RT_HB 0
#endif
#ifndef RT_HALF
#define RT_HALF 0
#endif
namespace rows_ {
// (r, c) of a tile: face (r / 16) * 2 + c / 16, then row-major within the face
FORCE_INLINE uint32_t fo(uint32_t r, uint32_t c) { return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15); }
// the index of x's top bit (x > 0): five steps (no Zbb on the RISCs: __builtin_clz is a library loop)
FORCE_INLINE int32_t msb32(uint32_t x) {
    int32_t m = 0;
    if (x >> 16) {
        x >>= 16, m += 16;
    }
    if (x >> 8) {
        x >>= 8, m += 8;
    }
    if (x >> 4) {
        x >>= 4, m += 4;
    }
    if (x >> 2) {
        x >>= 2, m += 2;
    }
    if (x >> 1) {
        m += 1;
    }
    return m;
}
// x (> 0, < 2^31): a value scaled by 2^f -> bf16 bits, round to nearest even
FORCE_INLINE uint32_t bf16_of(uint32_t x, int32_t f) {
    if (!x) {
        return 0;
    }
    const int32_t msb = msb32(x);
    uint32_t t;
    int32_t e = msb;
    if (msb > 7) {  // (keep 8 bits: the hidden one + 7)
        const int32_t drop = msb - 7;
        const uint32_t half = 1u << (drop - 1), rem = x & ((1u << drop) - 1);
        t = x >> drop;
        if (rem > half || (rem == half && (t & 1))) {
            ++t;
        }
        if (t >> 8) {
            t >>= 1, ++e;
        }
    } else {
        t = x << (7 - msb);
    }
    return ((uint32_t)(e - f + 127) << 7) | (t & 0x7F);
}
// the expert id bits of a top-K key (the low mantissa bits of its fp32 score)
constexpr uint32_t em_of(uint32_t et) { return et * 32 <= 256 ? 0xFF : et * 32 <= 512 ? 0x1FF : 0x3FF; }
// the fp32 score of expert eid (one of mine: expert tiles RT_HB .., ETL of them) for token column c - the fin's score
// tiles at sv, transposed like the logits: (eid % 32, c) of tile eid / 32
template <uint32_t ETL>
FORCE_INLINE uint32_t score_of(uint32_t sv, uint32_t eid, uint32_t c) {
    return reinterpret_cast<const uint32_t*>(sv + ((eid >> 5) - RT_HB) * 4096)[fo(eid & 31, c)];
}
// (split half 1) its picks' scores, for its partner's rows: the score of the key at (q, c) of the best page at (q, c)
// of the page after its own PART page (sent with the best page)
template <uint32_t K, uint32_t ET, uint32_t ETL>
inline void picked(uint32_t bv, uint32_t sv, uint32_t c0, uint32_t c1) {
#ifdef RT_PART
    constexpr uint32_t EM = em_of(ET);
    for (uint32_t c = c0; c < c1; ++c) {
        for (uint32_t q = 0; q < K; ++q) {
            const uint32_t key = reinterpret_cast<const uint32_t*>(bv)[fo(q, c)];
            reinterpret_cast<uint32_t*>(RT_PART + 4096)[fo(q, c)] = score_of<ETL>(sv, key & EM, c);
        }
    }
#else
    (void)bv, (void)sv, (void)c0, (void)c1;
#endif
}
template <uint32_t K, uint32_t ET, uint32_t ETL>
inline void make(uint32_t bv, uint32_t sv, uint32_t o_l1, uint32_t scale_bits, uint32_t c0, uint32_t c1) {
    // weights = score / their sum x scale in integers (the RISCs have no FPU, no 64-bit divide): the K scores in one
    // block exponent (the largest's top bit at 25: a sum of up to 32 below 2^31), the sum normalized to bit 30, its
    // reciprocal by one 32-bit divide (16 bits: a bf16 weight needs 8 + rounding), x the scale's mantissa, then one
    // 32 x 32 -> 64 product a weight
    const uint32_t sc = (scale_bits & 0x7FFFFF) | 0x800000;  // (scale = sc x 2^se)
    const int32_t se = (int32_t)((scale_bits >> 23) & 0xFF) - 150;
    constexpr uint32_t EM = em_of(ET);
    for (uint32_t c = c0; c < c1; ++c) {
        uint32_t ids[K], w[K], sum = 0;  // (w: the scores' fp32 bits, then their block-exponent integers)
        int32_t emax = 0;
        for (uint32_t q = 0; q < K; ++q) {
            const uint32_t key =
                reinterpret_cast<const uint32_t*>(bv)[fo(q, c)];  // (the top-K's key: id in the low bits)
            const uint32_t eid = key & EM;
            ids[q] = eid;
            if ((eid >> 5) - RT_HB < ETL) {
                w[q] = score_of<ETL>(sv, eid, c);
            } else {  // (split: the partner half's pick - its key and score at the same place of its two PART pages)
                w[q] = 0;
#ifdef RT_PART
                for (uint32_t r = 0; r < K; ++r) {
                    if (reinterpret_cast<const uint32_t*>(RT_PART)[fo(r, c)] == key) {
                        w[q] = reinterpret_cast<const uint32_t*>(RT_PART + 4096)[fo(r, c)];
                        break;
                    }
                }
#endif
            }
            if (w[q] >> 31) {
                w[q] = 0;  // (sigmoid and sqrt(softplus) scores are >= 0)
            }
            const int32_t e = (w[q] >> 23) & 0xFF;
            if (e > emax) {
                emax = e;
            }
        }
        for (uint32_t q = 0; q < K; ++q) {
            const int32_t e = (w[q] >> 23) & 0xFF, sh = e - emax + 2;  // (mantissa top bit 23 -> 25 for the largest)
            const uint32_t m = (w[q] & 0x7FFFFF) | 0x800000;
            w[q] = e == 0 ? 0 : sh >= 0 ? (m << sh) : (sh > -31 ? (m >> -sh) : 0);
            sum += w[q];
        }
        uint16_t* o = reinterpret_cast<uint16_t*>(o_l1 + c * 4 * K);
        const int32_t sh = sum ? 30 - msb32(sum) : 0;
        const uint32_t sn = sh >= 0 ? sum << sh : sum >> -sh;
        const uint32_t r1 = sn ? 0xFFFFFFFFu / (sn >> 14) : 0;     // (2^32 / (sn / 2^14): 2^62 / sn = r1 x 2^16)
        const uint32_t rs = (uint32_t)(((uint64_t)r1 * sc) >> 8);  // (weight = wn x rs / 2^32 x 2^(se - 6))
        for (uint32_t q = 0; q < K; ++q) {
            o[q] = static_cast<uint16_t>(ids[q] / RT_FOLD);  // (candidate -> expert: RouterSpec.fold)
            const uint32_t wn = sh >= 0 ? (uint32_t)w[q] << sh : (uint32_t)w[q] >> -sh;
            o[K + q] = static_cast<uint16_t>(bf16_of((uint32_t)(((uint64_t)wn * rs) >> 32), 6 - se));
        }
    }
}
}  // namespace rows_
