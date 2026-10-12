// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#ifndef ALWI
#define ALWI inline __attribute__((always_inline))
#endif

// Nearest-neighbour pixel index of a bf16 normalized coordinate with integer arithmetic only.
//
// The RISC-V data-movement cores have no FPU: the float version (bf16_to_fp32, two fp32 mul/add and
// round_half_to_even) costs ~1000 cycles per coordinate in soft float and dominated the nearest
// grid_sample kernel. This reproduces the fp32 formula of PyTorch / prepare_grid.cpp bit for bit:
//     align_corners :  p = rhe(fp32(v + 1) * (size - 1) / 2)
//     otherwise     :  p = rhe((fp32(v + 1) * size - 1) / 2)
// rounded half to even. For |v| >= 2^-17 the fp32 arithmetic is exact (v has 8 significant bits), so
// the exact integer result is the fp32 result; for smaller |v| only the rounding of (v + 1) to fp32
// matters (it becomes exactly 1.0, a .5 tie, or 1.0 +- one ulp, a tiny push off the tie).
// Returns -1 when the point is outside [0, size) or v is inf/nan.
//
// General (slow) form, also used at compile time; the hot path is nearest_index_bf16 below.
template <bool align_corners, uint32_t size>
constexpr inline __attribute__((always_inline)) int32_t nearest_index_bf16_general(uint16_t bits) {
    constexpr int32_t S = align_corners ? static_cast<int32_t>(size) - 1 : static_cast<int32_t>(size);
    constexpr int32_t Y = static_cast<int32_t>(size) - 1;  // 2 * (value for v == 0) in both modes
    constexpr int32_t K = Y >> 1;
    const uint32_t exp = (bits >> 7) & 0xFF;
    const bool neg = (bits & 0x8000) != 0;
    if (exp == 0xFF) {
        return -1;  // inf / nan
    }
    if constexpr (S == 0) {
        return 0;  // size 1 with align_corners: scale 0, every finite v maps to pixel 0
    }
    // v = sign * M * 2^E  (M = 128 + mantissa for normals; denormals: M = mantissa, E = -133)
    int32_t M = static_cast<int32_t>(bits & 0x7F);
    int32_t E = -133;
    if (exp != 0) {
        M += 128;
        E = static_cast<int32_t>(exp) - 134;
    }
    // fp32(1 + v) == 1.0 exactly (v == 0, or |v| below half an ulp of 1.0): the value is Y/2, a tie when Y is odd.
    const bool one = (M == 0) || (neg ? (E <= -33 || (E == -32 && M == 128)) : (E <= -32 || (E == -31 && M == 128)));
    if (one) {
        constexpr int32_t p = (Y & 1) ? (K + (K & 1)) : K;
        return p;
    }
    // |v| < 2^-17: fp32(1 + v) is 1.0 +- a few ulps; the product moves off Y/2 by far less than 0.5 in the direction of v.
    if (E <= -25) {
        return (Y & 1) ? (K + (neg ? 0 : 1)) : K;
    }
    if (E > 20) {
        return -1;  // |v| >= 2^20: outside any image
    }
    if (neg) {
        M = -M;
    }
    int64_t num = 0;
    int32_t D = 0;
    if (E >= 0) {
        num = (static_cast<int64_t>(M) * (static_cast<int64_t>(1) << E) + 1) * S;
    } else {
        D = -E;
        num = (static_cast<int64_t>(M) + (static_cast<int64_t>(1) << D)) * S;
    }
    if constexpr (!align_corners) {
        num -= static_cast<int64_t>(1) << D;
    }
    const int32_t sh = D + 1;
    const int64_t q64 = num >> sh;  // floor
    const int64_t r = num - q64 * (static_cast<int64_t>(1) << sh);
    const int64_t half = static_cast<int64_t>(1) << D;
    int64_t q = q64;
    if (r > half || (r == half && (q & 1))) {
        q += 1;
    }
    return (q >= 0 && q < static_cast<int64_t>(size)) ? static_cast<int32_t>(q) : -1;
}

// Byte table of the general form for |v| in [1/8, 2) (bf16 exponents 124..126, both signs): 768 B per
// size, built at compile time, lives in the kernel's (2 KB) data region. 255 = outside. Sizes <= 254 only.
template <bool align_corners, uint32_t size>
struct NearestByteTable {
    static constexpr uint32_t EXP_LO = 124;
    static constexpr uint32_t EXPS = 3;
    static constexpr uint32_t N = EXPS * 128;
    uint8_t t[2 * N];
    constexpr NearestByteTable() : t() {
        for (uint32_t sgn = 0; sgn < 2; ++sgn) {
            for (uint32_t e = 0; e < EXPS; ++e) {
                for (uint32_t m = 0; m < 128; ++m) {
                    const uint16_t bits = static_cast<uint16_t>((sgn << 15) | ((EXP_LO + e) << 7) | m);
                    const int32_t p = nearest_index_bf16_general<align_corners, size>(bits);
                    t[sgn * N + e * 128 + m] = static_cast<uint8_t>(p < 0 ? 255 : p);
                }
            }
        }
    }
};

// Hot path: |v| in [1/8, 2) by table (sizes <= 254); |v| in [2^-9, 2) with one fixed-point formula and
// no variable shift; |v| >= 2 is always outside; |v| < 2^-9 is a constant pixel next to the centre
// (sizes < 512); everything else goes through the general form. Same result as the general form for every bf16.
template <bool align_corners, uint32_t size>
ALWI int32_t nearest_index_bf16(uint16_t bits) {
    constexpr int32_t S = align_corners ? static_cast<int32_t>(size) - 1 : static_cast<int32_t>(size);
    constexpr int32_t Y = static_cast<int32_t>(size) - 1;
    constexpr int32_t K = Y >> 1;
    constexpr int32_t TINY_E = (size < 512) ? -17 : -25;  // |v| * S / 2 < 0.5 below this exponent
    const uint32_t exp = (bits >> 7) & 0xFF;
    if constexpr (S == 0) {
        return (exp == 0xFF) ? -1 : 0;  // size 1 with align_corners: scale 0, every finite v maps to pixel 0
    }
#ifndef GRID_SAMPLE_NEAREST_NO_TABLE  // the compute RISCs have a 704 B data region: arithmetic only there
    if constexpr (size <= 254) {
        static constexpr NearestByteTable<align_corners, size> table{};
        if (exp - NearestByteTable<align_corners, size>::EXP_LO < NearestByteTable<align_corners, size>::EXPS) {
            const uint32_t idx = ((bits >> 15) * NearestByteTable<align_corners, size>::N) +
                                 ((exp - NearestByteTable<align_corners, size>::EXP_LO) << 7) + (bits & 0x7F);
            const uint32_t p = table.t[idx];
            return (p == 255) ? -1 : static_cast<int32_t>(p);
        }
    }
#endif
    const int32_t E = static_cast<int32_t>(exp) - 134;
    // |v| in [2^-9, 2): v = M * 2^E is exact at scale 2^16 (V = |v| * 2^16 < 2^17), and
    // num = (1 + v) * S * 2^16 stays below 2^31 (S < 2^14 for |v| < 1, S < 2^13 for |v| in [1, 2)).
    // The [1, 2) range matters on the compute RISCs, which have no byte table: a grid that marks
    // its invalid cells with a coordinate just outside [-1, 1] would otherwise take the general
    // form for every one of them.
    if ((E >= -16 && E <= -8 && size <= 16384) || (E == -7 && size <= 8192)) {
        // v = M * 2^E, V = v * 2^16 < 2^17; num = (1 + v) * S * 2^16 < 2^31
        const uint32_t V = (static_cast<uint32_t>(bits & 0x7F) + 128u) << (16 + E);
        int32_t num = static_cast<int32_t>((bits & 0x8000) ? (65536u - V) : (65536u + V)) * S;
        if constexpr (!align_corners) {
            num -= 65536;
        }
        int32_t q = num >> 17;  // floor (num >= -65536 only when !align_corners and v -> -1: still fine)
        const int32_t r = num - (q << 17);
        if (r > 65536 || (r == 65536 && (q & 1))) {
            q += 1;
        }
        return (q >= 0 && q < static_cast<int32_t>(size)) ? q : -1;
    }
    if (E >= -6 && E < 121) {
        // 2 <= |v| < inf: outside, except that for size 2 with align_corners v == -2 lands on -0.5, which rounds to pixel 0
        if constexpr (S > 1) {
            return -1;
        } else {
            return nearest_index_bf16_general<align_corners, size>(bits);
        }
    }
    if (E <= TINY_E && E > -134) {
        const bool neg = (bits & 0x8000) != 0;
        const int32_t M = static_cast<int32_t>(bits & 0x7F) + ((E > -134) ? 128 : 0);
        const bool one = neg ? (E <= -33 || (E == -32 && M == 128)) : (E <= -32 || (E == -31 && M == 128));
        if (one) {
            return (Y & 1) ? (K + (K & 1)) : K;
        }
        return (Y & 1) ? (K + (neg ? 0 : 1)) : K;
    }
    return nearest_index_bf16_general<align_corners, size>(bits);
}

