// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused TurboQuant gather-centroids kernel.
//
// Replaces 21 TTNN ops (7×ge + 7×full_like + 7×where) with a single kernel
// that maps integer indices to centroid values entirely in DST registers.
//
// Input  CB c_0: index tiles  [BF16, values 0 … num_levels-1]
// Output CB c_2: centroid tiles [BF16, corresponding centroid values]
//
// Compile-time args:
//   0  num_tiles
//   1  num_levels         number of centroids  (2^bits, e.g. 8 for 3-bit)
//   2… centroid_bits[i]   float32 bit-patterns of each centroid value

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_binary_sfpu.h"

template <uint32_t Idx>
inline float get_centroid_float() {
    constexpr uint32_t bits = get_compile_time_arg_val(2 + Idx);
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = bits;
    return conv.f;
}

// Integer level values as uint32 bit-patterns of float32.
// Level 1 = 1.0f → 0x3F800000, level 2 = 2.0f → 0x40000000, etc.
inline uint32_t level_to_float_bits(uint32_t level) {
    float f = static_cast<float>(level);
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));
    return bits;
}

template <uint32_t Idx, uint32_t N>
struct LoadCentroids {
    static void load(float* c) {
        c[Idx] = get_centroid_float<Idx>();
        LoadCentroids<Idx + 1, N>::load(c);
    }
};
template <uint32_t N>
struct LoadCentroids<N, N> {
    static void load(float*) {}
};

// Unrolled gather loop using copy_tile + unary_ge_tile (scalar compare).
// For each level: reload indices → ge(indices, level) → mask →
//   result = result + mask × (centroid − result)
template <uint32_t Lev, uint32_t N>
struct GatherLoop {
    static void run(uint32_t cb_in, float* centroids, uint32_t* level_bits) {
        // Reload indices to DST[2], compare against integer level
        copy_tile(cb_in, 0, 2);
        unary_ge_tile_init();
        unary_ge_tile(2, level_bits[Lev]);  // DST[2] = mask

        // DST[3] = centroid[level]
        fill_tile_init();
        fill_tile(3, centroids[Lev]);

        // DST[3] = centroid − result
        sub_binary_tile_init();
        sub_binary_tile(3, 1, 3);

        // DST[3] = mask × delta
        mul_binary_tile_init();
        mul_binary_tile(2, 3, 3);

        // result += masked delta
        add_binary_tile_init();
        add_binary_tile(1, 3, 1);

        GatherLoop<Lev + 1, N>::run(cb_in, centroids, level_bits);
    }
};
template <uint32_t N>
struct GatherLoop<N, N> {
    static void run(uint32_t, float*, uint32_t*) {}
};

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_levels = get_compile_time_arg_val(1);

    float centroids[16];
    LoadCentroids<0, num_levels>::load(centroids);

    // Precompute float32 bit-patterns for integer level values (1.0, 2.0, …)
    uint32_t level_bits[16];
    for (uint32_t i = 0; i < num_levels; i++) {
        level_bits[i] = level_to_float_bits(i);
    }

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_2;

    init_sfpu(cb_in, cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // ── DST[1] = centroid[0]  (initial result) ──
        fill_tile_init();
        fill_tile(1, centroids[0]);

        // ── Conditional overwrite for each higher level ──
        GatherLoop<1, num_levels>::run(cb_in, centroids, level_bits);

        // ── PACK ──
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, cb_out);
        tile_regs_release();

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
}
