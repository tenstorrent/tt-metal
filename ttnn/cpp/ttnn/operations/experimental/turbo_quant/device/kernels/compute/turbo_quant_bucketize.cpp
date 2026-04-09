// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Fused TurboQuant bucketize kernel.
//
// Replaces 13 TTNN ops (7×ge + 6×add) with a single kernel that processes
// each tile entirely in DST registers—no intermediate DRAM round-trips.
//
// Input  CB c_0: y_hat tiles  [BF16, normalised rotated values in ~[-1, 1]]
// Output CB c_2: index tiles  [BF16, values 0 … 2^bits - 1]
//
// Compile-time args:
//   0  num_tiles            tiles to process on this core
//   1  num_boundaries       number of inner boundaries  (2^bits - 1)
//   2… boundary_bits[i]     float32 bit-patterns of each boundary value

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_binary_sfpu.h"

// Template-unrolled boundary loading from compile-time args.
template <uint32_t Idx>
inline uint32_t get_boundary_bits() {
    return get_compile_time_arg_val(2 + Idx);
}

template <uint32_t Idx, uint32_t N>
struct LoadBoundaryBits {
    static void load(uint32_t* bits) {
        bits[Idx] = get_boundary_bits<Idx>();
        LoadBoundaryBits<Idx + 1, N>::load(bits);
    }
};
template <uint32_t N>
struct LoadBoundaryBits<N, N> {
    static void load(uint32_t*) {}
};

// Template-unrolled bucketize loop.
// For each boundary: reload input → unary_ge (scalar compare) → add to accumulator.
// Uses copy_tile to reload input from CB (still in L1, not re-fetched from DRAM).
template <uint32_t Idx, uint32_t N>
struct BucketizeLoop {
    static void run(uint32_t cb_in, uint32_t* boundary_bits) {
        // Reload input to DST[2] and compare against boundary scalar
        copy_tile(cb_in, 0, 2);
        unary_ge_tile_init();
        unary_ge_tile(2, boundary_bits[Idx]);  // DST[2] = (input >= boundary) ? 1.0 : 0.0

        // DST[1] += DST[2]
        add_binary_tile_init();
        add_binary_tile(1, 2, 1);

        BucketizeLoop<Idx + 1, N>::run(cb_in, boundary_bits);
    }
};
template <uint32_t N>
struct BucketizeLoop<N, N> {
    static void run(uint32_t, uint32_t*) {}
};

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_boundaries = get_compile_time_arg_val(1);

    uint32_t boundary_bits[15];
    LoadBoundaryBits<0, num_boundaries>::load(boundary_bits);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_out = tt::CBIndex::c_2;

    init_sfpu(cb_in, cb_out);

    for (uint32_t t = 0; t < num_tiles; t++) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        tile_regs_acquire();

        // ── DST[1] = 0.0  (accumulator for bucket index) ──
        fill_tile_init();
        fill_tile(1, 0.0f);

        // ── For each boundary: reload input → scalar ge → accumulate ──
        BucketizeLoop<0, num_boundaries>::run(cb_in, boundary_bits);

        // ── PACK result ──
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(1, cb_out);
        tile_regs_release();

        cb_pop_front(cb_in, 1);
        cb_push_back(cb_out, 1);
    }
}
