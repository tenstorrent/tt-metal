// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) FPU compute kernel for binary_ng's no-broadcast binary op,
// Quasar-native.
//
// Diverges from kernels_dfb/compute/eltwise_binary_no_bcast_dfb.cpp in two ways: thread c of C
// processes its own share of num_tiles, and a borrowed shard's tiles past the largest multiple of C
// come through the tail rings (TAIL_TILES). Which tiles a thread gets is the DFB's assignment.
//
// Otherwise mirrors the CircularBuffer kernels/compute/eltwise_binary_no_bcast.cpp, with the CB->DFB
// swap. Uses the same define machinery the descriptor factory builds: BINARY_OP / BINARY_OP_TYPE
// (the binary op), HAS_ACTIVATIONS / PREPROCESS / PROCESS_POST_ACTIVATIONS (lhs/rhs/post activation
// chains), PACK_RELU (fused RELU fast path). Apart from the tail rings, the reader and the writer
// absorb every layout difference.
//
// DFB operand naming mirrors the CB CBIndex mapping:
//   dfb::pre_lhs  (= CBIndex::c_0)   dfb::pre_rhs (= c_1)   dfb::out (= c_2)
//   dfb::post_lhs (= c_3, used only when LHS has activations)   dfb::post_rhs (= c_4, RHS activations)
// post_* default to pre_* when that operand has no activations (HAS_ACTIVATIONS(op) == 0).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#if HAS_MAIN_RING
// Its preprocess helper names dfb::pre_lhs, which a program with no borrowed ring does not have.
#include "eltwise_utils_dfb.hpp"
#endif

void kernel_main() {
#if HAS_MAIN_RING
    const uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    constexpr auto dfb_pre_lhs_id = static_cast<uint32_t>(dfb::pre_lhs);
    constexpr auto dfb_pre_rhs_id = static_cast<uint32_t>(dfb::pre_rhs);
    constexpr auto dfb_out_id = static_cast<uint32_t>(dfb::out);

    // post_lhs/post_rhs DFBs (c_3/c_4) exist only when that operand has an activation chain, so the
    // factory binds dfb::post_lhs / dfb::post_rhs only in that case. Guard the references with #if (not
    // a ?: ) so the no-activation build, where the accessors are unbound, still compiles. When absent,
    // the post id aliases the pre id (PREPROCESS is a no-op and the binary op reads pre directly).
#if HAS_ACTIVATIONS(LHS)
    constexpr uint32_t dfb_post_lhs_id = static_cast<uint32_t>(dfb::post_lhs);
#else
    constexpr uint32_t dfb_post_lhs_id = dfb_pre_lhs_id;
#endif
#if HAS_ACTIVATIONS(RHS)
    constexpr uint32_t dfb_post_rhs_id = static_cast<uint32_t>(dfb::post_rhs);
#else
    constexpr uint32_t dfb_post_rhs_id = dfb_pre_rhs_id;
#endif

    compute_kernel_hw_startup(dfb_post_lhs_id, dfb_post_rhs_id, dfb_out_id);
#ifdef PACK_RELU
    pack_relu_config(ReluConfig::zero());
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
#endif

    DataflowBuffer dfb_post_lhs(dfb_post_lhs_id);
    DataflowBuffer dfb_post_rhs(dfb_post_rhs_id);
    DataflowBuffer dfb_out(dfb_out_id);

    // Inline helper to process n tiles
    auto process_tiles = [&](uint32_t n) {
        PREPROCESS(LHS, dfb_pre_lhs_id, dfb_post_lhs_id, dfb_out_id, n);
        dfb_post_lhs.wait_front(n);

        PREPROCESS(RHS, dfb_pre_rhs_id, dfb_post_rhs_id, dfb_out_id, n);
        dfb_post_rhs.wait_front(n);

        dfb_out.reserve_back(n);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
#endif
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            BINARY_OP(dfb_post_lhs_id, dfb_post_rhs_id, i, i, i);
            PROCESS_POST_ACTIVATIONS(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, dfb_out_id);
        }
        tile_regs_release();

        dfb_out.push_back(n);
        dfb_post_lhs.pop_front(n);
        dfb_post_rhs.pop_front(n);
    };

    // Thread c of C consumes its own share. WHICH tiles is the DFB's business, so unlike the
    // dataflow kernels only the trip count changes here. The DFB hands consumer thread c the
    // sub-stream {c, c+C, ...}, whose length is floor(n/C) + (c < n mod C): low thread ids take
    // the remainder. A truncating divide would leave num_tiles % C entries unconsumed and hang.
    const uint32_t num_threads = get_num_threads();
    const uint32_t my_tiles = num_tiles / num_threads + (get_my_thread_id() < num_tiles % num_threads ? 1u : 0u);

    const uint32_t num_full_chunks = my_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_tiles(num_tiles_per_cycle);
    }
    // The batch need not divide a thread's share, so a remainder can be left. Dropping it would hang a NoC
    // writer on credits that never arrive, or leave borrowed output tiles uncomputed. Mirrors kernels_dfb's
    // compute kernel.
    const uint32_t remainder = my_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles(remainder);
    }
#else
    // No borrowed ring: a shard smaller than C lives entirely in the tail rings.
    compute_kernel_hw_startup(
        static_cast<uint32_t>(dfb::pre_lhs_tail),
        static_cast<uint32_t>(dfb::pre_rhs_tail),
        static_cast<uint32_t>(dfb::out_tail));
#endif
#if TAIL_TILES
#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST) or defined(PACK_RELU)
#error "the tail rings carry no activation chain or fused RELU, and the native gate admits neither"
#endif
    // A borrowed shard's tiles past the largest multiple of C: one tail entry per compute thread. A thread
    // past TAIL_TILES computes on padding, which the writer drops.
    {
        constexpr auto dfb_tail_lhs_id = static_cast<uint32_t>(dfb::pre_lhs_tail);
        constexpr auto dfb_tail_rhs_id = static_cast<uint32_t>(dfb::pre_rhs_tail);
        constexpr auto dfb_tail_out_id = static_cast<uint32_t>(dfb::out_tail);
        DataflowBuffer dfb_tail_lhs(dfb_tail_lhs_id);
        DataflowBuffer dfb_tail_rhs(dfb_tail_rhs_id);
        DataflowBuffer dfb_tail_out(dfb_tail_out_id);
        binary_tiles_init<true, BINARY_OP_TYPE>(dfb_tail_lhs_id, dfb_tail_rhs_id);
        // pack_tile keeps writing to the ring the packer was set up for, whatever id it is given, so
        // retarget it; the format is the output's already.
        pack_init(dfb_tail_out_id);
        dfb_tail_lhs.wait_front(1);
        dfb_tail_rhs.wait_front(1);
        dfb_tail_out.reserve_back(1);
        tile_regs_acquire();
        BINARY_OP(dfb_tail_lhs_id, dfb_tail_rhs_id, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb_tail_out_id);
        tile_regs_release();
        dfb_tail_out.push_back(1);
        dfb_tail_lhs.pop_front(1);
        dfb_tail_rhs.pop_front(1);
    }
#endif
}
