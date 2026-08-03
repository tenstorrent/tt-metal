// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) FPU compute kernel for binary_ng's tensor-SCALAR op.
//
// FPU tensor-scalar compute: `in1`/`pre_rhs` holds ONE writer-filled scalar tile; wait once, reuse
// index 0. Mirror of kernels/compute/eltwise_binary_scalar.cpp. Copy of eltwise_binary_no_bcast_dfb.cpp
// with the RHS handling changed for a single, reused scalar tile: the RHS PREPROCESS + wait_front(1)
// move OUT of process_tiles (the scalar tile is produced once, by the writer), the per-tile binary op
// reads RHS tile index 0 every iteration, and the RHS DFB is popped once after all chunks. The LHS
// stream and the lhs/rhs/post activation machinery are identical to the no-broadcast kernel.
//
// DFB operand naming mirrors the CB CBIndex mapping:
//   dfb::pre_lhs  (= CBIndex::c_0)   dfb::pre_rhs (= c_1)   dfb::out (= c_2)
//   dfb::post_lhs (= c_3, used only when LHS has activations)   dfb::post_rhs (= c_4, RHS activations)
// post_* default to pre_* when that operand has no activations (HAS_ACTIVATIONS(op) == 0).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_dfb.hpp"

void kernel_main() {
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

    binary_op_init_common(dfb_post_lhs_id, dfb_post_rhs_id, dfb_out_id);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
#endif

    DataflowBuffer dfb_post_lhs(dfb_post_lhs_id);
    DataflowBuffer dfb_post_rhs(dfb_post_rhs_id);
    DataflowBuffer dfb_out(dfb_out_id);

    // The scalar RHS tile is filled ONCE by writer_scalar_dfb.cpp: preprocess + wait it a single time
    // here (outside the chunk loop), then reuse tile index 0 on every binary op below.
    PREPROCESS(RHS, dfb_pre_rhs_id, dfb_post_rhs_id, dfb_out_id, 1);
    dfb_post_rhs.wait_front(1);

    // Inline helper to process n tiles (LHS only; RHS is the single reused scalar tile)
    auto process_tiles = [&](uint32_t n) {
        PREPROCESS(LHS, dfb_pre_lhs_id, dfb_post_lhs_id, dfb_out_id, n);
        dfb_post_lhs.wait_front(n);

        dfb_out.reserve_back(n);

#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST)
        binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
#endif
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            BINARY_OP(dfb_post_lhs_id, dfb_post_rhs_id, i, 0, i);  // RHS tile index fixed at 0
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
    };

    // Process full chunks
    uint32_t num_full_chunks = num_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_tiles(num_tiles_per_cycle);
    }

    // Process remainder
    uint32_t remainder = num_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_tiles(remainder);
    }

    // Pop the single reused scalar tile from the RHS DFB.
    dfb_post_rhs.pop_front(1);
}
