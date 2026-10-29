// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) SFPU compute kernel for binary_ng, Quasar-native: the SFPU op on each output
// tile, with each operand in its mode (A_MODE, B_MODE; kernels_qsr/common/operand_walk.hpp).
//
// Diverges from kernels_dfb/compute/eltwise_binary_sfpu_no_bcast_dfb.cpp as the native FPU kernel diverges
// from its own: thread c of C processes its own share of num_tiles, and a borrowed shard's tiles past the
// largest multiple of C come through the tail rings (TAIL_TILES). Which tiles a thread gets is the DFB's
// assignment. Each tile takes two DST slots, the operands' copies, so a batch holds at most 4 tiles in a 16-bit DST
// and 2 in a 32-bit DST. A ROW operand goes from its row tile straight into its DST slot with unary_bcast<ROW>, in
// place of the copy. A held operand (COL, SCALAR) is expanded once per page change into its held ring, and a VALUE
// operand is one ring entry; each is copied from entry 0 for every tile. The native gate runs bf16 multiply and divide
// on it, and fp32 add, subtract, multiply and divide.

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
// The SFPU binary op headers that are ported to Quasar, as in the kernels_dfb kernel. The ops the native gate admits
// use eltwise_binary_sfpu.h.
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/add_int_sfpu.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/binary_comp.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "ttnn/operations/experimental/quasar/binary_ng/device/kernels_qsr/common/operand_walk.hpp"
#if A_ROW || B_ROW
#include "api/compute/bcast.h"
#endif
#if HAS_MAIN_RING
// Its format reference names dfb::pre_lhs, which a program with no borrowed ring does not have.
#include "eltwise_utils_dfb.hpp"

// The DataflowBuffer arguments are kernel_main's objects: an object built here would drain at every return.
template <typename DFBPreLhs, typename DFBPostLhs, typename DFBPreRhs, typename DFBPostRhs, typename DFBOut>
FORCE_INLINE void process_sfpu_tiles(
    uint32_t n,
    DFBPreLhs& dfb_pre_lhs,
    DFBPostLhs& dfb_post_lhs,
    DFBPreRhs& dfb_pre_rhs,
    DFBPostRhs& dfb_post_rhs,
    DFBOut& dfb_out ISCLOSE_RT_ARG_PARAMS) {
    const uint32_t dfb_post_lhs_id = dfb_post_lhs.get_id();
    const uint32_t dfb_post_rhs_id = dfb_post_rhs.get_id();
    const uint32_t dfb_out_id = dfb_out.get_id();
    // A held or VALUE operand is ready at entry 0 of its ring before the batch: kernel_main holds it.
    constexpr bool a_one_entry = A_HELD || A_VALUE;
    constexpr bool b_one_entry = B_HELD || B_VALUE;
#if !A_HELD && !A_VALUE
    PREPROCESS(LHS, dfb_pre_lhs, dfb_post_lhs, dfb_out_id, n);
    dfb_post_lhs.wait_front(n);
#endif

#if !B_HELD && !B_VALUE
    PREPROCESS(RHS, dfb_pre_rhs, dfb_post_rhs, dfb_out_id, n);
    dfb_post_rhs.wait_front(n);
#endif

    dfb_out.reserve_back(n);

#if (HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT;
#endif

    tile_regs_acquire();
    // Startup and preprocessing preserve the physical-LHS SrcA format. An activation chain on a ROW operand ran on its
    // row tiles, and the broadcast copies their row 0, so the bits are those of the activation after the broadcast.
#if A_ROW
    unary_bcast_init<BroadcastType::ROW>(dfb_post_lhs_id);
    for (uint32_t i = 0; i < n; ++i) {
        unary_bcast<BroadcastType::ROW>(dfb_post_lhs_id, i, i * 2);
    }
#else
    copy_init(dfb_post_lhs_id);
    for (uint32_t i = 0; i < n; ++i) {
        copy_tile(dfb_post_lhs_id, a_one_entry ? 0u : i, i * 2);
    }
#endif
    reconfig_data_format_srca(dfb_post_lhs_id, dfb_post_rhs_id);
#if B_ROW
    unary_bcast_init<BroadcastType::ROW>(dfb_post_rhs_id);
#else
    copy_init(dfb_post_rhs_id);
#endif
    for (uint32_t i = 0; i < n; ++i) {
#if B_ROW
        unary_bcast<BroadcastType::ROW>(dfb_post_rhs_id, i, i * 2 + 1);
#else
        copy_tile(dfb_post_rhs_id, b_one_entry ? 0u : i, i * 2 + 1);
#endif
#if HAS_ACTIVATIONS(POST)
        // Each tile's post activation runs its own SFPU init, which can overwrite what the binary op set up (a
        // TANH init rewrites the constant that divide's reciprocal reads), so the binary init runs before every tile.
        BINARY_SFPU_INIT;
#endif
#if ISCLOSE_OP
        BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2, rtol_bits, atol_bits);
#else
        BINARY_SFPU_OP(i * 2, i * 2 + 1, i * 2);
#endif
        PROCESS_POST_ACTIVATIONS(i * 2);
    }
    reconfig_data_format_srca(dfb_post_rhs_id, dfb_post_lhs_id);
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < n; ++i) {
        pack_tile(i * 2, dfb_out_id);
    }
    tile_regs_release();

    dfb_out.push_back(n);
#if !A_HELD && !A_VALUE
    dfb_post_lhs.pop_front(n);
#endif
#if !B_HELD && !B_VALUE
    dfb_post_rhs.pop_front(n);
#endif
}
#endif

void kernel_main() {
#ifdef ISCLOSE_OP
    const uint32_t rtol_bits = get_arg(args::rtol_bits);
    const uint32_t atol_bits = get_arg(args::atol_bits);
#endif
#if HAS_MAIN_RING
    const uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    constexpr auto dfb_pre_lhs_id = static_cast<uint32_t>(dfb::pre_lhs);
    constexpr auto dfb_pre_rhs_id = static_cast<uint32_t>(dfb::pre_rhs);
    constexpr auto dfb_out_id = static_cast<uint32_t>(dfb::out);

    // post_lhs/post_rhs DFBs (c_3/c_4) exist only when that operand has an activation chain, so the
    // factory binds dfb::post_lhs / dfb::post_rhs only in that case. Guard the references with #if (not
    // a ?: ) so the no-activation build, where the accessors are unbound, still compiles. When absent,
    // the post id aliases the pre id (PREPROCESS is a no-op and the binary op reads pre directly). A held operand runs
    // its chain while it expands, and has no post ring.
#if HAS_ACTIVATIONS(LHS) && !A_HELD
    constexpr uint32_t dfb_post_lhs_id = static_cast<uint32_t>(dfb::post_lhs);
#else
    constexpr uint32_t dfb_post_lhs_id = dfb_pre_lhs_id;
#endif
#if HAS_ACTIVATIONS(RHS) && !B_HELD
    constexpr uint32_t dfb_post_rhs_id = static_cast<uint32_t>(dfb::post_rhs);
#else
    constexpr uint32_t dfb_post_rhs_id = dfb_pre_rhs_id;
#endif

    compute_kernel_hw_startup(dfb_post_lhs_id, dfb_out_id);
    copy_init(dfb_post_lhs_id);
#ifdef PACK_RELU
    pack_relu_config(ReluConfig::zero());
#endif

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS)) and not(HAS_ACTIVATIONS(POST))
    BINARY_SFPU_INIT
#endif

    // Each object drains once, at kernel exit. An activation chain reads its operand from the pre ring. Without
    // one, PREPROCESS compiles to nothing, and the pre object is the post one.
    DataflowBuffer dfb_post_lhs(dfb_post_lhs_id);
    DataflowBuffer dfb_post_rhs(dfb_post_rhs_id);
    DataflowBuffer dfb_out(dfb_out_id);
#if HAS_ACTIVATIONS(LHS) && !A_HELD
    DataflowBuffer dfb_pre_lhs(dfb_pre_lhs_id);
#else
    auto& dfb_pre_lhs = dfb_post_lhs;
#endif
#if HAS_ACTIVATIONS(RHS) && !B_HELD
    DataflowBuffer dfb_pre_rhs(dfb_pre_rhs_id);
#else
    auto& dfb_pre_rhs = dfb_post_rhs;
#endif
    // The ring each operand is copied from: a held operand's held ring, else its post ring.
#if A_HELD
    DataflowBuffer dfb_held_lhs(static_cast<uint32_t>(dfb::held_lhs));
    bool holding_lhs = false;
    auto& dfb_bin_lhs = dfb_held_lhs;
#else
    auto& dfb_bin_lhs = dfb_post_lhs;
#endif
#if B_HELD
    DataflowBuffer dfb_held_rhs(static_cast<uint32_t>(dfb::held_rhs));
    bool holding_rhs = false;
    auto& dfb_bin_rhs = dfb_held_rhs;
#else
    auto& dfb_bin_rhs = dfb_post_rhs;
#endif

    // A VALUE operand is one entry for the whole op: wait for it once, and pop it at the end. Its activation chain runs
    // once, into its post ring. Every compute thread takes its entry, also with no tiles.
#if A_VALUE
    PREPROCESS(LHS, dfb_pre_lhs, dfb_post_lhs, dfb_out_id, 1);
    dfb_post_lhs.wait_front(1);
#endif
#if B_VALUE
    PREPROCESS(RHS, dfb_pre_rhs, dfb_post_rhs, dfb_out_id, 1);
    dfb_post_rhs.wait_front(1);
#endif

    // Thread c of C consumes its own share, as in the native FPU kernel: the DFB hands consumer thread c the
    // sub-stream {c, c+C, ...}, whose length is floor(n/C) + (c < n mod C).
    const uint32_t num_threads = get_num_threads();
    const uint32_t my_tiles = num_tiles / num_threads + (get_my_thread_id() < num_tiles % num_threads ? 1u : 0u);

#if A_HELD || B_HELD
    // As in the native FPU kernel: a cursor follows this thread's tiles, and a batch ends at the next multiple of the
    // batch size, as in the same-shape loop below, or earlier where a held page changes.
    ComputeWalk walk = compute_walk(get_my_thread_id(), num_threads);
    constexpr uint32_t held_mask = (A_HELD ? 1u : 0u) | (B_HELD ? 2u : 0u);
    binary_ng_qsr::for_each_batch(
        walk.cursor, walk.step, my_tiles, num_tiles_per_cycle, held_mask, [&](uint32_t n, uint32_t batch_changed) {
#if A_HELD
            hold_dfb<A_MODE == QSR_MODE_COL ? BroadcastType::COL : BroadcastType::SCALAR, (HAS_ACTIVATIONS(LHS) != 0)>(
                dfb_post_lhs,
                dfb_held_lhs,
                dfb_out_id,
                n,
                (batch_changed & 1u) != 0,
                holding_lhs,
                [&]([[maybe_unused]] uint32_t i) { PROCESS_ACTIVATIONS(LHS, i); });
#endif
#if B_HELD
            hold_dfb<B_MODE == QSR_MODE_COL ? BroadcastType::COL : BroadcastType::SCALAR, (HAS_ACTIVATIONS(RHS) != 0)>(
                dfb_post_rhs,
                dfb_held_rhs,
                dfb_out_id,
                n,
                (batch_changed & 2u) != 0,
                holding_rhs,
                [&]([[maybe_unused]] uint32_t i) { PROCESS_ACTIVATIONS(RHS, i); });
#endif
            process_sfpu_tiles(n, dfb_pre_lhs, dfb_bin_lhs, dfb_pre_rhs, dfb_bin_rhs, dfb_out ISCLOSE_RT_ARG_FWD);
        });
#if A_HELD
    if (holding_lhs) {
        dfb_held_lhs.pop_front(1);
    }
#endif
#if B_HELD
    if (holding_rhs) {
        dfb_held_rhs.pop_front(1);
    }
#endif
#else
    const uint32_t num_full_chunks = my_tiles / num_tiles_per_cycle;
    for (uint32_t chunk = 0; chunk < num_full_chunks; ++chunk) {
        process_sfpu_tiles(
            num_tiles_per_cycle, dfb_pre_lhs, dfb_bin_lhs, dfb_pre_rhs, dfb_bin_rhs, dfb_out ISCLOSE_RT_ARG_FWD);
    }
    // The batch need not divide a thread's share; a remainder left behind would hang the writer or leave
    // borrowed output tiles uncomputed.
    const uint32_t remainder = my_tiles % num_tiles_per_cycle;
    if (remainder > 0) {
        process_sfpu_tiles(remainder, dfb_pre_lhs, dfb_bin_lhs, dfb_pre_rhs, dfb_bin_rhs, dfb_out ISCLOSE_RT_ARG_FWD);
    }
#endif
#if A_VALUE
    dfb_post_lhs.pop_front(1);
#endif
#if B_VALUE
    dfb_post_rhs.pop_front(1);
#endif
#else
    // No borrowed ring: a shard smaller than C lives entirely in the tail rings.
    compute_kernel_hw_startup(static_cast<uint32_t>(dfb::pre_lhs_tail), static_cast<uint32_t>(dfb::out_tail));
    BINARY_SFPU_INIT
#endif
#if TAIL_TILES
#if HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST) or defined(PACK_RELU)
#error "the tail rings carry no activation chain or fused RELU: the factory runs no tail beside one"
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
        // pack_tile keeps writing to the ring the packer was set up for, whatever id it is given, so
        // retarget it; the format is the output's already.
        pack_init(dfb_tail_out_id);
        dfb_tail_lhs.wait_front(1);
        dfb_tail_rhs.wait_front(1);
        dfb_tail_out.reserve_back(1);
        tile_regs_acquire();
        copy_init(dfb_tail_lhs_id);
        copy_tile(dfb_tail_lhs_id, 0, 0);
        reconfig_data_format_srca(dfb_tail_lhs_id, dfb_tail_rhs_id);
        copy_init(dfb_tail_rhs_id);
        copy_tile(dfb_tail_rhs_id, 0, 1);
#if ISCLOSE_OP
        BINARY_SFPU_OP(0, 1, 0, rtol_bits, atol_bits);
#else
        BINARY_SFPU_OP(0, 1, 0);
#endif
        reconfig_data_format_srca(dfb_tail_rhs_id, dfb_tail_lhs_id);
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
