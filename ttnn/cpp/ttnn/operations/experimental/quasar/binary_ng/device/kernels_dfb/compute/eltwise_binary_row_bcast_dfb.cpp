// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) FPU compute kernel for binary_ng's ROW subtile-broadcast binary op.
//
// 1:1 port of the CircularBuffer kernels_ng/compute/eltwise_binary_row_bcast.cpp, with the CB->DFB
// swap (mirroring eltwise_binary_no_bcast_dfb.cpp's API style: get_arg(args::...), dfb::<name> accessor
// ids, DataflowBuffer instances). Per output tile the kernel runs a broadcast pass that expands the
// partial tile the reader delivered (unary_bcast<ROW>) into the intermediate llk_post DFB, then the
// normal binary-op body (identical to the no-bcast kernel).
//
// DFB operand naming (SRC_BCAST -> a is the broadcast operand; SRC_BCAST_B -> b is):
//   dfb::pre_lhs (c_0)  dfb::pre_rhs (c_1)  dfb::out (c_2)  dfb::llk_post (c_5 for SRC_BCAST / c_6 for
//   SRC_BCAST_B, the unary_bcast output).  post_lhs/post_rhs (c_3/c_4) exist only when that operand has
//   an activation chain; without activations the post id aliases the pre id, and the broadcast operand's
//   pre id aliases llk_post (the expanded tile feeds the binary op).

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "experimental/kernel_args.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_dfb.hpp"

void kernel_main() {
    const uint32_t num_tiles = get_arg(args::num_tiles);

    constexpr uint32_t num_tiles_per_cycle = get_arg(args::num_tiles_per_cycle);

    constexpr auto dfb_out_id = static_cast<uint32_t>(dfb::out);

    // The broadcast operand's raw (partial) tile comes in on its pre_* DFB; unary_bcast expands it into
    // llk_post, and the operand's pre id used by the binary op then aliases llk_post (the expanded tile).
#if SRC_BCAST
    constexpr auto dfb_bcast_id = static_cast<uint32_t>(dfb::pre_lhs);      // c_0: raw a (broadcast operand)
    constexpr auto dfb_llk_post_id = static_cast<uint32_t>(dfb::llk_post);  // c_5: expanded a
    constexpr auto dfb_pre_lhs_id = dfb_llk_post_id;                        // expanded a feeds LHS
    constexpr auto dfb_pre_rhs_id = static_cast<uint32_t>(dfb::pre_rhs);    // c_1: raw b
#endif
#if SRC_BCAST_B
    constexpr auto dfb_bcast_id = static_cast<uint32_t>(dfb::pre_rhs);      // c_1: raw b (broadcast operand)
    constexpr auto dfb_llk_post_id = static_cast<uint32_t>(dfb::llk_post);  // c_6: expanded b
    constexpr auto dfb_pre_lhs_id = static_cast<uint32_t>(dfb::pre_lhs);    // c_0: raw a
    constexpr auto dfb_pre_rhs_id = dfb_llk_post_id;                        // expanded b feeds RHS
#endif

    // post_* DFBs (c_3/c_4) exist only when that operand has an activation chain (guarded like the
    // no-bcast kernel). Without activations the post id aliases the pre id (PREPROCESS is a no-op and the
    // binary op reads the pre DFB directly -- which, for the broadcast operand, is the expanded llk_post).
#if HAS_ACTIVATIONS(LHS)
    constexpr auto dfb_post_lhs_id = static_cast<uint32_t>(dfb::post_lhs);
#else
    constexpr auto dfb_post_lhs_id = dfb_pre_lhs_id;
#endif
#if HAS_ACTIVATIONS(RHS)
    constexpr auto dfb_post_rhs_id = static_cast<uint32_t>(dfb::post_rhs);
#else
    constexpr auto dfb_post_rhs_id = dfb_pre_rhs_id;
#endif

    DataflowBuffer dfb_bcast(dfb_bcast_id);
    DataflowBuffer dfb_llk_post(dfb_llk_post_id);
    DataflowBuffer dfb_post_lhs(dfb_post_lhs_id);
    DataflowBuffer dfb_post_rhs(dfb_post_rhs_id);
    DataflowBuffer dfb_out(dfb_out_id);

    binary_op_init_common(dfb_post_lhs_id, dfb_post_rhs_id, dfb_out_id);
#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluConfig::zero())));
#endif

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // --- Broadcast pass: partial tile (from the reader) -> full tile in the intermediate llk_post. ---
        dfb_bcast.wait_front(num_tiles_per_cycle);
        dfb_llk_post.reserve_back(num_tiles_per_cycle);
        pack_reconfig_data_format(dfb_out_id, dfb_llk_post_id);
#ifdef ARCH_QUASAR
        // On Quasar pack_reconfig_data_format reprograms only the packer format gasket, not the packer
        // destination ring (see reconfig_data_format.h and the no-bcast port's eltwise_utils_dfb.hpp);
        // retarget the packer to llk_post with pack_init so pack_tile writes there. (unary_bcast_init
        // below also re-inits the packer for llk_post, so this is belt-and-suspenders on the LHS side.)
        pack_init(dfb_llk_post_id);
#endif
        unary_bcast_init<BroadcastType::ROW>(dfb_bcast_id, dfb_llk_post_id);

        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(dfb_bcast_id, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb_llk_post_id);
        dfb_llk_post.push_back(num_tiles_per_cycle);
        tile_regs_release();
        dfb_bcast.pop_front(num_tiles_per_cycle);

        pack_reconfig_data_format(dfb_llk_post_id, dfb_out_id);
#ifdef ARCH_QUASAR
        // Retarget the packer destination ring back to dfb_out for the binary-op pack below; without
        // this the gasket-only pack_reconfig above leaves the ring on llk_post and pack_tile(0, out)
        // writes the wrong buffer (the ~constant-output symptom). Mirrors eltwise_utils_dfb.hpp.
        pack_init(dfb_out_id);
#endif
#if defined(ARCH_BLACKHOLE)
        PACK((llk_pack_hw_configure<DST_ACCUM_MODE>(dfb_out_id)));
#elif defined(ARCH_QUASAR)
        PACK((llk_pack_hw_configure(dfb_out_id)));
#endif

        // --- Binary op (verbatim from eltwise_binary_no_bcast_dfb.cpp's body, single tile). ---
        PREPROCESS(LHS, dfb_pre_lhs_id, dfb_post_lhs_id, dfb_out_id, num_tiles_per_cycle);
        dfb_post_lhs.wait_front(num_tiles_per_cycle);

        PREPROCESS(RHS, dfb_pre_rhs_id, dfb_post_rhs_id, dfb_out_id, num_tiles_per_cycle);
        dfb_post_rhs.wait_front(num_tiles_per_cycle);

        binary_tiles_init<true, BINARY_OP_TYPE>(dfb_post_lhs_id, dfb_post_rhs_id);
        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();
        BINARY_OP(dfb_post_lhs_id, dfb_post_rhs_id, 0, 0, 0);
        PROCESS_POST_ACTIVATIONS(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb_out_id);
        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_post_lhs.pop_front(num_tiles_per_cycle);
        dfb_post_rhs.pop_front(num_tiles_per_cycle);
    }
}
