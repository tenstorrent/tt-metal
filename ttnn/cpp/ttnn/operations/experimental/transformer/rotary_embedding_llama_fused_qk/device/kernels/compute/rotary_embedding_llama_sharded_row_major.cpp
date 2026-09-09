// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    // TODO: Add back early return? Currently, running out of code size in TRISC2 by 4B
    // const bool has_work = get_arg(args::has_work);
    // if (!has_work) {
    //     return;
    // }
    const bool is_q = get_arg(args::is_q);

    // The q and k halves each carry their own in/out DFBs and head count (Ht); the per-core
    // is_q runtime arg selects which half this core works on.
    constexpr uint32_t q_in_dfb = dfb::q_input;
    constexpr uint32_t q_out_dfb = dfb::q_out;
    constexpr uint32_t q_Ht = get_arg(args::q_Ht);
    constexpr uint32_t k_in_dfb = dfb::k_input;
    constexpr uint32_t k_out_dfb = dfb::k_out;
    constexpr uint32_t k_Ht = get_arg(args::k_Ht);
    uint32_t in_dfb = q_in_dfb;
    uint32_t out_dfb = q_out_dfb;
    uint32_t Ht = q_Ht;
    if (!is_q) {
        in_dfb = k_in_dfb;
        out_dfb = k_out_dfb;
        Ht = k_Ht;
    }

    constexpr uint32_t Wt = get_arg(args::Wt);  // How many tiles in wrapped RM inputs

    constexpr uint32_t cos_dfb = dfb::cos;
    constexpr uint32_t sin_dfb = dfb::sin;
    constexpr uint32_t trans_mat_dfb = dfb::trans_mat;

    constexpr uint32_t rotated_in_interm_dfb = dfb::rotated_interm;
    constexpr uint32_t cos_interm_dfb = dfb::cos_interm;
    constexpr uint32_t sin_interm_dfb = dfb::sin_interm;

    // The in/out DFB identity is runtime-selected (q vs k), so these objects are constructed
    // from the selected id (the dfb:: tokens carry the same ids).
    DataflowBuffer in_dfb_obj(in_dfb);
    DataflowBuffer out_dfb_obj(out_dfb);
    DataflowBuffer rotated_in_interm_dfb_obj(rotated_in_interm_dfb);
    DataflowBuffer cos_interm_dfb_obj(cos_interm_dfb);
    DataflowBuffer sin_interm_dfb_obj(sin_interm_dfb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_dfb, trans_mat_dfb, out_dfb);
    matmul_init(in_dfb, trans_mat_dfb);
    compute_kernel_hw_startup(rotated_in_interm_dfb, sin_dfb, sin_interm_dfb);  // General Init for all binary ops

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        rotated_in_interm_dfb_obj.reserve_back(Wt);
        sin_interm_dfb_obj.reserve_back(Wt);
        cos_interm_dfb_obj.reserve_back(Wt);
        out_dfb_obj.reserve_back(Wt);

        // Get the input
        in_dfb_obj.reserve_back(Wt);
        in_dfb_obj.push_back(Wt);
        in_dfb_obj.wait_front(Wt);

        // Do the computation

        // rotated = x @ trans_mat
        matmul_init(in_dfb, trans_mat_dfb);
        ACQ();

        matmul_tiles(in_dfb, trans_mat_dfb, 0, 0, 0);
        pack_tile(0, rotated_in_interm_dfb, 0);

        REL();
        rotated_in_interm_dfb_obj.push_back(Wt);
        rotated_in_interm_dfb_obj.wait_front(Wt);

        mul_init(rotated_in_interm_dfb, sin_dfb);
        ACQ();
        // sin_interim = rotated * sin
        mul_tiles(rotated_in_interm_dfb, sin_dfb, 0, 0, 0);
        pack_tile(0, sin_interm_dfb, 0);
        REL();
        sin_interm_dfb_obj.push_back(Wt);
        rotated_in_interm_dfb_obj.pop_front(Wt);

        mul_init(in_dfb, cos_dfb);
        ACQ();
        // cos_interim = x * cos
        mul_tiles(in_dfb, cos_dfb, 0, 0, 0);
        pack_tile(0, cos_interm_dfb, 0);
        REL();
        cos_interm_dfb_obj.push_back(Wt);
        in_dfb_obj.pop_front(Wt);  // Done with input

        sin_interm_dfb_obj.wait_front(Wt);
        cos_interm_dfb_obj.wait_front(Wt);
        add_init(cos_interm_dfb, sin_interm_dfb);
        ACQ();
        // out = cos_interim + sin_interim
        add_tiles(cos_interm_dfb, sin_interm_dfb, 0, 0, 0);
        pack_tile(0, out_dfb, 0);
        REL();
        out_dfb_obj.push_back(Wt);
        sin_interm_dfb_obj.pop_front(Wt);
        cos_interm_dfb_obj.pop_front(Wt);
    }
}
