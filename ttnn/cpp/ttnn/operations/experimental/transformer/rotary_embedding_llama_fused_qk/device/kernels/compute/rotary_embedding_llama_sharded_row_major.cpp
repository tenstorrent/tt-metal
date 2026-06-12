// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
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
    // const bool has_work = get_arg_val<uint32_t>(0);
    // if (!has_work) {
    //     return;
    // }
    const bool is_q = get_arg(args::is_q);

    // First 6 args for q and k heads
    // - First 3 are for q
    // - Next 3 are for k
    // q/k in/out CBs are selected at runtime from is_q; both candidate DFBs are bound on
    // this kernel, so the runtime-dynamic CB id flows through the DFBAccessor->uint32_t
    // implicit conversion.
    constexpr uint32_t q_Ht = get_arg(args::q_Ht);
    constexpr uint32_t k_Ht = get_arg(args::k_Ht);
    uint32_t in_cb = dfb::q_in;
    uint32_t out_cb = dfb::q_out;
    uint32_t Ht = q_Ht;
    if (!is_q) {
        in_cb = dfb::k_in;
        out_cb = dfb::k_out;
        Ht = k_Ht;
    }

    constexpr uint32_t Wt = get_arg(args::Wt);  // How many tiles in wrapped RM inputs

    constexpr auto cos_cb = dfb::cos;
    constexpr auto sin_cb = dfb::sin;
    constexpr auto trans_mat_cb = dfb::trans_mat;

    constexpr auto rotated_in_interm_cb = dfb::rotated_in_interm;
    constexpr auto cos_interm_cb = dfb::cos_interm;
    constexpr auto sin_interm_cb = dfb::sin_interm;

    DataflowBuffer in_cb_obj(in_cb);
    DataflowBuffer out_cb_obj(out_cb);
    DataflowBuffer rotated_in_interm_cb_obj(rotated_in_interm_cb);
    DataflowBuffer cos_interm_cb_obj(cos_interm_cb);
    DataflowBuffer sin_interm_cb_obj(sin_interm_cb);

    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, sin_cb, sin_interm_cb);  // General Init for all binary ops

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        rotated_in_interm_cb_obj.reserve_back(Wt);
        sin_interm_cb_obj.reserve_back(Wt);
        cos_interm_cb_obj.reserve_back(Wt);
        out_cb_obj.reserve_back(Wt);

        // Get the input
        in_cb_obj.reserve_back(Wt);
        in_cb_obj.push_back(Wt);
        in_cb_obj.wait_front(Wt);

        // Do the computation

        // rotated = x @ trans_mat
        mm_init_short(in_cb, trans_mat_cb);
        ACQ();

        matmul_tiles(in_cb, trans_mat_cb, 0, 0, 0);
        pack_tile(0, rotated_in_interm_cb, 0);

        REL();
        rotated_in_interm_cb_obj.push_back(Wt);
        rotated_in_interm_cb_obj.wait_front(Wt);

        mul_tiles_init(rotated_in_interm_cb, sin_cb);
        ACQ();
        // sin_interim = rotated * sin
        mul_tiles(rotated_in_interm_cb, sin_cb, 0, 0, 0);
        pack_tile(0, sin_interm_cb, 0);
        REL();
        sin_interm_cb_obj.push_back(Wt);
        rotated_in_interm_cb_obj.pop_front(Wt);

        mul_tiles_init(in_cb, cos_cb);
        ACQ();
        // cos_interim = x * cos
        mul_tiles(in_cb, cos_cb, 0, 0, 0);
        pack_tile(0, cos_interm_cb, 0);
        REL();
        cos_interm_cb_obj.push_back(Wt);
        in_cb_obj.pop_front(Wt);  // Done with input

        sin_interm_cb_obj.wait_front(Wt);
        cos_interm_cb_obj.wait_front(Wt);
        add_tiles_init(cos_interm_cb, sin_interm_cb);
        ACQ();
        // out = cos_interim + sin_interim
        add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0);
        pack_tile(0, out_cb, 0);
        REL();
        out_cb_obj.push_back(Wt);
        sin_interm_cb_obj.pop_front(Wt);
        cos_interm_cb_obj.pop_front(Wt);
    }
}
