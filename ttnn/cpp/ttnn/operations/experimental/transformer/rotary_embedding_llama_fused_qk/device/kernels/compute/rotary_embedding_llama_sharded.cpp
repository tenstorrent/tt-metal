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
    // const bool has_work = get_arg_val<uint32_t>(0);
    // if (!has_work) {
    //     return;
    // }
    const bool is_q = get_arg(args::is_q);

    // First 6 args for q and k heads
    // - First 3 are for q
    // - Next 3 are for k
    // The CB handles are DFB bindings (dfb::); the per-head tile counts are named CTAs (args::).
    constexpr uint32_t q_Ht = get_arg(args::q_Ht);
    constexpr uint32_t k_Ht = get_arg(args::k_Ht);
    uint32_t in_cb = dfb::q_in_cb;
    uint32_t out_cb = dfb::q_out_cb;
    uint32_t Ht = q_Ht;
    if (!is_q) {
        in_cb = dfb::k_in_cb;
        out_cb = dfb::k_out_cb;
        Ht = k_Ht;
    }

    constexpr uint32_t Wt = get_arg(args::Wt);  // How many rows (tiles) in n_heads dimension

    DataflowBuffer in_cb_obj(in_cb);
    DataflowBuffer out_cb_obj(out_cb);
    DataflowBuffer cos_cb_obj(dfb::cos_cb);
    DataflowBuffer sin_cb_obj(dfb::sin_cb);
    DataflowBuffer trans_mat_cb_obj(dfb::trans_mat_cb);
    DataflowBuffer rotated_in_interm_cb_obj(dfb::rotated_in_interm_cb);
    DataflowBuffer cos_interm_cb_obj(dfb::cos_interm_cb);
    DataflowBuffer sin_interm_cb_obj(dfb::sin_interm_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb, dfb::trans_mat_cb, out_cb);
    matmul_init(in_cb, dfb::trans_mat_cb);
    binary_op_init_common(
        dfb::rotated_in_interm_cb, dfb::sin_cb, dfb::sin_interm_cb);  // General Init for all binary ops

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
        matmul_init(in_cb, dfb::trans_mat_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_cb, dfb::trans_mat_cb, j, 0, j);
            pack_tile(j, dfb::rotated_in_interm_cb, j);
        }
        REL();
        rotated_in_interm_cb_obj.push_back(Wt);
        rotated_in_interm_cb_obj.wait_front(Wt);

        mul_bcast_rows_init_short(dfb::rotated_in_interm_cb, dfb::sin_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // sin_interim = rotated * sin
            mul_tiles_bcast<BroadcastType::ROW>(dfb::rotated_in_interm_cb, dfb::sin_cb, j, j, j);
            pack_tile(j, dfb::sin_interm_cb, j);
        }
        REL();
        sin_interm_cb_obj.push_back(Wt);
        rotated_in_interm_cb_obj.pop_front(Wt);

        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // cos_interim = x * cos
            mul_tiles_bcast<BroadcastType::ROW>(in_cb, dfb::cos_cb, j, j, j);
            pack_tile(j, dfb::cos_interm_cb, j);
        }
        REL();
        cos_interm_cb_obj.push_back(Wt);
        in_cb_obj.pop_front(Wt);  // Done with input

        sin_interm_cb_obj.wait_front(Wt);
        cos_interm_cb_obj.wait_front(Wt);
        add_tiles_init(dfb::cos_interm_cb, dfb::sin_interm_cb);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // out = cos_interim + sin_interim
            add_tiles(dfb::cos_interm_cb, dfb::sin_interm_cb, j, j, j);
            pack_tile(j, out_cb, j);
        }
        REL();
        out_cb_obj.push_back(Wt);
        sin_interm_cb_obj.pop_front(Wt);
        cos_interm_cb_obj.pop_front(Wt);
    }
}
