// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

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
    const bool is_q = get_arg_val<uint32_t>(0);

    // First 6 args for q and k heads
    // - First 3 are for q
    // - Next 3 are for k
    constexpr uint32_t q_in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_cb = get_compile_time_arg_val(1);
    constexpr uint32_t q_Ht = get_compile_time_arg_val(2);
    constexpr uint32_t k_in_cb = get_compile_time_arg_val(3);
    constexpr uint32_t k_out_cb = get_compile_time_arg_val(4);
    constexpr uint32_t k_Ht = get_compile_time_arg_val(5);
    uint32_t in_cb = q_in_cb;
    uint32_t out_cb = q_out_cb;
    uint32_t Ht = q_Ht;
    if (!is_q) {
        in_cb = k_in_cb;
        out_cb = k_out_cb;
        Ht = k_Ht;
    }

    constexpr uint32_t Wt = get_compile_time_arg_val(6);  // How many tiles in wrapped RM inputs

    constexpr uint32_t cos_cb = get_compile_time_arg_val(7);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(8);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(9);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(10);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(11);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(12);

    CircularBuffer in_cb_obj(in_cb);
    CircularBuffer out_cb_obj(out_cb);
    CircularBuffer rotated_in_interm_cb_obj(rotated_in_interm_cb);
    CircularBuffer cos_interm_cb_obj(cos_interm_cb);
    CircularBuffer sin_interm_cb_obj(sin_interm_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb, trans_mat_cb, out_cb);
    matmul_init(in_cb, trans_mat_cb);
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
        matmul_init(in_cb, trans_mat_cb);
        ACQ();

        matmul_tiles(in_cb, trans_mat_cb, 0, 0, 0);
        pack_tile(0, rotated_in_interm_cb, 0);

        REL();
        rotated_in_interm_cb_obj.push_back(Wt);
        rotated_in_interm_cb_obj.wait_front(Wt);

        compute_kernel_lib::mul<
            compute_kernel_lib::input(rotated_in_interm_cb, compute_kernel_lib::InputLifecycle::CallerManaged),
            compute_kernel_lib::input(sin_cb, compute_kernel_lib::InputLifecycle::CallerManaged),
            compute_kernel_lib::output(
                sin_interm_cb,
                compute_kernel_lib::OutputLifecycle::CallerManaged,
                compute_kernel_lib::DataFormatReconfig::Disabled),
            compute_kernel_lib::BroadcastDim::None>(compute_kernel_lib::EltwiseShape::single());
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
