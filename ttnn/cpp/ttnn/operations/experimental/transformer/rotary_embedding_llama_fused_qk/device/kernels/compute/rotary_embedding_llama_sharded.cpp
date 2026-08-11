// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    const bool is_q = get_arg_val<uint32_t>(0);

    // First 6 args for q and k heads
    // - First 3 are for q
    // - Next 3 are for k
    constexpr uint32_t q_in_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t q_out_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t q_Ht = get_compile_time_arg_val(2);
    constexpr uint32_t k_in_dfb_id = get_compile_time_arg_val(3);
    constexpr uint32_t k_out_dfb_id = get_compile_time_arg_val(4);
    constexpr uint32_t k_Ht = get_compile_time_arg_val(5);
    uint32_t in_dfb_id = q_in_dfb_id;
    uint32_t out_dfb_id = q_out_dfb_id;
    uint32_t Ht = q_Ht;
    if (!is_q) {
        in_dfb_id = k_in_dfb_id;
        out_dfb_id = k_out_dfb_id;
        Ht = k_Ht;
    }

    constexpr uint32_t Wt = get_compile_time_arg_val(6);  // How many rows (tiles) in n_heads dimension

    constexpr uint32_t cos_dfb_id = get_compile_time_arg_val(7);
    constexpr uint32_t sin_dfb_id = get_compile_time_arg_val(8);
    constexpr uint32_t trans_mat_dfb_id = get_compile_time_arg_val(9);

    constexpr uint32_t rotated_in_interm_dfb_id = get_compile_time_arg_val(10);
    constexpr uint32_t cos_interm_dfb_id = get_compile_time_arg_val(11);
    constexpr uint32_t sin_interm_dfb_id = get_compile_time_arg_val(12);

    DataflowBuffer dfb_in(in_dfb_id);
    DataflowBuffer dfb_out(out_dfb_id);
    DataflowBuffer dfb_rotated_in_interm(rotated_in_interm_dfb_id);
    DataflowBuffer dfb_cos_interm(cos_interm_dfb_id);
    DataflowBuffer dfb_sin_interm(sin_interm_dfb_id);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_dfb_id, trans_mat_dfb_id, out_dfb_id);
    matmul_init(in_dfb_id, trans_mat_dfb_id);
    compute_kernel_hw_startup(rotated_in_interm_dfb_id, sin_dfb_id, sin_interm_dfb_id);

    for (uint32_t ht = 0; ht < Ht; ht++) {  // Over n_heads_t dimension
        dfb_rotated_in_interm.reserve_back(Wt);
        dfb_sin_interm.reserve_back(Wt);
        dfb_cos_interm.reserve_back(Wt);
        dfb_out.reserve_back(Wt);

        // Get the input
        dfb_in.reserve_back(Wt);
        dfb_in.push_back(Wt);
        dfb_in.wait_front(Wt);

        // Do the computation

        // rotated = x @ trans_mat
        matmul_init(in_dfb_id, trans_mat_dfb_id);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            matmul_tiles(in_dfb_id, trans_mat_dfb_id, j, 0, j);
            pack_tile(j, rotated_in_interm_dfb_id, j);
        }
        REL();
        dfb_rotated_in_interm.push_back(Wt);

        ckl::mul<
            ckl::input(
                rotated_in_interm_dfb_id, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            ckl::input(
                sin_dfb_id,
                ckl::BroadcastDim::Row,
                ckl::WaitPolicy::None,
                ckl::PopPolicy::None,
                ckl::OperandKind::Block),
            ckl::output(sin_interm_dfb_id, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd)>(
            ckl::IterationShape::tiles(Wt).block_size(/*block_size=*/Wt));

        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // cos_interim = x * cos
            mul_tiles_bcast<BroadcastType::ROW>(in_dfb_id, cos_dfb_id, j, j, j);
            pack_tile(j, cos_interm_dfb_id, j);
        }
        REL();
        dfb_cos_interm.push_back(Wt);
        dfb_in.pop_front(Wt);  // Done with input

        dfb_sin_interm.wait_front(Wt);
        dfb_cos_interm.wait_front(Wt);
        add_init(cos_interm_dfb_id, sin_interm_dfb_id);
        ACQ();
        for (uint32_t j = 0; j < Wt; ++j) {
            // out = cos_interim + sin_interim
            add_tiles(cos_interm_dfb_id, sin_interm_dfb_id, j, j, j);
            pack_tile(j, out_dfb_id, j);
        }
        REL();
        dfb_out.push_back(Wt);
        dfb_sin_interm.pop_front(Wt);
        dfb_cos_interm.pop_front(Wt);
    }
}
