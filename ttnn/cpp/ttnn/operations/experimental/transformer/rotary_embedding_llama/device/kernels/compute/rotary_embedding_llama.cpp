// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

void kernel_main() {
    uint32_t argrt = 0;
    uint32_t batch_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_end = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t n_heads = get_compile_time_arg_val(9);

    const uint32_t my_seq_tiles = seq_t_end - seq_t_start;
    const uint32_t my_cos_sin_tiles = my_seq_tiles * Wt;

    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, cos_cb, out_cb);  // General Init for all binary ops

    // Get the trans_mat
    cb_wait_front(trans_mat_cb, onetile);

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
#if RELOAD_IMPL == 0
        cb_wait_front(sin_cb, my_cos_sin_tiles);
        cb_wait_front(cos_cb, my_cos_sin_tiles);
#endif
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            uint32_t sin_cos_row_cnt = 0;
            for (uint32_t seq_tile = seq_t_start; seq_tile < seq_t_end; ++seq_tile) {
                // input cb wait and reserve
                cb_wait_front(in_cb, Wt);
#if RELOAD_IMPL == 1
                cb_wait_front(sin_cb, Wt);
                cb_wait_front(cos_cb, Wt);
#endif

                cb_reserve_back(rotated_in_interm_cb, Wt);
                cb_reserve_back(out_cb, Wt);

                // rotated = x @ trans_mat (matmul — not covered by binary_op helpers)
                mm_init_short(in_cb, trans_mat_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(in_cb, trans_mat_cb, j, in1_index, j);
                    pack_tile(j, rotated_in_interm_cb, j);
                }
                REL();
                cb_push_back(rotated_in_interm_cb, Wt);
                cb_wait_front(rotated_in_interm_cb, Wt);

                // sin_interim = rotated * sin  — caller pre-waited both CBs.
                // B's absolute index is j + (sin_cos_row_cnt * Wt) — exactly what
                // BinaryInputExtras::base expresses. Bulk output manages cb_reserve/push.
                // First mul call does init+reconfig (switching out of matmul state).
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                    compute_kernel_lib::BinaryOutputPolicy::Bulk>(
                    rotated_in_interm_cb,
                    sin_cb,
                    sin_interm_cb,
                    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt),
                    compute_kernel_lib::BinaryInputExtras{.base = 0},
                    compute_kernel_lib::BinaryInputExtras{.base = sin_cos_row_cnt * Wt});
                cb_pop_front(rotated_in_interm_cb, Wt);

                // cos_interim = x * cos  — same shape as sin_interim.
                // Second mul call: skip reconfig + init (matches the pre-migration
                // single mul_tiles_init across both loops).
                compute_kernel_lib::mul<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
                    compute_kernel_lib::BinaryOutputPolicy::Bulk,
                    compute_kernel_lib::BinaryDataFormatReconfig::NONE,
                    /*init=*/false>(
                    in_cb,
                    cos_cb,
                    cos_interm_cb,
                    compute_kernel_lib::BinaryInputBlockShape::of(1, Wt),
                    compute_kernel_lib::BinaryInputExtras{.base = 0},
                    compute_kernel_lib::BinaryInputExtras{.base = sin_cos_row_cnt * Wt});
                cb_pop_front(in_cb, Wt);  // Done with input
#if RELOAD_IMPL == 1
                cb_pop_front(sin_cb, Wt);
                cb_pop_front(cos_cb, Wt);
#endif

                // out = cos_interim + sin_interim (sequential NONE-bcast, full Wt block)
                // Both operands are produced by this kernel just above, waited upfront and
                // popped at end; output is reserved bulk and pushed once.
                compute_kernel_lib::add<
                    compute_kernel_lib::BroadcastDim::NONE,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
                    compute_kernel_lib::BinaryOutputPolicy::Bulk>(
                    cos_interm_cb, sin_interm_cb, out_cb, compute_kernel_lib::BinaryInputBlockShape::of(1, Wt));

#if RELOAD_IMPL == 0
                // no-reload needs to increment this counter
                // Used a sin/cos row
                sin_cos_row_cnt++;
#endif
            }
        }

#if RELOAD_IMPL == 0
        cb_pop_front(sin_cb, my_cos_sin_tiles);
        cb_pop_front(cos_cb, my_cos_sin_tiles);
#endif
    }

    // Done with the transformation matrix, so remove from CB
    cb_pop_front(trans_mat_cb, onetile);
}
