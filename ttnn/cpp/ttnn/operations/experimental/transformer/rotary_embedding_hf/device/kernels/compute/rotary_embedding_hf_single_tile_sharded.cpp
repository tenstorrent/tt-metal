// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_in_interm_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t heads_per_batch_t = get_compile_time_arg_val(8);
    constexpr uint32_t batch_per_core = get_compile_time_arg_val(9);
    constexpr auto pre_reserved_output = [](uint32_t cb) {
        return ckl::output(cb, ckl::OutputLifecycle::ReserveNonePushEnd);
    };

    CircularBuffer in_cb(in_cb_id);
    CircularBuffer cos_cb(cos_cb_id);
    CircularBuffer sin_cb(sin_cb_id);
    CircularBuffer trans_mat_cb(trans_mat_cb_id);
    CircularBuffer rotated_in_interm_cb(rotated_in_interm_cb_id);
    CircularBuffer cos_interm_cb(cos_interm_cb_id);
    CircularBuffer sin_interm_cb(sin_interm_cb_id);
    CircularBuffer out_cb(out_cb_id);

    trans_mat_cb.wait_front(onetile);
    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb_id, trans_mat_cb_id, rotated_in_interm_cb_id);
    matmul_init(in_cb_id, trans_mat_cb_id);
    binary_op_init_common(rotated_in_interm_cb_id, sin_cb_id, sin_interm_cb_id);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        sin_cb.reserve_back(onetile);
        cos_cb.reserve_back(onetile);
        sin_cb.push_back(onetile);
        cos_cb.push_back(onetile);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            rotated_in_interm_cb.reserve_back(onetile);
            sin_interm_cb.reserve_back(onetile);
            cos_interm_cb.reserve_back(onetile);
            out_cb.reserve_back(onetile);

            in_cb.reserve_back(onetile);
            in_cb.push_back(onetile);
            in_cb.wait_front(onetile);

            reconfig_data_format(in_cb_id, trans_mat_cb_id);
            pack_reconfig_data_format(rotated_in_interm_cb_id);
            matmul_init(in_cb_id, trans_mat_cb_id);
            tile_regs_acquire();
            matmul_tiles(in_cb_id, trans_mat_cb_id, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, rotated_in_interm_cb_id);
            tile_regs_release();
            rotated_in_interm_cb.push_back(onetile);

            ckl::mul<
                ckl::input(rotated_in_interm_cb_id),
                ckl::input(sin_cb_id, ckl::InputLifecycle::HeldBulk),
                pre_reserved_output(sin_interm_cb_id),
                ckl::BroadcastDim::Row>(ckl::EltwiseShape::single());
            ckl::mul<
                ckl::input(in_cb_id, ckl::InputLifecycle::DeferredPop),
                ckl::input(cos_cb_id, ckl::InputLifecycle::HeldBulk),
                pre_reserved_output(cos_interm_cb_id),
                ckl::BroadcastDim::Row>(ckl::EltwiseShape::single());
            ckl::add<ckl::input(cos_interm_cb_id), ckl::input(sin_interm_cb_id), pre_reserved_output(out_cb_id)>(
                ckl::EltwiseShape::single());
        }

        sin_cb.pop_front(onetile);
        cos_cb.pop_front(onetile);
    }
}
