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
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t cos_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t sin_dfb_id = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_dfb_id = get_compile_time_arg_val(3);
    constexpr uint32_t rotated_in_interm_dfb_id = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_dfb_id = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_dfb_id = get_compile_time_arg_val(6);
    constexpr uint32_t out_dfb_id = get_compile_time_arg_val(7);
    constexpr uint32_t heads_per_batch_t = get_compile_time_arg_val(8);
    constexpr uint32_t batch_per_core = get_compile_time_arg_val(9);
    constexpr auto pre_reserved_output = [](uint32_t dfb_id) {
        return ckl::output(dfb_id, ckl::ReservePolicy::None, ckl::PushPolicy::AtEnd);
    };

    DataflowBuffer dfb_in(in_dfb_id);
    DataflowBuffer dfb_cos(cos_dfb_id);
    DataflowBuffer dfb_sin(sin_dfb_id);
    DataflowBuffer dfb_trans_mat(trans_mat_dfb_id);
    DataflowBuffer dfb_rotated_in_interm(rotated_in_interm_dfb_id);
    DataflowBuffer dfb_cos_interm(cos_interm_dfb_id);
    DataflowBuffer dfb_sin_interm(sin_interm_dfb_id);
    DataflowBuffer dfb_out(out_dfb_id);

    trans_mat_dfb_id.wait_front(onetile);
    compute_kernel_hw_startup<SrcOrder::Reverse>(in_dfb_id, trans_mat_dfb_id, rotated_in_interm_dfb_id);
    matmul_init(in_dfb_id, trans_mat_dfb_id);
    compute_kernel_hw_startup(rotated_in_interm_dfb_id, sin_dfb_id, sin_interm_dfb_id);

    for (uint32_t batch_idx = 0; batch_idx < batch_per_core; ++batch_idx) {
        sin_dfb_id.reserve_back(onetile);
        cos_dfb_id.reserve_back(onetile);
        sin_dfb_id.push_back(onetile);
        cos_dfb_id.push_back(onetile);

        for (uint32_t ht = 0; ht < heads_per_batch_t; ++ht) {
            rotated_in_interm_dfb_id.reserve_back(onetile);
            sin_interm_dfb_id.reserve_back(onetile);
            cos_interm_dfb_id.reserve_back(onetile);
            out_dfb_id.reserve_back(onetile);

            in_dfb_id.reserve_back(onetile);
            in_dfb_id.push_back(onetile);
            in_dfb_id.wait_front(onetile);

            reconfig_data_format(in_dfb_id, trans_mat_dfb_id);
            pack_reconfig_data_format(rotated_in_interm_dfb_id);
            matmul_init(in_dfb_id, trans_mat_dfb_id);
            tile_regs_acquire();
            matmul_tiles(in_dfb_id, trans_mat_dfb_id, 0, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, rotated_in_interm_dfb_id);
            tile_regs_release();
            rotated_in_interm_dfb_id.push_back(onetile);

            ckl::mul<
                ckl::input(rotated_in_interm_dfb_id),
                ckl::input(sin_dfb_id, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
                pre_reserved_output(sin_interm_dfb_id)>(ckl::IterationShape::one_tile());
            ckl::mul<
                ckl::input(in_dfb_id, ckl::WaitPolicy::None, ckl::PopPolicy::AtEnd),
                ckl::input(cos_dfb_id, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None),
                pre_reserved_output(cos_interm_dfb_id)>(ckl::IterationShape::one_tile());
            ckl::add<ckl::input(cos_interm_dfb_id), ckl::input(sin_interm_dfb_id), pre_reserved_output(out_dfb_id)>(
                ckl::IterationShape::one_tile());
        }

        sin_dfb_id.pop_front(onetile);
        cos_dfb_id.pop_front(onetile);
    }
}
