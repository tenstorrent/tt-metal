// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Every core: for each expert in order, takes the expert's weight slice once and multiplies this
// core's blocks of that expert's region by it. Each output subblock is one row of tiles: K in one
// block, accumulated in DST in the order ttnn.matmul's 2D kernel uses, so the two agree bit for bit.
// The activation runs on the pack thread, beside the math of the next subblock.

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/dataflow_buffer.h"
#ifdef SFPU_ACTIVATION
#include "ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"
#endif
#include "../dataflow/expert_schedule.hpp"

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");
    constexpr uint32_t slot_n_tiles = get_named_compile_time_arg_val("slot_n_tiles");
    constexpr uint32_t subblock_w = get_named_compile_time_arg_val("subblock_w");
    constexpr bool row_major_output = get_named_compile_time_arg_val("row_major_output") == 1;
    constexpr bool transpose_weight = get_named_compile_time_arg_val("transpose_weight") == 1;
    constexpr uint32_t input_cb_id = get_named_compile_time_arg_val("input_cb");
    constexpr uint32_t weight_cb_id = get_named_compile_time_arg_val("weight_cb");
    constexpr uint32_t output_cb_id = get_named_compile_time_arg_val("output_cb");
    constexpr uint32_t schedule_cb_id = get_named_compile_time_arg_val("schedule_cb");
    constexpr uint32_t slot_tiles = k_tiles * slot_n_tiles;

    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t num_subblocks = n_tiles / subblock_w;

    DataflowBuffer input_cb(input_cb_id);
    DataflowBuffer weight_cb(weight_cb_id);
    DataflowBuffer output_cb(output_cb_id);
    DataflowBuffer schedule_cb(schedule_cb_id);

#ifdef SFPU_ACTIVATION
    constexpr KernelActivation activation_type =
        static_cast<KernelActivation>(get_named_compile_time_arg_val("activation_type"));
    constexpr uint32_t activation_param0 = get_named_compile_time_arg_val("activation_param0");
    ActivationInitHelper<activation_type, activation_param0>::init();
#endif

    compute_kernel_hw_startup<SrcOrder::Reverse>(input_cb_id, weight_cb_id, output_cb_id);
    matmul_block_init(input_cb_id, weight_cb_id, transpose_weight, subblock_w, 1, k_tiles);
    if constexpr (row_major_output) {
        pack_untilize_dest_init<subblock_w, slot_n_tiles>(output_cb_id);
    }

    schedule_cb.wait_front(1);
    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        const uint32_t expert_blocks = read_tile_value(schedule_cb_id, 0, expert);
        if (expert_blocks == grouped_matmul::kSkipExpert) {
            continue;
        }
        weight_cb.wait_front(slot_tiles);
        for (uint32_t block = 0; block < expert_blocks; ++block) {
            input_cb.wait_front(k_tiles);
            if constexpr (row_major_output) {
                output_cb.reserve_back(n_tiles);
            }
            for (uint32_t subblock = 0; subblock < num_subblocks; ++subblock) {
                tile_regs_acquire();
                uint32_t weight_index = subblock * subblock_w;
                for (uint32_t k = 0; k < k_tiles; ++k) {
                    matmul_block(
                        input_cb_id, weight_cb_id, k, weight_index, 0, transpose_weight, subblock_w, 1, k_tiles);
                    weight_index += slot_n_tiles;
                }
                tile_regs_commit();
                if constexpr (!row_major_output) {
                    output_cb.reserve_back(subblock_w);
                }
#ifdef SFPU_ACTIVATION
                apply_activation_from_pack<activation_type, activation_param0>(subblock_w);
#else
                tile_regs_wait();
#endif
                if constexpr (row_major_output) {
                    pack_untilize_dest<subblock_w, slot_n_tiles>(output_cb_id, 1, subblock);
                } else {
                    pack_block(0, output_cb_id, subblock_w);
                }
                tile_regs_release();
                if constexpr (!row_major_output) {
                    output_cb.push_back(subblock_w);
                }
            }
            if constexpr (row_major_output) {
                output_cb.push_back(n_tiles);
            }
            input_cb.pop_front(k_tiles);
        }
        weight_cb.pop_front(slot_tiles);
    }
    schedule_cb.pop_front(1);
}
