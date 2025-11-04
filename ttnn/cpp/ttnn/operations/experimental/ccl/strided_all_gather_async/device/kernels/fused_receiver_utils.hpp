// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <array>

struct MinimalMatmulOpReceiver {
    bool wait_for_op_signal = false;
    uint32_t num_k_blocks_per_device = 0;
    uint32_t num_devices = 0;
    uint32_t my_chip_id = 0;
    std::array<volatile tt_l1_ptr uint32_t*, 3> signal_op_semaphore_addr_ptrs = {};  // backward, forward, self
    std::array<uint32_t, 3> sem_targets = {};
    uint32_t curr_k_block_source = 0;
    uint32_t target_k_block_slice = 0;

    MinimalMatmulOpReceiver() {}

    MinimalMatmulOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx, uint32_t num_k_blocks) :
        wait_for_op_signal(wait_for_op_signal) {
        sem_targets[0] = 0;  // backward
        sem_targets[1] = 0;  // forward

        // Runtime args
        uint32_t num_transfers = get_arg_val<uint32_t>(rt_args_idx++);  // TODO remove
        uint32_t num_devices = get_arg_val<uint32_t>(rt_args_idx++);
        num_k_blocks_per_device = num_k_blocks / num_devices;
        my_chip_id = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t tensor_slice_shape_width = get_arg_val<uint32_t>(rt_args_idx++);  // TODO remove
        uint32_t output_page_offset = get_arg_val<uint32_t>(rt_args_idx++);        // TODO remove
        uint32_t last_output_page_offset = get_arg_val<uint32_t>(rt_args_idx++);   // TODO remove
        uint32_t is_clockwise_direction = get_arg_val<uint32_t>(rt_args_idx++);    // TODO remove

        if (this->wait_for_op_signal) {
            this->signal_op_semaphore_addr_ptrs[0] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
            this->signal_op_semaphore_addr_ptrs[1] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
        }
    }

    uint32_t compute_actual_k_block_iter(const uint32_t& curr_k_block_iter) {
        uint32_t k_block_slice_iter = curr_k_block_iter % num_k_blocks_per_device;
        if (k_block_slice_iter == 0) {
            uint32_t device_index = curr_k_block_iter / num_k_blocks_per_device;
            if (device_index == 0) {
                curr_k_block_source = 1;
                target_k_block_slice = my_chip_id;
            } else if (device_index % 2) {
                curr_k_block_source = 0;
                target_k_block_slice = my_chip_id - (device_index / 2 + 1);
            } else {
                curr_k_block_source = 1;
                target_k_block_slice = my_chip_id + (device_index / 2);
            }
        }
        if (wait_for_op_signal) {
            volatile tt_l1_ptr uint32_t* semaphore = signal_op_semaphore_addr_ptrs[curr_k_block_source];
            uint32_t sem_target = sem_targets[curr_k_block_source];
            noc_semaphore_wait_min(semaphore, sem_target + 1);
            sem_targets[curr_k_block_source]++;
        }

        return (target_k_block_slice * num_k_blocks_per_device) + k_block_slice_iter;
    }
};
