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
    uint32_t curr_k_block_dir = 0;
    uint32_t device_id = 0;
    uint32_t device_chunk_id = 0;
    uint32_t devices_received = 0;

    MinimalMatmulOpReceiver() {}

    MinimalMatmulOpReceiver(bool wait_for_op_signal, uint32_t& rt_args_idx, uint32_t num_k_blocks) :
        wait_for_op_signal(wait_for_op_signal) {
        sem_targets[0] = 0;  // backward
        sem_targets[1] = 0;  // forward
        sem_targets[2] = 0;  // self

        // Runtime args
        uint32_t num_transfers = get_arg_val<uint32_t>(rt_args_idx++);  // TODO remove
        num_devices = get_arg_val<uint32_t>(rt_args_idx++);
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
            this->signal_op_semaphore_addr_ptrs[2] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
        }
    }

    void reset(uint32_t* k_block_device_received, uint32_t num_k_blocks) {
        device_id = my_chip_id;
        device_chunk_id = 0;
        curr_k_block_dir = 2;  // start with self
        devices_received = 0;

        for (uint32_t k = 0; k < num_k_blocks; k++) {
            k_block_device_received[k] = 0;
        }
    }

    uint32_t get_num_devices() { return num_devices; }

    // TODO: Don't compute the mapping every time, you know the device order, compute it once
    uint32_t compute_actual_k_block_iter(
        uint32_t* k_block_device_expected,
        uint32_t* k_block_device_received,
        uint32_t* device_k_block_counts,
        uint32_t* device_k_block_start_ids) {
        uint32_t k_block_received = 0;
        while (true) {
            if (wait_for_op_signal) {
                volatile tt_l1_ptr uint32_t* semaphore = signal_op_semaphore_addr_ptrs[curr_k_block_dir];
                uint32_t sem_target = sem_targets[curr_k_block_dir];
                noc_semaphore_wait_min(semaphore, sem_target + 1);
                sem_targets[curr_k_block_dir]++;
            }

            k_block_received = device_k_block_start_ids[device_id] + device_chunk_id;
            k_block_device_received[k_block_received]++;
            device_chunk_id++;
            if (device_chunk_id >= device_k_block_counts[device_id]) {
                // Move to next device
                devices_received++;
                device_chunk_id = 0;
                if (curr_k_block_dir > 0) {  // currently self or forward, next is backwards
                    curr_k_block_dir = 0;
                    int32_t unwrapped_device_id = my_chip_id - (devices_received / 2 + 1);
                    device_id = (unwrapped_device_id < 0) ? num_devices + unwrapped_device_id : unwrapped_device_id;
                } else {  // currently backwards, next is forwards
                    curr_k_block_dir = 1;
                    uint32_t unwrapped_device_id = my_chip_id + (devices_received / 2);
                    device_id =
                        unwrapped_device_id >= num_devices ? unwrapped_device_id - num_devices : unwrapped_device_id;
                }
            }
            if (k_block_device_received[k_block_received] == k_block_device_expected[k_block_received]) {
                break;
            }
        }

        return k_block_received;
    }
};
