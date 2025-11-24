// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <array>

uint32_t increment_arg_idx(uint32_t& arg_idx, uint32_t num_args = 1) {
    uint32_t old_arg_idx = arg_idx;
    arg_idx += num_args;
    return old_arg_idx;
}

struct MinimalMatmulOpReceiver {
    bool wait_for_op_signal = false;
    uint32_t num_devices = 0;
    uint32_t my_chip_id = 0;
    uint32_t input_tensor_Wt = 0;
    uint32_t num_k_blocks = 0;
    std::array<volatile tt_l1_ptr uint32_t*, 3> signal_op_semaphore_addr_ptrs = {};  // backward, forward, self
    std::array<uint32_t, 3> sem_targets = {};
    uint32_t curr_k_block_dir = 0;
    uint32_t device_id = 0;
    uint32_t device_chunk_id = 0;
    uint32_t devices_received = 0;
    uint32_t* k_block_device_expected = nullptr;
    uint32_t* k_block_device_received = nullptr;
    uint32_t* device_k_block_counts = nullptr;
    uint32_t* device_k_block_start_ids = nullptr;
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;
    uint32_t next_forward = 0;
    int32_t next_backward = 0;

    MinimalMatmulOpReceiver() {}

    MinimalMatmulOpReceiver(
        bool wait_for_op_signal,
        uint32_t& rt_args_idx,
        uint32_t* k_block_device_expected,
        uint32_t* k_block_device_received,
        uint32_t* device_k_block_counts,
        uint32_t* device_k_block_start_ids) :
        wait_for_op_signal(wait_for_op_signal),
        k_block_device_expected(k_block_device_expected),
        k_block_device_received(k_block_device_received),
        device_k_block_counts(device_k_block_counts),
        device_k_block_start_ids(device_k_block_start_ids) {
        sem_targets[0] = 0;  // backward
        sem_targets[1] = 0;  // forward
        sem_targets[2] = 0;  // self

        // Runtime args
        num_devices = get_arg_val<uint32_t>(rt_args_idx++);
        num_k_blocks = get_arg_val<uint32_t>(rt_args_idx++);
        my_chip_id = get_arg_val<uint32_t>(rt_args_idx++);
        input_tensor_Wt = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t k_block_tiles = get_arg_val<uint32_t>(rt_args_idx++);
        topology = static_cast<ttnn::ccl::Topology>(get_arg_val<uint32_t>(rt_args_idx++));

        if (this->wait_for_op_signal) {
            this->signal_op_semaphore_addr_ptrs[0] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
            this->signal_op_semaphore_addr_ptrs[1] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
            this->signal_op_semaphore_addr_ptrs[2] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
        }

        uint32_t curr_device = 0;
        uint32_t curr_device_start = 0;
        uint32_t curr_device_end = input_tensor_Wt - 1;
        for (uint32_t k_block_iter = 0; k_block_iter < num_k_blocks; k_block_iter++) {
            uint32_t curr_k_block_start = k_block_iter * k_block_tiles;
            uint32_t curr_k_block_end = (k_block_iter + 1) * k_block_tiles - 1;
            if (curr_k_block_end < curr_device_end) {
                k_block_device_expected[k_block_iter]++;
                device_k_block_counts[curr_device]++;
            } else if (curr_k_block_end == curr_device_end) {
                k_block_device_expected[k_block_iter]++;
                device_k_block_counts[curr_device]++;
                curr_device++;
                curr_device_start = curr_device_end + 1;
                curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
                if (curr_device < num_devices) {
                    device_k_block_start_ids[curr_device] = k_block_iter + 1;
                }
            } else if (curr_k_block_end > curr_device_end) {
                k_block_device_expected[k_block_iter]++;
                device_k_block_counts[curr_device]++;
                if (curr_device + 1 < num_devices) {
                    k_block_device_expected[k_block_iter]++;
                    device_k_block_counts[curr_device + 1]++;
                    device_k_block_start_ids[curr_device + 1] = k_block_iter;
                }
                curr_device++;
                curr_device_start = curr_device_end + 1;
                curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
            }
        }
    }

    void reset() {
        device_id = my_chip_id;
        device_chunk_id = 0;
        curr_k_block_dir = 2;  // start with self
        devices_received = 0;

        for (uint32_t k = 0; k < num_k_blocks; k++) {
            k_block_device_received[k] = 0;
        }
    }

    uint32_t compute_actual_k_block_iter() {
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
                if (topology == ttnn::ccl::Topology::Ring) {
                    if (curr_k_block_dir > 0) {  // currently self or forward, next is backwards
                        curr_k_block_dir = 0;
                        int32_t unwrapped_device_id = my_chip_id - (devices_received / 2 + 1);
                        device_id = (unwrapped_device_id < 0) ? num_devices + unwrapped_device_id : unwrapped_device_id;
                    } else {  // currently backwards, next is forwards
                        curr_k_block_dir = 1;
                        uint32_t unwrapped_device_id = my_chip_id + (devices_received / 2);
                        device_id = unwrapped_device_id >= num_devices ? unwrapped_device_id - num_devices
                                                                       : unwrapped_device_id;
                    }
                } else {
                    if (curr_k_block_dir == 2) {  // currently self, check backwards first
                        next_forward = my_chip_id + 1;
                        next_backward = my_chip_id - 1;
                        if (next_backward < 0) {
                            curr_k_block_dir = 1;
                            device_id = next_forward;
                            next_forward++;
                        } else {
                            curr_k_block_dir = 0;
                            device_id = next_backward;
                            next_backward--;
                        }
                    } else if (curr_k_block_dir == 1) {  // currently forward, check backwards first
                        if (next_backward < 0) {
                            curr_k_block_dir = 1;
                            device_id = next_forward;
                            next_forward++;
                        } else {
                            curr_k_block_dir = 0;
                            device_id = next_backward;
                            next_backward--;
                        }
                    } else {  // currently backwards, check_forwards first
                        if (next_forward >= num_devices) {
                            curr_k_block_dir = 0;
                            device_id = next_backward;
                            next_backward--;
                        } else {
                            curr_k_block_dir = 1;
                            device_id = next_forward;
                            next_forward++;
                        }
                    }
                }
            }
            if (k_block_device_received[k_block_received] == k_block_device_expected[k_block_received]) {
                break;
            }
        }

        return k_block_received;
    }
};
