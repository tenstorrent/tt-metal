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

void compute_device_chunk_stats(
    uint32_t input_tensor_Wt,
    uint32_t num_k_blocks,
    uint32_t k_block_tiles,
    uint32_t num_devices,
    uint8_t* k_block_device_received,
    uint8_t* k_block_device_expected,
    uint32_t* device_k_block_counts,
    uint32_t* device_k_block_start_ids) {
    uint32_t curr_device = 0;
    uint32_t curr_device_end = input_tensor_Wt - 1;

    uint32_t max_chunks_per_device = 0;
    for (uint32_t k_block_iter = 0; k_block_iter < num_k_blocks; k_block_iter++) {
        uint32_t curr_k_block_end = (k_block_iter + 1) * k_block_tiles - 1;
        if (curr_k_block_end < curr_device_end) {
            k_block_device_expected[k_block_iter]++;
            device_k_block_counts[curr_device]++;
        } else if (curr_k_block_end == curr_device_end) {
            k_block_device_expected[k_block_iter]++;
            device_k_block_counts[curr_device]++;
            curr_device++;
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
            curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
        }
    }
}

struct MinimalMatmulOpReceiver {
    bool wait_for_op_signal = false;
    uint32_t num_devices = 0;
    uint32_t my_chip_id = 0;
    uint32_t input_tensor_Wt = 0;
    uint32_t num_k_blocks = 0;
    uint32_t local_k_start = 0;
    uint32_t local_k_end = 0;
    std::array<volatile tt_l1_ptr uint32_t*, 3> signal_op_semaphore_addr_ptrs = {};  // backward, forward, self
    std::array<uint32_t, 3> sem_targets = {};
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;
    bool read_local_slice_from_input;

    uint8_t* k_block_device_expected;
    uint8_t* k_block_device_received;
    uint32_t* device_k_block_counts;
    uint32_t* device_k_block_start_ids;
    uint32_t* forward_k_block_schedule;

    uint8_t curr_k_block_dir = 2;  // start with self
    uint8_t device_id = my_chip_id;
    uint32_t device_chunk_id = 0;
    uint8_t devices_received = 0;
    uint8_t next_forward = 0;
    int8_t next_backward = 0;

    MinimalMatmulOpReceiver() {}

    MinimalMatmulOpReceiver(
        bool wait_for_op_signal,
        uint32_t& rt_args_idx,
        uint8_t* k_block_device_expected,
        uint8_t* k_block_device_received,
        uint32_t* device_k_block_counts,
        uint32_t* device_k_block_start_ids,
        uint32_t* forward_k_block_schedule) :
        wait_for_op_signal(wait_for_op_signal),
        k_block_device_expected(k_block_device_expected),
        k_block_device_received(k_block_device_received),
        device_k_block_counts(device_k_block_counts),
        device_k_block_start_ids(device_k_block_start_ids),
        forward_k_block_schedule(forward_k_block_schedule) {
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
        read_local_slice_from_input = (bool)get_arg_val<uint32_t>(rt_args_idx++);
        local_k_start = get_arg_val<uint32_t>(rt_args_idx++);
        local_k_end = get_arg_val<uint32_t>(rt_args_idx++);

        if (this->wait_for_op_signal) {
            this->signal_op_semaphore_addr_ptrs[0] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
            this->signal_op_semaphore_addr_ptrs[1] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
            this->signal_op_semaphore_addr_ptrs[2] =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(get_arg_val<uint32_t>(rt_args_idx++)));
        }

        compute_device_chunk_stats(
            input_tensor_Wt,
            num_k_blocks,
            k_block_tiles,
            num_devices,
            k_block_device_received,
            k_block_device_expected,
            device_k_block_counts,
            device_k_block_start_ids);
    }

    void reset() {
        curr_k_block_dir = 2;  // start with self
        device_id = my_chip_id;
        device_chunk_id = 0;
        devices_received = 0;
        next_forward = 0;
        next_backward = 0;

        for (uint32_t k = 0; k < num_k_blocks; k++) {
            k_block_device_received[k] = 0;
        }
    }

    int32_t process_chunk(
        uint8_t& device,
        uint32_t& chunk,
        uint8_t& direction,
        uint8_t& devices_received,
        uint8_t& next_forward,
        int8_t& next_backward) {
        uint32_t k_block_received = device_k_block_start_ids[device] + chunk;
        k_block_device_received[k_block_received]++;
        chunk++;
        if (chunk >= device_k_block_counts[device]) {
            // Move to next device
            devices_received++;
            chunk = 0;
            if (topology == ttnn::ccl::Topology::Ring) {
                if (direction > 0) {  // currently self or forward, next is backwards
                    direction = 0;
                    int32_t unwrapped_device = my_chip_id - (devices_received / 2 + 1);
                    device = (unwrapped_device < 0) ? num_devices + unwrapped_device : unwrapped_device;
                } else {  // currently backwards, next is forwards
                    direction = 1;
                    uint32_t unwrapped_device = my_chip_id + (devices_received / 2);
                    device = unwrapped_device >= num_devices ? unwrapped_device - num_devices : unwrapped_device;
                }
            } else {
                if (direction == 2) {  // currently self, check backwards first
                    next_forward = my_chip_id + 1;
                    next_backward = my_chip_id - 1;
                    if (next_backward < 0) {
                        direction = 1;
                        device = next_forward;
                        next_forward++;
                    } else {
                        direction = 0;
                        device = next_backward;
                        next_backward--;
                    }
                } else if (direction == 1) {  // currently forward, check backwards first
                    if (next_backward < 0) {
                        direction = 1;
                        device = next_forward;
                        next_forward++;
                    } else {
                        direction = 0;
                        device = next_backward;
                        next_backward--;
                    }
                } else {  // currently backwards, check_forwards first
                    if (next_forward >= num_devices) {
                        direction = 0;
                        device = next_backward;
                        next_backward--;
                    } else {
                        direction = 1;
                        device = next_forward;
                        next_forward++;
                    }
                }
            }
        }
        if (k_block_device_received[k_block_received] == k_block_device_expected[k_block_received]) {
            return k_block_received;
        } else {
            return -1;
        }
    }

    uint32_t compute_actual_k_block_iter(bool is_first_n_block_iter, uint32_t k_block_iter, bool is_forward) {
        uint32_t k_block_received = 0;

        if (is_first_n_block_iter) {
            while (true) {
                if (wait_for_op_signal && !(read_local_slice_from_input && (curr_k_block_dir == 2))) {
                    volatile tt_l1_ptr uint32_t* semaphore = signal_op_semaphore_addr_ptrs[curr_k_block_dir];
                    uint32_t sem_target = sem_targets[curr_k_block_dir];
                    noc_semaphore_wait_min(semaphore, sem_target + 1);
                    sem_targets[curr_k_block_dir]++;
                }
                int32_t k_block = process_chunk(
                    device_id, device_chunk_id, curr_k_block_dir, devices_received, next_forward, next_backward);
                if (k_block >= 0) {
                    k_block_received = (uint32_t)k_block;
                    forward_k_block_schedule[k_block_iter] = k_block_received;
                    break;
                }
            }
        } else {
            if (is_forward) {
                k_block_received = forward_k_block_schedule[k_block_iter];
            } else {
                k_block_received = forward_k_block_schedule[num_k_blocks - k_block_iter - 1];
            }
        }

        return k_block_received;
    }
};
