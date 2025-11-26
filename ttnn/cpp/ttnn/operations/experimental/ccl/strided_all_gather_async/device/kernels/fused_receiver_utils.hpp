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

uint32_t compute_device_chunk_stats(
    uint32_t input_tensor_Wt,
    uint32_t num_k_blocks,
    uint32_t k_block_tiles,
    uint32_t num_devices,
    uint32_t* k_block_device_received,
    uint32_t* k_block_device_expected,
    uint32_t* device_k_block_counts,
    uint32_t* device_k_block_start_ids) {
    uint32_t curr_device = 0;
    uint32_t curr_device_end = input_tensor_Wt - 1;

    uint32_t max_chunks_per_device = 0;
    uint32_t total_chunks = 0;
    for (uint32_t k_block_iter = 0; k_block_iter < num_k_blocks; k_block_iter++) {
        uint32_t curr_k_block_end = (k_block_iter + 1) * k_block_tiles - 1;
        if (curr_k_block_end < curr_device_end) {
            k_block_device_expected[k_block_iter]++;
            device_k_block_counts[curr_device]++;
            total_chunks++;
        } else if (curr_k_block_end == curr_device_end) {
            k_block_device_expected[k_block_iter]++;
            device_k_block_counts[curr_device]++;
            total_chunks++;
            curr_device++;
            curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
            if (curr_device < num_devices) {
                device_k_block_start_ids[curr_device] = k_block_iter + 1;
            }
        } else if (curr_k_block_end > curr_device_end) {
            k_block_device_expected[k_block_iter]++;
            device_k_block_counts[curr_device]++;
            total_chunks++;
            if (curr_device + 1 < num_devices) {
                k_block_device_expected[k_block_iter]++;
                device_k_block_counts[curr_device + 1]++;
                device_k_block_start_ids[curr_device + 1] = k_block_iter;
                total_chunks++;
            }
            curr_device++;
            curr_device_end = (curr_device + 1) * input_tensor_Wt - 1;
        }
    }

    return total_chunks;
}

struct MinimalMatmulOpReceiver {
    bool wait_for_op_signal = false;
    uint32_t num_devices = 0;
    uint32_t my_chip_id = 0;
    uint32_t input_tensor_Wt = 0;
    uint32_t num_k_blocks = 0;
    std::array<volatile tt_l1_ptr uint32_t*, 3> signal_op_semaphore_addr_ptrs = {};  // backward, forward, self
    std::array<uint32_t, 3> sem_targets = {};
    ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring;
    std::pair<int32_t, uint32_t>* chunk_to_k_block_map = nullptr;
    uint32_t* forward_map = nullptr;
    uint32_t global_chunk_id = 0;

    MinimalMatmulOpReceiver() {}

    MinimalMatmulOpReceiver(
        bool wait_for_op_signal,
        uint32_t& rt_args_idx,
        uint32_t* k_block_device_expected,
        uint32_t* k_block_device_received,
        uint32_t* device_k_block_counts,
        uint32_t* device_k_block_start_ids,
        std::pair<int32_t, uint32_t>* chunk_to_k_block_map,
        uint32_t* forward_map) :
        wait_for_op_signal(wait_for_op_signal), chunk_to_k_block_map(chunk_to_k_block_map), forward_map(forward_map) {
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

        uint32_t curr_k_block_dir = 2;  // start with self
        uint32_t device_id = my_chip_id;
        uint32_t device_chunk_id = 0;
        global_chunk_id = 0;
        uint32_t devices_received = 0;
        uint32_t next_forward = 0;
        int32_t next_backward = 0;
        for (uint32_t k_block_iter = 0; k_block_iter < num_k_blocks; k_block_iter++) {
            uint32_t actual_k_block = get_next_k_block_iter(
                device_id,
                device_chunk_id,
                global_chunk_id,
                curr_k_block_dir,
                devices_received,
                next_forward,
                next_backward,
                k_block_device_received,
                k_block_device_expected,
                device_k_block_counts,
                device_k_block_start_ids,
                chunk_to_k_block_map);
            forward_map[k_block_iter] = actual_k_block;
        }
    }

    void reset() { global_chunk_id = 0; }

    uint32_t get_next_k_block_iter(
        uint32_t& device,
        uint32_t& chunk,
        uint32_t& global_chunk,
        uint32_t& direction,
        uint32_t& received,
        uint32_t& next_forward,
        int32_t& next_backward,
        uint32_t* k_block_device_received,
        uint32_t* k_block_device_expected,
        uint32_t* device_k_block_counts,
        uint32_t* device_k_block_start_ids,
        std::pair<int32_t, uint32_t>* chunk_to_k_block_map) {
        uint32_t k_block_received = 0;
        while (true) {
            k_block_received = device_k_block_start_ids[device] + chunk;
            k_block_device_received[k_block_received]++;
            chunk++;
            uint32_t orig_direction = direction;
            if (chunk >= device_k_block_counts[device]) {
                // Move to next device
                received++;
                chunk = 0;
                if (topology == ttnn::ccl::Topology::Ring) {
                    if (direction > 0) {  // currently self or forward, next is backwards
                        direction = 0;
                        int32_t unwrapped_device = my_chip_id - (received / 2 + 1);
                        device = (unwrapped_device < 0) ? num_devices + unwrapped_device : unwrapped_device;
                    } else {  // currently backwards, next is forwards
                        direction = 1;
                        uint32_t unwrapped_device = my_chip_id + (received / 2);
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
                chunk_to_k_block_map[global_chunk] = std::pair<int32_t, uint32_t>(k_block_received, orig_direction);
                global_chunk++;
                break;
            } else {
                chunk_to_k_block_map[global_chunk] = std::pair<int32_t, uint32_t>(-1, orig_direction);
                global_chunk++;
            }
        }
        return k_block_received;
    }

    uint32_t compute_actual_k_block_iter(bool is_first_n_block_iter, uint32_t k_block_iter, bool is_forward) {
        uint32_t k_block_received = 0;

        if (is_first_n_block_iter) {
            while (true) {
                if (wait_for_op_signal) {
                    uint32_t curr_k_block_dir = chunk_to_k_block_map[global_chunk_id].second;
                    volatile tt_l1_ptr uint32_t* semaphore = signal_op_semaphore_addr_ptrs[curr_k_block_dir];
                    uint32_t sem_target = sem_targets[curr_k_block_dir];
                    noc_semaphore_wait_min(semaphore, sem_target + 1);
                    sem_targets[curr_k_block_dir]++;
                }

                if (chunk_to_k_block_map[global_chunk_id].first >= 0) {
                    k_block_received = (uint32_t)chunk_to_k_block_map[global_chunk_id].first;
                    global_chunk_id++;
                    break;
                } else {
                    global_chunk_id++;
                }
            }
        } else {
            if (is_forward) {
                k_block_received = forward_map[k_block_iter];
            } else {
                k_block_received = forward_map[num_k_blocks - k_block_iter - 1];
            }
        }

        return k_block_received;
    }
};
