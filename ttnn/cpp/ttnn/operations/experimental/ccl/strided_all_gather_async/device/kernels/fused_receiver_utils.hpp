// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include <array>

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
    // Per-worker signaling: each of the N all-gather workers in a remote direction increments its own
    // semaphore, so a k-block is ready only once all N have signaled. self is a single semaphore
    // (aggregated by the writer-side worker barrier). N==1 reproduces the legacy [backward,forward,self].
    uint32_t num_ag_workers = 1;
    uint32_t* backward_sem_addrs = nullptr;  // [num_ag_workers] L1 semaphore addresses
    uint32_t* forward_sem_addrs = nullptr;   // [num_ag_workers] L1 semaphore addresses
    volatile tt_l1_ptr uint32_t* self_sem_ptr = nullptr;
    std::array<uint32_t, 3> sem_targets = {};  // indexed by direction: [backward, forward, self]
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
        uint32_t* forward_k_block_schedule,
        uint32_t* backward_sem_addrs,
        uint32_t* forward_sem_addrs) :
        wait_for_op_signal(wait_for_op_signal),
        backward_sem_addrs(backward_sem_addrs),
        forward_sem_addrs(forward_sem_addrs),
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
        num_ag_workers = get_arg_val<uint32_t>(rt_args_idx++);

        if (this->wait_for_op_signal) {
            // Semaphore ids arrive as [backward_0..N-1, forward_0..N-1, self].
            for (uint32_t w = 0; w < num_ag_workers; w++) {
                this->backward_sem_addrs[w] = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
            }
            for (uint32_t w = 0; w < num_ag_workers; w++) {
                this->forward_sem_addrs[w] = get_semaphore(get_arg_val<uint32_t>(rt_args_idx++));
            }
            this->self_sem_ptr =
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
                // On an even ring the diametric device's slice is split-forwarded: its second half
                // is relayed on the forward link, so await that half on the forward signal semaphore.
                if (topology == ttnn::ccl::Topology::Ring && num_devices % 2 == 0 && num_devices > 2) {
                    uint32_t diametric_device = (my_chip_id + num_devices / 2) % num_devices;
                    if (device_id == diametric_device && device_k_block_counts[device_id] >= 2 &&
                        device_chunk_id >= device_k_block_counts[device_id] / 2) {
                        curr_k_block_dir = 1;  // forward
                    }
                }
                if (wait_for_op_signal && !(read_local_slice_from_input && (curr_k_block_dir == 2))) {
                    uint32_t sem_target = sem_targets[curr_k_block_dir];
                    if (curr_k_block_dir == 2) {
                        // self: single semaphore (writer-side worker barrier already aggregated all workers)
                        noc_semaphore_wait_min(self_sem_ptr, sem_target + 1);
                    } else {
                        // remote direction: a k-block is ready only once all N workers have signaled their
                        // own semaphore (per-worker counters are drift-safe across independent fabric links)
                        uint32_t* addrs = (curr_k_block_dir == 0) ? backward_sem_addrs : forward_sem_addrs;
                        for (uint32_t w = 0; w < num_ag_workers; w++) {
                            noc_semaphore_wait_min(
                                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addrs[w]), sem_target + 1);
                        }
                    }
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
