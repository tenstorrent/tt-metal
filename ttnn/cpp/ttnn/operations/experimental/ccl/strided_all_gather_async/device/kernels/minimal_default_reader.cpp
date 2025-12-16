// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "ttnn/operations/experimental/ccl/strided_all_gather_async/device/kernels/strided_all_gather_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(1);
constexpr uint32_t max_tiles_per_packet = get_compile_time_arg_val(2);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(5);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(6));
constexpr bool direction = get_compile_time_arg_val(7);  // 1 is forward, 0 is backward
constexpr bool fuse_op = get_compile_time_arg_val(8);
constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(9);
constexpr uint32_t ag_worker_cores = get_compile_time_arg_val(10);
constexpr uint32_t ag_worker_id = get_compile_time_arg_val(11);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_batches = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_worker_tile_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mm_block_wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mm_block_ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t mm_cores_y = get_arg_val<uint32_t>(arg_idx++);

    uint32_t device_k_block_counts[ring_size];
    uint32_t device_max_chunks = get_arg_val<uint32_t>(arg_idx++);
    uint32_t device_chunk_widths[ring_size][device_max_chunks];
    for (uint32_t d = 0; d < ring_size; d++) {
        device_k_block_counts[d] = get_arg_val<uint32_t>(arg_idx++);
        uint32_t device_chunk_count = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t c = 0; c < device_chunk_count; c++) {
            device_chunk_widths[d][c] = get_arg_val<uint32_t>(arg_idx++);
        }
    }

    constexpr uint32_t ct_idx = 12;

    constexpr auto input_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = input_tensor_args.num_compile_time_args();
    const auto input_tensor_addrgen = TensorAccessor(input_tensor_args, input_tensor_address, input_tensor_page_size);

    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    const auto output_tensor_addrgen =
        TensorAccessor(output_tensor_args, output_tensor_address, input_tensor_page_size);

    OpSignaler op_signaler;
    if constexpr (fuse_op) {
        op_signaler = OpSignaler(arg_idx);
    }

    uint32_t slices_expected = 0;
    uint32_t writes_expected = 0;
    if (topology == Topology::Linear) {
        if (direction == 1) {
            slices_expected = num_targets_forward_direction;
            writes_expected = num_targets_backward_direction ? num_targets_forward_direction : 0;
        } else {
            slices_expected = num_targets_backward_direction;
            writes_expected = num_targets_forward_direction ? num_targets_backward_direction : 0;
        }
    } else if (topology == Topology::Ring) {
        if (direction == 1) {
            slices_expected = num_targets_backward_direction;
        } else {
            slices_expected = num_targets_forward_direction;
        }
        writes_expected = slices_expected - 1;
    }

    uint32_t batch_input_tile_offset = input_worker_tile_offset;
    uint32_t global_tile_index = 0;
    uint32_t tiles_per_batch = input_tensor_Wt * input_tensor_Ht;
    uint32_t sem_target = 0;

    uint32_t padded_M_tiles = round_up(input_tensor_Ht, mm_cores_y);
    uint32_t M_tiles_per_core = padded_M_tiles / mm_cores_y;
    uint32_t M_blocks_per_core = div_up(M_tiles_per_core, mm_block_ht);

    for (uint32_t b_idx = 0; b_idx < num_batches; b_idx++) {
        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            // Send out local
            uint32_t input_chunk_start_tile = global_tile_index;
            for (uint32_t chunk_idx = 0; chunk_idx < device_k_block_counts[my_chip_id]; chunk_idx++) {
                uint32_t actual_chunk_w = device_chunk_widths[my_chip_id][chunk_idx];
                uint32_t actual_chunk_h = next_mm_aligned_chunk_height(
                    input_chunk_start_tile, M_tiles_per_core, input_tensor_Wt, mm_block_ht);
                uint32_t tiles_in_current_chunk = actual_chunk_w * actual_chunk_h * mm_cores_y;
                read_chunk(
                    input_chunk_start_tile,
                    batch_input_tile_offset,
                    cb_output_id,
                    tiles_in_current_chunk,
                    actual_chunk_w,
                    actual_chunk_h,
                    padded_M_tiles / mm_cores_y,
                    max_tiles_per_packet,
                    ag_worker_id,
                    ag_worker_cores,
                    input_tensor_addrgen,
                    input_tensor_page_size,
                    output_tensor_addrgen,
                    input_tensor_Wt,
                    input_tensor_Ht,
                    output_tensor_Wt,
                    my_chip_id,
                    false);
            }

            // Receive remote chunks
            uint32_t slices_received = 0;
            uint32_t last_input_chunk_start_tile = input_chunk_start_tile;
            while (slices_received < slices_expected) {
                uint32_t actual_sender_chip_id = get_sender_id(direction, my_chip_id, slices_received, ring_size);

                input_chunk_start_tile = global_tile_index;
                for (uint32_t chunk_idx = 0; chunk_idx < device_k_block_counts[actual_sender_chip_id]; chunk_idx++) {
                    // Receive the next chunk of data
                    noc_semaphore_wait_min(
                        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                    sem_target++;

                    if ((topology == Topology::Linear && writes_expected > 0) ||
                        (topology == Topology::Ring && ((slices_received + 1) < (writes_expected + 1)))) {
                        uint32_t actual_chunk_w = device_chunk_widths[actual_sender_chip_id][chunk_idx];
                        uint32_t actual_chunk_h = next_mm_aligned_chunk_height(
                            input_chunk_start_tile, M_tiles_per_core, input_tensor_Wt, mm_block_ht);
                        uint32_t tiles_in_current_chunk = actual_chunk_w * actual_chunk_h * mm_cores_y;
                        read_chunk(
                            input_chunk_start_tile,
                            batch_input_tile_offset,
                            cb_output_id,
                            tiles_in_current_chunk,
                            actual_chunk_w,
                            actual_chunk_h,
                            padded_M_tiles / mm_cores_y,
                            max_tiles_per_packet,
                            ag_worker_id,
                            ag_worker_cores,
                            input_tensor_addrgen,
                            input_tensor_page_size,
                            output_tensor_addrgen,
                            input_tensor_Wt,
                            input_tensor_Ht,
                            output_tensor_Wt,
                            actual_sender_chip_id,
                            true);
                        last_input_chunk_start_tile = input_chunk_start_tile;
                    }
                    if constexpr (fuse_op) {
                        // Signal matmul to go
                        op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
                    }
                }
                slices_received++;
            }
            global_tile_index = last_input_chunk_start_tile;
        }
        batch_input_tile_offset += tiles_per_batch;
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}
