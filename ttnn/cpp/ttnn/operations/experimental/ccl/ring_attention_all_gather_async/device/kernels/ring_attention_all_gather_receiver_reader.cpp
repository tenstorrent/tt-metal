// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType intermediate_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(2);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(3);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(6);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr bool direction = get_compile_time_arg_val(8);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(9);
constexpr uint32_t num_inputs = get_compile_time_arg_val(10);

constexpr uint32_t N_DRAM_BANKS = 12;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    address_t intermediate_buffer_addrs[num_inputs];
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        address_t intermediate_buffer_addr = get_arg_val<address_t>(arg_idx++);
        intermediate_buffer_addrs[input_idx] = intermediate_buffer_addr;
    }
    constexpr bool intermediate_is_dram = intermediate_buffer_type == tt::tt_metal::BufferType::DRAM;
    InterleavedAddrGenFast<intermediate_is_dram> intermediate_tensor_addrgens[num_inputs];
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        auto intermediate_tensor_addrgen = InterleavedAddrGenFast<intermediate_is_dram>{
            .bank_base_address = intermediate_buffer_addrs[input_idx],
            .page_size = input_tensor_page_size,
            .data_format = get_dataformat(cb_intermediate_id)};
        intermediate_tensor_addrgens[input_idx] = intermediate_tensor_addrgen;
    }

    uint32_t slices_received = 0;
    uint32_t slices_expected;
    if (topology == Topology::Linear) {
        if (direction == 1) {
            slices_expected = num_targets_forward_direction;
        } else {
            slices_expected = num_targets_backward_direction;
        }
    } else if (topology == Topology::Ring) {
        if (direction == 1) {
            slices_expected = num_targets_backward_direction;
        } else {
            slices_expected = num_targets_forward_direction;
        }
    }

    const uint32_t payload_size_bytes = input_tensor_page_size * contig_pages_advanced;

    volatile tt_l1_ptr uint32_t* out_ready_sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem);
    while (slices_received < slices_expected) {
        // Do i expect more?
        // If direction == backward, do I expect more from the left?
        // In the linear case, I expect num_targets_backward_direction slices from the left
        // In the ring case, I expect num_targets_backward_direction slices from the right
        // If direction == forward, do i expect more from the right?
        // In the linear case, I expect num_targets_forward_direction slices from the right
        // In the ring case, I expect num_targets_forward_direction slices from the right (keep in mind this differs for
        // odd/even chips)
        noc_semaphore_wait_min(out_ready_sem_addr, slices_received + 1);
        // Got it
        slices_received++;

        int sender_chip_id;
        uint32_t actual_sender_chip_id;
        if (direction == 1) {
            sender_chip_id = my_chip_id + slices_received;
            actual_sender_chip_id = (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
        } else {
            sender_chip_id = my_chip_id - slices_received;
            actual_sender_chip_id = (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
        }

        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            uint32_t row_offset = 0;
            uint32_t intermediate_tile_id_start = 0;
            if (gather_dim == 3) {
                intermediate_tile_id_start = actual_sender_chip_id * input_tensor_Wt;
            } else {
                intermediate_tile_id_start = actual_sender_chip_id * input_tensor_Ht * input_tensor_Wt;
            }
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                for (uint32_t row_idx = 0; row_idx < input_tensor_Ht; row_idx++) {
                    for (uint32_t col_idx = 0; col_idx < input_tensor_Wt; col_idx += packet_size_in_pages) {
                        cb_reserve_back(cb_intermediate_id, packet_size_in_pages);
                        size_t l1_write_addr = get_write_ptr(cb_intermediate_id);
                        for (uint32_t j = 0; j < packet_size_in_pages; j += contig_pages_advanced) {
                            noc_async_read_tile(
                                intermediate_tile_id_start + row_offset + col_idx + j,
                                intermediate_tensor_addrgens[input_idx],
                                l1_write_addr);
                            l1_write_addr += payload_size_bytes;
                        }
                        noc_async_read_barrier();
                        cb_push_back(cb_intermediate_id, packet_size_in_pages);
                    }
                    row_offset += output_tensor_Wt;
                }
                row_offset = 0;
                intermediate_tile_id_start += output_tensor_Wt * output_tensor_Ht;
            }
        }
    }

    noc_semaphore_set(out_ready_sem_addr, 0);
}
