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
constexpr BufferType input_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr BufferType intermediate_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb_forward_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_backward_id = get_compile_time_arg_val(4);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(9));
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(10);

constexpr uint32_t N_DRAM_BANKS = 12;
constexpr uint32_t my_chip_id_x = my_chip_id % N_DRAM_BANKS;
constexpr uint32_t my_chip_id_y = my_chip_id / N_DRAM_BANKS;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    // Load the input tensor spec
    address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t intermediate_tensor_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_forward = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_backward = get_arg_val<uint32_t>(arg_idx++);
    size_t signal_receiver_sem_forward = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    size_t signal_receiver_sem_backward = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint8_t signal_receiver_sem_forward_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t signal_receiver_sem_forward_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t signal_receiver_sem_backward_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t signal_receiver_sem_backward_noc0_y = get_arg_val<uint32_t>(arg_idx++);

    // Push out our local slice
    constexpr bool input_tensor_is_dram = input_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto input_tensor_addrgen = InterleavedAddrGenFast<input_tensor_is_dram>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_forward_id)};
    uint32_t tiles_read = 0;
    uint32_t tiles_to_read = slice_num_pages;
    while (tiles_read < tiles_to_read) {
        uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
        cb_reserve_back(cb_forward_id, num_pages_to_read);
        const uint32_t l1_write_addr_base = get_write_ptr(cb_forward_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        for (uint32_t j = 0; j < num_pages_to_read; j++) {
            noc_async_read_tile(tiles_read, input_tensor_addrgen, l1_write_addr);
            l1_write_addr += input_tensor_page_size;
            tiles_read++;
        }

        noc_async_read_barrier();
        cb_push_back(cb_forward_id, num_pages_to_read);
    }

    constexpr bool intermediate_tensor_is_dram = intermediate_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<intermediate_tensor_is_dram>{
        .bank_base_address = intermediate_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_forward_id)};
    uint32_t forward_slices_received = 0;
    uint32_t backward_slices_received = 0;
    uint32_t forward_slices_expected, backward_slices_expected;
    uint32_t forward_writes_expected, backward_writes_expected;
    if (topology == Topology::Linear) {
        forward_slices_expected = num_targets_forward_direction;
        backward_slices_expected = num_targets_backward_direction;
    } else if (topology == Topology::Ring) {
        forward_slices_expected = num_targets_backward_direction;
        backward_slices_expected = num_targets_forward_direction;
        forward_writes_expected = num_targets_forward_direction - 1;
        backward_writes_expected = num_targets_backward_direction - 1;
    }

    uint64_t forward_receiver_semaphore_addr = get_noc_addr(
        signal_receiver_sem_forward_noc0_x, signal_receiver_sem_forward_noc0_y, signal_receiver_sem_forward);
    uint64_t backward_receiver_semaphore_addr = get_noc_addr(
        signal_receiver_sem_backward_noc0_x, signal_receiver_sem_backward_noc0_y, signal_receiver_sem_backward);

    const uint32_t payload_size_bytes = input_tensor_page_size * contig_pages_advanced;
    uint32_t actual_backward_chip_id_x = my_chip_id_x;
    uint32_t actual_backward_chip_id_y = my_chip_id_y;
    uint32_t actual_forward_chip_id_x = my_chip_id_x;
    uint32_t actual_forward_chip_id_y = my_chip_id_y;
    while (forward_slices_received < forward_slices_expected || backward_slices_received < backward_slices_expected) {
        // Do i expect more from the left?
        // In the linear case, I expect num_targets_backward_direction slices from the left
        // In the ring case, I expect num_targets_backward_direction slices from the right, (keep in mind this differs
        // for odd/even chips)
        if (backward_slices_received < backward_slices_expected) {
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_backward) <= backward_slices_received);
            noc_semaphore_inc(backward_receiver_semaphore_addr, 1);
            // Got it
            backward_slices_received++;
            int backward_chip_id = my_chip_id - backward_slices_received;
            uint32_t actual_backward_chip_id = (backward_chip_id < 0) ? ring_size + backward_chip_id : backward_chip_id;
            actual_backward_chip_id_x =
                (actual_backward_chip_id_x == 0) ? ring_size - 1 : actual_backward_chip_id_x - 1;

            // Should I forward what I got from the left to my right?
            // In the linear case, if I have any targets to my right, always forward
            // In the ring case, if I have received on the left less than my targets on the right, forward
            if ((topology == Topology::Linear && num_targets_forward_direction > 0) ||
                (topology == Topology::Ring && (backward_slices_received < (forward_writes_expected + 1)))) {
                // read the next backward slice out of memory, and put it in CB
                uint32_t output_tile_id_start = actual_backward_chip_id * input_tensor_Wt;
                tiles_read = 0;
                tiles_to_read = slice_num_pages;

                uint32_t packet_id = 0;
                uint32_t intermediate_packet_id_x = actual_backward_chip_id_x;
                uint32_t intermediate_packet_id_y = actual_backward_chip_id_y;
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                    cb_reserve_back(cb_forward_id, num_pages_to_read);
                    size_t l1_write_addr = get_write_ptr(cb_forward_id);
                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        uint32_t intermediate_packet_first_tile_id =
                            intermediate_packet_id_x + contig_pages_advanced * N_DRAM_BANKS * intermediate_packet_id_y;
                        uint64_t packet_addr = get_noc_addr(
                            intermediate_packet_first_tile_id, intermediate_tensor_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                        noc_async_read(packet_addr, l1_write_addr, payload_size_bytes);
                        l1_write_addr += payload_size_bytes;
                        tiles_read += contig_pages_advanced;
                        packet_id++;

                        intermediate_packet_id_x += ring_size;
                        if (intermediate_packet_id_x >= N_DRAM_BANKS) {
                            intermediate_packet_id_x -= N_DRAM_BANKS;
                            intermediate_packet_id_y++;
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_forward_id, num_pages_to_read);
                }
            }
        }

        // Do i expect more from the right?
        // In the linear case, I expect num_targets_forward_direction slices from the right
        // In the ring case, I expect num_targets_forward_direction slices from the right (keep in mind this differs for
        // odd/even chips)
        if (forward_slices_received < forward_slices_expected) {
            while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_forward) <= forward_slices_received);
            noc_semaphore_inc(forward_receiver_semaphore_addr, 1);
            // Got it
            forward_slices_received++;
            uint32_t forward_chip_id = my_chip_id + forward_slices_received;
            uint32_t actual_forward_chip_id =
                (forward_chip_id >= ring_size) ? forward_chip_id - ring_size : forward_chip_id;
            actual_forward_chip_id_x = (actual_forward_chip_id_x == ring_size - 1) ? 0 : actual_forward_chip_id_x + 1;
            // Should I forward what I got from the right to my left?
            // In the linear case, if I have any targets to my left, always forward
            // In the ring case, if I have received on the right less than my targets on the left, forward
            if ((topology == Topology::Linear && num_targets_backward_direction > 0) ||
                (topology == Topology::Ring && (forward_slices_received < (backward_writes_expected + 1)))) {
                // read the next forward slice out of memory, and put it in CB
                uint32_t output_tile_id_start = actual_forward_chip_id * input_tensor_Wt;
                tiles_read = 0;
                tiles_to_read = slice_num_pages;

                uint32_t packet_id = 0;
                uint32_t intermediate_packet_id_x = actual_forward_chip_id_x;
                uint32_t intermediate_packet_id_y = actual_forward_chip_id_y;
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
                    cb_reserve_back(cb_backward_id, num_pages_to_read);
                    size_t l1_write_addr = get_write_ptr(cb_backward_id);
                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        // uint32_t intermediate_packet_id = actual_forward_chip_id + packet_id * ring_size;
                        uint32_t intermediate_packet_first_tile_id =
                            intermediate_packet_id_x + contig_pages_advanced * N_DRAM_BANKS * intermediate_packet_id_y;
                        uint64_t packet_addr = get_noc_addr(
                            intermediate_packet_first_tile_id, intermediate_tensor_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                        noc_async_read(packet_addr, l1_write_addr, payload_size_bytes);
                        l1_write_addr += payload_size_bytes;
                        tiles_read += contig_pages_advanced;
                        packet_id++;

                        intermediate_packet_id_x += ring_size;
                        if (intermediate_packet_id_x >= N_DRAM_BANKS) {
                            intermediate_packet_id_x -= N_DRAM_BANKS;
                            intermediate_packet_id_y++;
                        }
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_backward_id, num_pages_to_read);
                }
            }
        }
    }

    const uint64_t dest_noc_addr_forward = get_noc_addr(my_x[0], my_y[0], out_ready_sem_forward);
    noc_inline_dw_write(dest_noc_addr_forward, 0);
    const uint64_t dest_noc_addr_backward = get_noc_addr(my_x[0], my_y[0], out_ready_sem_backward);
    noc_inline_dw_write(dest_noc_addr_backward, 0);
}
