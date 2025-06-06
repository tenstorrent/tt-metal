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

constexpr uint32_t N_DRAM_BANKS = 12;
constexpr uint32_t my_chip_id_x = my_chip_id % N_DRAM_BANKS;
constexpr uint32_t my_chip_id_y = my_chip_id / N_DRAM_BANKS;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t intermediate_buffer_addr = get_arg_val<address_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    constexpr bool intermediate_tensor_is_dram = intermediate_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<intermediate_tensor_is_dram>{
        .bank_base_address = intermediate_buffer_addr,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_intermediate_id)};

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
    uint32_t actual_sender_chip_id_x = my_chip_id_x;
    uint32_t actual_sender_chip_id_y = my_chip_id_y;
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
        if (direction == 1) {
            actual_sender_chip_id_x = (actual_sender_chip_id_x == ring_size - 1) ? 0 : actual_sender_chip_id_x + 1;
        } else {
            actual_sender_chip_id_x = (actual_sender_chip_id_x == 0) ? ring_size - 1 : actual_sender_chip_id_x - 1;
        }

        uint32_t tiles_read = 0;
        uint32_t tiles_to_read = slice_num_pages;

        uint32_t packet_id = 0;
        uint32_t intermediate_packet_id_x = actual_sender_chip_id_x;
        uint32_t intermediate_packet_id_y = actual_sender_chip_id_y;
        while (tiles_read < tiles_to_read) {
            uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
            cb_reserve_back(cb_intermediate_id, num_pages_to_read);
            size_t l1_write_addr = get_write_ptr(cb_intermediate_id);
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
            cb_push_back(cb_intermediate_id, num_pages_to_read);
        }
    }

    noc_semaphore_set(out_ready_sem_addr, 0);
}
