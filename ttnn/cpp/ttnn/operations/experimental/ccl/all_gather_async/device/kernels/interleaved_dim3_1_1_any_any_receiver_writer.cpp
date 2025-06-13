// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType output_tensor_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(2);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(3);
constexpr uint32_t output_tensor_page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(6);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr bool direction = get_compile_time_arg_val(8);
constexpr bool fuse_op = get_compile_time_arg_val(9);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(10);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);

    OpSignaler op_signaler;
    if constexpr (fuse_op) {
        op_signaler = OpSignaler(arg_idx);
    }

    // interleaved addrgen
    constexpr bool output_is_dram = output_tensor_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto output_tensor_addrgen = InterleavedAddrGenFast<output_is_dram>{
        .bank_base_address = output_tensor_address,
        .page_size = output_tensor_page_size,
        .data_format = get_dataformat(cb_intermediate_id)};

    uint32_t forward_writes = 0;
    uint32_t backward_writes = 0;

    uint32_t forward_writes_expected, backward_writes_expected;
    if (topology == Topology::Linear) {
        forward_writes_expected = num_targets_backward_direction;
        backward_writes_expected = num_targets_forward_direction;
    } else if (topology == Topology::Ring) {
        forward_writes_expected = num_targets_forward_direction - 1;
        backward_writes_expected = num_targets_backward_direction - 1;
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

    uint32_t actual_sender_chip_id = my_chip_id;
    while (slices_received < slices_expected) {
        slices_received++;
        if (direction == 1) {
            actual_sender_chip_id = (actual_sender_chip_id == ring_size - 1) ? 0 : actual_sender_chip_id + 1;
        } else {
            actual_sender_chip_id = (actual_sender_chip_id == 0) ? ring_size - 1 : actual_sender_chip_id - 1;
        }

        uint32_t pages_read_in_row = 0;
        uint32_t row_offset = 0;
        uint32_t tiles_read = 0;
        uint32_t tile_id_start = actual_sender_chip_id * input_tensor_Wt;
        uint32_t tiles_to_read = slice_num_pages;
        while (tiles_read < tiles_to_read) {
            uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, packet_size_in_pages);
            uint32_t payload_size_bytes = contig_pages_advanced * output_tensor_page_size;
            cb_wait_front(cb_intermediate_id, num_pages_to_read);
            size_t l1_read_addr = get_read_ptr(cb_intermediate_id);
            for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                uint32_t first_tile_id = tile_id_start + row_offset + pages_read_in_row;
                noc_async_write_tile(first_tile_id, output_tensor_addrgen, l1_read_addr);
                pages_read_in_row += 1;
                if (pages_read_in_row >= input_tensor_Wt) {
                    row_offset += output_tensor_Wt;
                    pages_read_in_row = 0;
                }

                uint32_t second_tile_id = tile_id_start + row_offset + pages_read_in_row;
                noc_async_write_tile(second_tile_id, output_tensor_addrgen, l1_read_addr + output_tensor_page_size);
                pages_read_in_row += 1;
                if (pages_read_in_row >= input_tensor_Wt) {
                    row_offset += output_tensor_Wt;
                    pages_read_in_row = 0;
                }

                l1_read_addr += payload_size_bytes;
                tiles_read += contig_pages_advanced;
            }
            cb_pop_front(cb_intermediate_id, num_pages_to_read);
        }
        if (fuse_op) {
            // Signal matmul to go
            op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
        }
    }

    noc_async_write_barrier();
}
