// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType input_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr BufferType intermediate_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(2));
constexpr uint32_t cb_input_id = get_compile_time_arg_val(3);
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(4);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(5);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(6);
constexpr uint32_t input_tensor_page_size = get_compile_time_arg_val(7);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(8);
constexpr uint32_t batch_slice_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t ring_size = get_compile_time_arg_val(10);
constexpr uint32_t num_batches = get_compile_time_arg_val(11);
constexpr uint32_t fuse_op = get_compile_time_arg_val(12);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(13);

constexpr uint32_t stride_Wt = input_tensor_Wt;
constexpr uint32_t slice_Wt = input_tensor_Wt / ring_size;

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
    size_t out_ready_sem_fwd = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem_bwd = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_links = get_arg_val<uint32_t>(arg_idx++);

    uint32_t pages_read_in_row_0 = get_arg_val<uint32_t>(arg_idx++);
    uint32_t row_offset_0 = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tiles_read_0 = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tiles_to_read_0 = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_packet_offset_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t intermediate_packet_offset_y = get_arg_val<uint32_t>(arg_idx++);

    ReduceScatterOpReceiver matmul_receiver;
    if constexpr (fuse_op) {
        matmul_receiver = ReduceScatterOpReceiver(arg_idx);
    }

    constexpr uint32_t batch_num_pages = batch_slice_num_pages * ring_size;

    constexpr bool input_tensor_is_dram = input_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto input_tensor_addrgen = InterleavedAddrGenFast<input_tensor_is_dram>{
        .bank_base_address = input_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_input_id)};
    constexpr bool intermediate_tensor_is_dram = intermediate_buffer_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<intermediate_tensor_is_dram>{
        .bank_base_address = intermediate_tensor_address,
        .page_size = input_tensor_page_size,
        .data_format = get_dataformat(cb_input_id)};

    for (uint32_t b = 0; b < num_batches; b++) {
        if (fuse_op) {
            matmul_receiver.wait_for_matmul_batch(b);
        }
        uint32_t batch_offset = batch_num_pages * b;

        uint32_t actual_fwd_slice_id_x = my_chip_id_x;
        uint32_t actual_fwd_slice_id_y = my_chip_id_y;
        uint32_t actual_bwd_slice_id_x = my_chip_id_x;
        uint32_t actual_bwd_slice_id_y = my_chip_id_y;

        // Loop over the slices, starting from the furthest, and working backwards until we get to ourselves
        // Read our local slice at this slice idx into cb_input_id or cb_output_id
        // If we are not the first slice, then read intermediate into the cb_intermediate_id
        // Then reduce those two CB's, and push that to cb_output_id
        // If slices_forwarded in writer is 7, we don't forward anymore and write it to output_buffer
        // Otherwise, the writer will write cb_output_id to the next chip in the forward direction
        for (uint32_t i = 0; i < ring_size; ++i) {
            const bool do_reduce = i != 0;
            uint32_t cb_in0 = do_reduce ? cb_input_id : cb_reader_output_id;

            actual_fwd_slice_id_x = (actual_fwd_slice_id_x == 0) ? ring_size - 1 : actual_fwd_slice_id_x - 1;
            actual_bwd_slice_id_x = (actual_bwd_slice_id_x == ring_size - 1) ? 0 : actual_bwd_slice_id_x + 1;

            uint32_t intermediate_packet_id_fwd_x = actual_fwd_slice_id_x + intermediate_packet_offset_x;
            uint32_t intermediate_packet_id_fwd_y = actual_fwd_slice_id_y + intermediate_packet_offset_y;
            if (intermediate_packet_id_fwd_x >= N_DRAM_BANKS) {
                intermediate_packet_id_fwd_x -= N_DRAM_BANKS;
                intermediate_packet_id_fwd_y++;
            }

            uint32_t intermediate_packet_id_bwd_x = actual_bwd_slice_id_x + intermediate_packet_offset_x;
            uint32_t intermediate_packet_id_bwd_y = actual_bwd_slice_id_y + intermediate_packet_offset_y;
            if (intermediate_packet_id_bwd_x >= N_DRAM_BANKS) {
                intermediate_packet_id_bwd_x -= N_DRAM_BANKS;
                intermediate_packet_id_bwd_y++;
            }

            uint32_t fwd_input_tile_id_start = actual_fwd_slice_id_x * slice_Wt + batch_offset;
            uint32_t bwd_input_tile_id_start = actual_bwd_slice_id_x * slice_Wt + batch_offset;

            uint32_t pages_read_in_row = pages_read_in_row_0;
            uint32_t row_offset = row_offset_0;
            uint32_t tiles_read = tiles_read_0;
            uint32_t tiles_to_read = tiles_to_read_0;

            /**
             * Interleave forward and backward ring reads
             * forward handles even tiles, backward handles odd tiles
             * after ring_size-1 steps, we've transferred all tiles
             */
            if (do_reduce) {
                while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_fwd) <= i - 1);
                while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bwd) <= i - 1);
            }
            bool read_forward = true;
            while (tiles_read < tiles_to_read) {
                uint32_t num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);

                uint32_t intermediate_pages_read_in_row = pages_read_in_row;
                uint32_t intermediate_row_offset = row_offset;

                cb_reserve_back(cb_in0, num_pages_to_read);
                const uint32_t l1_write_addr_base = get_write_ptr(cb_in0);
                uint32_t l1_write_addr = l1_write_addr_base;

                for (uint32_t j = 0; j < num_pages_to_read; j++) {
                    noc_async_read_tile(
                        (read_forward ? fwd_input_tile_id_start : bwd_input_tile_id_start) + row_offset +
                            pages_read_in_row,
                        input_tensor_addrgen,
                        l1_write_addr);
                    l1_write_addr += input_tensor_page_size;
                    tiles_read++;

                    pages_read_in_row++;
                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = 0;
                    }
                }

                if (do_reduce) {
                    // read the next intermediate slice out of the intermediate buffer, and put it in intermediate CB
                    cb_reserve_back(cb_intermediate_id, num_pages_to_read);
                    size_t intermediate_l1_write_addr = get_write_ptr(cb_intermediate_id);
                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        uint32_t payload_size_bytes =
                            input_tensor_page_size * std::min(contig_pages_advanced, num_pages_to_read - j);

                        uint32_t intermediate_packet_first_tile_id =
                            read_forward ? (intermediate_packet_id_fwd_x +
                                            contig_pages_advanced * N_DRAM_BANKS * intermediate_packet_id_fwd_y)
                                         : (intermediate_packet_id_bwd_x +
                                            contig_pages_advanced * N_DRAM_BANKS * intermediate_packet_id_bwd_y);

                        uint64_t packet_addr = get_noc_addr(
                            intermediate_packet_first_tile_id, intermediate_tensor_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                        noc_async_read(packet_addr, intermediate_l1_write_addr, payload_size_bytes);
                        intermediate_l1_write_addr += payload_size_bytes;

                        intermediate_packet_id_fwd_x += ring_size;
                        intermediate_packet_id_bwd_x += ring_size;
                        if (intermediate_packet_id_fwd_x >= N_DRAM_BANKS) {
                            intermediate_packet_id_fwd_x -= N_DRAM_BANKS;
                            intermediate_packet_id_fwd_y++;
                        }
                        if (intermediate_packet_id_bwd_x >= N_DRAM_BANKS) {
                            intermediate_packet_id_bwd_x -= N_DRAM_BANKS;
                            intermediate_packet_id_bwd_y++;
                        }
                    }

                    noc_async_read_barrier();
                    cb_push_back(cb_intermediate_id, num_pages_to_read);
                }
                read_forward = !read_forward;
                noc_async_read_barrier();
                cb_push_back(cb_in0, num_pages_to_read);
            }
        }
    }
    DPRINT << "Done Reader\n";
}
