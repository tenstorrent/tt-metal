// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"


void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg_idx++);
    // Different per worker receiver writer
    const uint32_t worker_sender_reader_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_sender_reader_noc_y = get_arg_val<uint32_t>(arg_idx++);

    #ifdef SHARDED_MEM_LAYOUT
    uint32_t output_shard_grid_nrows = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t * const output_shard_grid_row_map = reinterpret_cast<const uint32_t * const>(get_arg_addr(arg_idx));
    arg_idx += output_shard_grid_nrows;
    uint32_t output_shard_grid_ncols = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t * const output_shard_grid_col_map = reinterpret_cast<const uint32_t * const>(get_arg_addr(arg_idx));
    arg_idx += output_shard_grid_ncols;
    #endif

    // Compile time Args
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_pages = get_compile_time_arg_val(5);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(6);
    constexpr uint32_t output_start_page_idx = get_compile_time_arg_val(7);
    constexpr uint32_t output_start_addr_offset = get_compile_time_arg_val(8);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(9);
    constexpr uint32_t col_start_idx = get_compile_time_arg_val(10);
    constexpr uint32_t row_offset = get_compile_time_arg_val(11);
    constexpr uint32_t col_offset = get_compile_time_arg_val(12);
    constexpr uint32_t num_rows = get_compile_time_arg_val(13);
    constexpr uint32_t num_cols = get_compile_time_arg_val(14);
    constexpr uint32_t last_output_page_offset = get_compile_time_arg_val(15);
    constexpr uint32_t output_page_offset = get_compile_time_arg_val(16);
    constexpr uint32_t last_output_addr_offset = get_compile_time_arg_val(17);
    constexpr uint32_t output_addr_offset = get_compile_time_arg_val(18);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(19);
    // Same per worker receiver writer
    uint32_t sem_addr = get_semaphore(get_compile_time_arg_val(20));
    constexpr bool is_clockwise_direction = get_compile_time_arg_val(21) == 1;
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(22);
    constexpr uint32_t ring_size = get_compile_time_arg_val(23);
    static_assert(half_cb_n_pages > rem_num_pages, "half_cb_n_pages must be greater than or equal to rem_num_pages");

    #ifdef SHARDED_MEM_LAYOUT
    constexpr tt::tt_metal::TensorMemoryLayout output_tensor_memory_layout = static_cast<tt::tt_metal::TensorMemoryLayout>(get_compile_time_arg_val(24));
    constexpr uint32_t output_tensor_shard_grid_height = get_compile_time_arg_val(25);
    constexpr uint32_t output_tensor_shard_grid_width = get_compile_time_arg_val(26);
    constexpr uint32_t output_tensor_shard_grid_start_y_logical = get_compile_time_arg_val(27);
    constexpr uint32_t output_tensor_shard_grid_start_x_logical = get_compile_time_arg_val(28);
    constexpr uint32_t output_tensor_shard_pages_per_shard_y = get_compile_time_arg_val(29);
    constexpr uint32_t output_tensor_shard_pages_per_shard_x = get_compile_time_arg_val(30);
    constexpr bool output_tensor_shard_grid_transposed = get_compile_time_arg_val(31) != 0;
    #endif

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    #ifdef ROW_MAJOR_LAYOUT
        #ifdef INTERLEAVED_MEM_LAYOUT
        InterleavedAddrGen<dst_is_dram> d = {
            .bank_base_address = dst_addr + output_start_addr_offset, .page_size = output_page_size};
        #elif defined SHARDED_MEM_LAYOUT
            auto d =
                tt::tt_metal::address_generators::build_sharded_addr_gen<output_tensor_memory_layout>(
                tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(output_shard_grid_nrows, output_shard_grid_row_map, output_shard_grid_ncols, output_shard_grid_col_map),
                tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<output_tensor_memory_layout>::type(
                    output_tensor_shard_pages_per_shard_y,
                    output_tensor_shard_pages_per_shard_x,
                    output_tensor_shard_grid_height,
                    output_tensor_shard_grid_width,
                    output_tensor_shard_grid_start_y_logical,
                    output_tensor_shard_grid_start_x_logical,
                    output_tensor_shard_grid_transposed
                ),
                output_page_size,
                dst_addr
            );
            ASSERT(false);
        #endif
    #elif defined TILED_LAYOUT
        #ifdef INTERLEAVED_MEM_LAYOUT
        const DataFormat in0_df = get_dataformat(cb_id_in0);

        InterleavedAddrGenFast<dst_is_dram> d = {
            .bank_base_address = dst_addr,
            .page_size = output_page_size,
            .data_format = in0_df
        };

        #elif defined SHARDED_MEM_LAYOUT
            auto d =
                tt::tt_metal::address_generators::build_sharded_addr_gen<output_tensor_memory_layout>(
                tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(output_shard_grid_nrows, output_shard_grid_row_map, output_shard_grid_ncols, output_shard_grid_col_map),
                tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<output_tensor_memory_layout>::type(
                    output_tensor_shard_pages_per_shard_y,
                    output_tensor_shard_pages_per_shard_x,
                    output_tensor_shard_grid_height,
                    output_tensor_shard_grid_width,
                    output_tensor_shard_grid_start_y_logical,
                    output_tensor_shard_grid_start_x_logical,
                    output_tensor_shard_grid_transposed
                ),
                output_page_size,
                dst_addr
            );
        #endif
    #endif

    // Each worker receiver writer matches with a specific worker sender reader
    // Used to signal that data has been committed to memory and can be read
    const uint64_t worker_send_reader_semaphore_noc_addr = get_noc_addr(worker_sender_reader_noc_x, worker_sender_reader_noc_y, sem_addr);

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t output_base_page_idx = output_start_page_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;

#ifdef OVERLAP_OP
    /* Args for overlapped all gather */

    // Compile time args
    constexpr uint32_t overlap_op = get_compile_time_arg_val(24) ? 1 : 0;
    constexpr uint32_t global_num_workers = get_compile_time_arg_val(25);
    constexpr uint32_t curr_worker_index = get_compile_time_arg_val(26);
    constexpr uint32_t worker_sync_sem_addr = get_compile_time_arg_val(27);

    // Runtime args
    uint32_t worker_noc_coords[global_num_workers * 2]; // Worker NOC coordinates [x1, y1, x2, y2...]
    uint32_t op_worker_noc_x;
    uint32_t op_worker_noc_y;
    uint32_t signal_op_sem_addr;

    if (overlap_op) {
        for (uint32_t i = 0; i < global_num_workers * 2; i+=2) {
            worker_noc_coords[i] = get_arg_val<uint32_t>(arg_idx++);
            worker_noc_coords[i + 1] = get_arg_val<uint32_t>(arg_idx++);
        }

        op_worker_noc_x = get_arg_val<uint32_t>(arg_idx++);
        op_worker_noc_y = get_arg_val<uint32_t>(arg_idx++);
        signal_op_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    }


    /* Setup for overlapped all_gather */
    uint32_t master_worker_receiver_noc_x;
    uint32_t master_worker_receiver_noc_y;
    uint32_t curr_receiver_worker_noc_x;
    uint32_t curr_receiver_worker_noc_y;
    uint32_t curr_worker_is_master;
    volatile tt_l1_ptr uint32_t* curr_worker_l1_semaphore_addr_ptr = nullptr;
    uint64_t worker_sem_noc_addrs[global_num_workers]; // First one is for master
    uint64_t signal_op_sem_noc_addr = 0;


    if (overlap_op) {
        master_worker_receiver_noc_x = worker_noc_coords[0];
        master_worker_receiver_noc_y = worker_noc_coords[1];

        curr_receiver_worker_noc_x = worker_noc_coords[curr_worker_index * 2];
        curr_receiver_worker_noc_y = worker_noc_coords[curr_worker_index * 2 + 1];

        curr_worker_is_master = is_master(
            master_worker_receiver_noc_x, master_worker_receiver_noc_y, curr_receiver_worker_noc_x, curr_receiver_worker_noc_y) ? 1 : 0;

        curr_worker_l1_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(worker_sync_sem_addr);

        // Convert sem addresses into remote sem addresses
        if (curr_worker_is_master) { // If cur is master, skip doing the conversion for the master slot
            for (uint32_t i = 1; i < global_num_workers; i++) {
                worker_sem_noc_addrs[i] = get_noc_addr(worker_noc_coords[i * 2], worker_noc_coords[i * 2 + 1], worker_sync_sem_addr);
            }
        } else { // If cur is slave, only do conversion for the master slot
            worker_sem_noc_addrs[0] = get_noc_addr(master_worker_receiver_noc_x, master_worker_receiver_noc_y, worker_sync_sem_addr);
        }

        signal_op_sem_noc_addr = get_noc_addr(op_worker_noc_x, op_worker_noc_y, signal_op_sem_addr);
    }
#endif

    for (uint32_t i = 0; i < num_transfers; ++i) {
        if constexpr (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {

                #ifdef SHARDED_MEM_LAYOUT
                ASSERT(output_page_idx < output_tensor_shard_pages_per_shard_y * output_tensor_shard_pages_per_shard_x * output_tensor_shard_grid_height * output_tensor_shard_grid_width);
                #endif
                write_chunk(output_page_idx, col_idx, row_idx, cb_id_in0, d, num_cols, num_rows, col_offset, row_offset, num_pages, page_size);
                noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, 1);
            }
        }
        if constexpr (rem_num_pages > 0) {
            #ifdef SHARDED_MEM_LAYOUT
            ASSERT(output_page_idx < output_tensor_shard_pages_per_shard_y * output_tensor_shard_pages_per_shard_x * output_tensor_shard_grid_height * output_tensor_shard_grid_width);
            #endif
            write_chunk(output_page_idx, col_idx, row_idx, cb_id_in0, d, num_cols, num_rows, col_offset, row_offset, rem_num_pages, page_size);
            noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, 1);
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }

        if (is_clockwise_direction) {
            if (input_ring_idx == 0) {
                input_ring_idx = ring_size - 1;
                if constexpr(output_addr_offset != 0) {
                    d.bank_base_address += last_output_addr_offset;
                }
                if constexpr(output_page_offset != 0) {
                    output_base_page_idx += last_output_page_offset;
                }
            } else {
                input_ring_idx--;
                if constexpr(output_addr_offset != 0) {
                    d.bank_base_address -= output_addr_offset;
                }
                if constexpr(output_page_offset != 0) {
                    output_base_page_idx -= output_page_offset;
                }
            }
        } else {
            if (input_ring_idx == ring_size - 1) {
                input_ring_idx = 0;
                if constexpr(output_addr_offset != 0) {
                    d.bank_base_address -= last_output_addr_offset;
                }
                if constexpr(output_page_offset != 0) {
                    output_base_page_idx -= last_output_page_offset;
                }
            } else {
                input_ring_idx++;
                if constexpr(output_addr_offset != 0) {
                    d.bank_base_address += output_addr_offset;
                }
                if constexpr(output_page_offset != 0) {
                    output_base_page_idx += output_page_offset;
                }
            }

        }
#ifdef OVERLAP_OP
        // Synchronize if all gather fusion is enabled
        if (overlap_op) {
            // DPRINT << "curr_worker_is_master: " << curr_worker_is_master << ENDL();
            if (curr_worker_is_master) {
                master_sync_slaves(curr_worker_l1_semaphore_addr_ptr, global_num_workers - 1, worker_sem_noc_addrs + 1, signal_op_sem_noc_addr/* OP semaphore */);
            } else {
                slave_sync_master(curr_worker_l1_semaphore_addr_ptr, worker_sem_noc_addrs[0]);
            }
        }
#endif
        output_page_idx = output_base_page_idx;
        col_idx = col_start_idx;
        row_idx = row_start_idx;
    }
}
