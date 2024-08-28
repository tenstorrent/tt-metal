// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"

using ttnn::ccl::coord_t;

void kernel_main() {
    constexpr bool is_sharded = get_compile_time_arg_val(0) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(2);

    constexpr tt::tt_metal::TensorMemoryLayout output_tensor_memory_layout =
        static_cast<tt::tt_metal::TensorMemoryLayout>(get_compile_time_arg_val(3));
    #ifdef SHARDED_MEM_LAYOUT
    constexpr uint32_t output_tensor_shard_grid_height = get_compile_time_arg_val(4);
    constexpr uint32_t output_tensor_shard_grid_width = get_compile_time_arg_val(5);
    constexpr uint32_t output_tensor_shard_grid_start_y_logical = get_compile_time_arg_val(6);
    constexpr uint32_t output_tensor_shard_grid_start_x_logical = get_compile_time_arg_val(7);
    constexpr uint32_t output_tensor_shard_pages_per_shard_y = get_compile_time_arg_val(8);
    constexpr uint32_t output_tensor_shard_pages_per_shard_x = get_compile_time_arg_val(9);
    constexpr bool output_tensor_shard_grid_transposed = get_compile_time_arg_val(10) != 0;
    #endif

    uint32_t arg_idx = 0;
    uint32_t const dst_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const eth_sender_l1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const eth_sender_l1_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const eth_sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const eth_sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const num_transfers = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const page_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const full_chunk_num_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const writer_send_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t const half_cb_n_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t const num_concurrent_workers = get_arg_val<uint32_t>(arg_idx++);

    coord_t const& output_tensor_shape = ttnn::ccl::coord_from_args(arg_idx);
    coord_t const& worker_slice_shape = ttnn::ccl::coord_from_args(arg_idx);
    coord_t worker_slice_base_offset = ttnn::ccl::coord_from_args(arg_idx);

    uint32_t total_eltwise_kernel_num_pages = get_arg_val<uint32_t>(arg_idx++);


    #ifdef SHARDED_MEM_LAYOUT
    uint32_t output_shard_grid_nrows = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t * const output_shard_grid_row_map = reinterpret_cast<const uint32_t * const>(get_arg_addr(arg_idx));
    arg_idx += output_shard_grid_nrows;
    uint32_t output_shard_grid_ncols = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t * const output_shard_grid_col_map = reinterpret_cast<const uint32_t * const>(get_arg_addr(arg_idx));
    arg_idx += output_shard_grid_ncols;
    #endif

    // Argument validation
    ASSERT(half_cb_n_pages >= full_chunk_num_pages);
    ASSERT(full_chunk_num_pages > 0);
    ASSERT(page_size > 0);
    ASSERT(half_cb_n_pages > 0);

    constexpr uint32_t cb_id_in0 = tt::CB::c_out0;
    constexpr uint32_t cb_id_in_short_circuit = tt::CB::c_out1;
    const DataFormat in0_df = get_dataformat(cb_id_in0);
#ifdef ROW_MAJOR_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = page_size};
    #elif defined SHARDED_MEM_LAYOUT
            auto d = tt::tt_metal::address_generators::build_sharded_addr_gen<output_tensor_memory_layout>(
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
                page_size,
                dst_addr
            );
            ASSSERT(false); // unimplemented and untested
        #endif
#elif defined TILED_LAYOUT
    #ifdef INTERLEAVED_MEM_LAYOUT
    InterleavedAddrGenFast<dst_is_dram> d = {
        .bank_base_address = dst_addr, .page_size = page_size, .data_format = in0_df};
    #elif defined SHARDED_MEM_LAYOUT
    auto d = tt::tt_metal::address_generators::build_sharded_addr_gen<output_tensor_memory_layout>(
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
        page_size,
        dst_addr
    );
    #endif
#endif

    // Used to wait until eth sender has space available
    volatile tt_l1_ptr uint32_t* writer_send_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_send_sem_addr);

    ccl::edm::WorkerToEdmSender<ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED> sender(
        ttnn::ccl::WorkerXY(eth_sender_noc_x, eth_sender_noc_y),
        eth_sender_l1_base_addr,
        num_buffers_per_channel,
        eth_sender_l1_sem_addr,
        // (num_full_chunks > 0 ? num_pages_per_full_chunk : rem_num_pages) * page_size,
        full_chunk_num_pages * page_size,
        writer_send_semaphore_addr_ptr);

    uint32_t total_lifetime_cb_pages_popped_from_math = 0;
    while (worker_slice_base_offset.x < output_tensor_shape.x && worker_slice_base_offset.y < output_tensor_shape.y) {
        // First phase - we only forward messages to EDM
        // Set the valid_worker_slice_shape
        coord_t valid_worker_slice_shape = worker_slice_shape;
        if (worker_slice_base_offset.y == output_tensor_shape.y - 1) { // Worker is on last row of tensor_slice
            if (output_tensor_shape.x - worker_slice_base_offset.x < worker_slice_shape.x) { // Worker is cutoff by the end of the tensor_slice
                valid_worker_slice_shape.x = output_tensor_shape.x - worker_slice_base_offset.x;
            }
        }
        uint32_t const num_pages_to_write = valid_worker_slice_shape.x * valid_worker_slice_shape.y;

        ASSERT(total_lifetime_cb_pages_popped_from_math + num_pages_to_write <= total_eltwise_kernel_num_pages);
        for (uint32_t i = 0; i < num_transfers; ++i) {
            const uint32_t cb_in = i == 0 ? cb_id_in_short_circuit : cb_id_in0;
            for (uint32_t p = 0; p < num_pages_to_write; p += full_chunk_num_pages) {
                uint32_t n_pages = std::min(full_chunk_num_pages, num_pages_to_write - p);
                ASSERT(n_pages > 0);
                sender.wait_for_empty_write_slot();
                sender.send_payload_blocking(cb_in, n_pages, page_size);

                if (i != 0) {
                    total_lifetime_cb_pages_popped_from_math += n_pages;
                }
                if (n_pages < half_cb_n_pages) {
                    uint32_t num_filler_pages = half_cb_n_pages - n_pages;

                    ASSERT(p + n_pages == num_pages_to_write);
                    pop_filler_pages_from_cb(cb_in, num_filler_pages);
                    if (i != 0) {
                        total_lifetime_cb_pages_popped_from_math += num_filler_pages;
                    }
                }
            }
        }

        // write the final reduced chunk for this chip out to the output tensor
        // Second phase - Dump the local output to the output tensor
        uint32_t curr_ring_slice_start_page_offset = 0;
        const uint32_t worker_relative_start_offset_into_slice =
            worker_slice_base_offset.x + (worker_slice_base_offset.y * output_tensor_shape.x);

        const uint32_t starting_tile_id = curr_ring_slice_start_page_offset + worker_relative_start_offset_into_slice;
        uint32_t curr_tile_id = starting_tile_id;

        uint32_t offset_into_worker_slice = 0;
        bool last_page_of_worker = false;
        for (uint32_t p = 0; p < num_pages_to_write; p += full_chunk_num_pages) {
            ASSERT(curr_tile_id < output_tensor_shape.x * output_tensor_shape.y);
            ASSERT(!last_page_of_worker);
            uint32_t n_pages = std::min(full_chunk_num_pages, num_pages_to_write - p);
            ASSERT(n_pages <= half_cb_n_pages);
            ASSERT(full_chunk_num_pages <= half_cb_n_pages);
            write_wrapped_chunk(
                curr_tile_id,
                offset_into_worker_slice,
                worker_slice_base_offset, // Offset into tensor slice
                valid_worker_slice_shape,
                output_tensor_shape,  // In tiles for tile layout
                output_tensor_shape,
                cb_id_in0,
                d,
                n_pages,
                page_size,
                last_page_of_worker);
            total_lifetime_cb_pages_popped_from_math += n_pages;
            if (n_pages < half_cb_n_pages) {
                uint32_t num_filler_pages = half_cb_n_pages - n_pages;
                ASSERT(p + n_pages == num_pages_to_write);
                pop_filler_pages_from_cb(cb_id_in0, num_filler_pages);
                total_lifetime_cb_pages_popped_from_math += num_filler_pages;
            }
        }

        worker_slice_base_offset = advance_wrapped_slice_row_major(
            worker_slice_base_offset, worker_slice_shape, output_tensor_shape, num_concurrent_workers);
    }

    ASSERT(total_lifetime_cb_pages_popped_from_math <= total_eltwise_kernel_num_pages);
    for (; total_lifetime_cb_pages_popped_from_math < total_eltwise_kernel_num_pages;
         total_lifetime_cb_pages_popped_from_math++) {
        pop_filler_pages_from_cb(cb_id_in0, 1);
    }

    sender.close();
}
