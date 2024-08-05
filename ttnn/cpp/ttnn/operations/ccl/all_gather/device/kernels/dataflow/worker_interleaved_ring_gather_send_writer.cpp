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
    const uint32_t eth_sender_l1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_sender_l1_sem_addr = get_arg_val<uint32_t>(arg_idx++);

    #ifdef SHARDED
    uint32_t output_shard_grid_nrows = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t * const output_shard_grid_row_map = reinterpret_cast<const uint32_t * const>(get_arg_addr(arg_idx));
    arg_idx += output_shard_grid_nrows;
    uint32_t output_shard_grid_ncols = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t * const output_shard_grid_col_map = reinterpret_cast<const uint32_t * const>(get_arg_addr(arg_idx));
    arg_idx += output_shard_grid_ncols;
    #endif

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t num_transfers = get_compile_time_arg_val(1);
    constexpr uint32_t num_full_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_pages = get_compile_time_arg_val(5);
    constexpr uint32_t rem_num_pages = get_compile_time_arg_val(6);
    constexpr uint32_t input_start_idx = get_compile_time_arg_val(7);
    constexpr uint32_t output_start_idx = get_compile_time_arg_val(8);
    constexpr uint32_t output_start_addr_offset = get_compile_time_arg_val(9);
    constexpr uint32_t row_start_idx = get_compile_time_arg_val(10);
    constexpr uint32_t col_start_idx = get_compile_time_arg_val(11);
    constexpr uint32_t row_offset = get_compile_time_arg_val(12);
    constexpr uint32_t col_offset = get_compile_time_arg_val(13);
    constexpr uint32_t num_rows = get_compile_time_arg_val(14);
    constexpr uint32_t num_cols = get_compile_time_arg_val(15);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(16);
    constexpr uint32_t writer_send_sem_addr = get_compile_time_arg_val(17);
    constexpr uint32_t eth_sender_noc_x = get_compile_time_arg_val(18);
    constexpr uint32_t eth_sender_noc_y = get_compile_time_arg_val(19);
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(20);
    static_assert(half_cb_n_pages > rem_num_pages, "half_cb_n_pages must be greater than or equal to rem_num_pages");

    #ifdef SHARDED
    constexpr uint32_t output_tensor_shard_grid_height = get_compile_time_arg_val(21);
    constexpr uint32_t output_tensor_shard_grid_width = get_compile_time_arg_val(22);
    constexpr uint32_t output_tensor_shard_grid_start_y_logical = get_compile_time_arg_val(23);
    constexpr uint32_t output_tensor_shard_grid_start_x_logical = get_compile_time_arg_val(24);
    constexpr uint32_t output_tensor_shard_pages_per_shard = get_compile_time_arg_val(25);
    constexpr bool output_tensor_shard_grid_transposed = get_compile_time_arg_val(26) != 0;
    #endif

    constexpr uint32_t cb_id_in0 = tt::CB::c_in0;
    #ifdef ROW_MAJOR
        #ifdef INTERLEAVED
        InterleavedAddrGen<dst_is_dram> d = {
            .bank_base_address = dst_addr + output_start_addr_offset, .page_size = output_page_size};
        #elif defined SHARDED
            WidthShardedAddressGenerator<HarvestedWormholeWorkerToNocLookup> d = {
                HarvestedWormholeWorkerToNocLookup(output_shard_grid_nrows, output_shard_grid_row_map, output_shard_grid_ncols, output_shard_grid_col_map),
                .device_shard_spec = {
                    .shard_grid_height = output_tensor_shard_grid_height,
                    .shard_grid_width = output_tensor_shard_grid_width,
                    .shard_grid_start_y_logical = output_tensor_shard_grid_start_y_logical,
                    .shard_grid_start_x_logical = output_tensor_shard_grid_start_x_logical,
                    .pages_per_shard = output_tensor_shard_pages_per_shard,
                    .transposed_grid = output_tensor_shard_grid_transposed
                },
                .page_size = output_page_size,
                .page_offset = dst_addr
            };
            ASSSERT(false); // unimplemented and untested
        #endif
    #elif defined TILED
        #ifdef INTERLEAVED
        const DataFormat in0_df = get_dataformat(cb_id_in0);

        const InterleavedAddrGenFast<dst_is_dram> d = {
            .bank_base_address = dst_addr,
            .page_size = output_page_size,
            .data_format = in0_df
        };
        #elif defined SHARDED
            WidthShardedAddressGenerator<HarvestedWormholeWorkerToNocLookup> d = {
                HarvestedWormholeWorkerToNocLookup(output_shard_grid_nrows, output_shard_grid_row_map, output_shard_grid_ncols, output_shard_grid_col_map),
                .device_shard_spec = {
                    .shard_grid_height = output_tensor_shard_grid_height,
                    .shard_grid_width = output_tensor_shard_grid_width,
                    .shard_grid_start_y_logical = output_tensor_shard_grid_start_y_logical,
                    .shard_grid_start_x_logical = output_tensor_shard_grid_start_x_logical,
                    .pages_per_shard = output_tensor_shard_pages_per_shard,
                    .transposed_grid = output_tensor_shard_grid_transposed
                },
                .page_size = output_page_size,
                .page_offset = dst_addr
            };
        #endif
    #endif

    for (uint32_t i = 0; i < output_shard_grid_nrows; i++) {
        DPRINT << "r(logical)=" << i << ", r(noc)=" << output_shard_grid_row_map[i] << "\n";
    }
    for (uint32_t i = 0; i < output_shard_grid_ncols; i++) {
        DPRINT << "c(logical)=" << i << ", c(noc)=" << output_shard_grid_col_map[i] << "\n";
    }

    // Used to wait until eth sender has space available
    volatile tt_l1_ptr uint32_t* writer_send_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_send_sem_addr);

    uint32_t output_page_idx = output_start_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;
    // This is different per writer core
    const uint64_t eth_l1_sender_base_noc_addr = get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_base_addr);
    // Used to signal eth sender that data is available. This is different per writer core
    const uint64_t eth_l1_sender_semaphore_addr = get_noc_addr(eth_sender_noc_x, eth_sender_noc_y, eth_sender_l1_sem_addr);

    uint32_t ID = (my_y[0] << 16) | my_x[0];

    if constexpr(num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
            noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
            // TODO: Might be better to split this?
            write_and_send_chunk(output_page_idx, col_idx, row_idx, cb_id_in0, d, num_cols, num_rows, col_offset, row_offset, num_pages, page_size, eth_l1_sender_base_noc_addr, eth_l1_sender_semaphore_addr);
        }
    }

    if constexpr(rem_num_pages > 0) {
        noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
        noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
        write_and_send_chunk(output_page_idx, col_idx, row_idx, cb_id_in0, d, num_cols, num_rows, col_offset, row_offset, rem_num_pages, page_size, eth_l1_sender_base_noc_addr,eth_l1_sender_semaphore_addr);
        ASSERT(num_pages == 0 || num_pages > rem_num_pages);
        ASSERT(half_cb_n_pages > rem_num_pages);
        pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
    }

    // num_transfers = num_devices - 1
    for (uint32_t i = 1; i < num_transfers; ++i) {
        if constexpr(num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
                noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
                send_chunk(cb_id_in0, num_pages, page_size, eth_l1_sender_base_noc_addr);
                noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
            }
        }
        if constexpr(rem_num_pages > 0) {
            noc_semaphore_wait(writer_send_semaphore_addr_ptr, 1);
            noc_semaphore_set(writer_send_semaphore_addr_ptr, 0);
            send_chunk(cb_id_in0, rem_num_pages, page_size, eth_l1_sender_base_noc_addr);
            noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }
    }
}
