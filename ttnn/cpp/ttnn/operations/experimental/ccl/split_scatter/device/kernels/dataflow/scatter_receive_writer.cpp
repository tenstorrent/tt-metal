// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/experimental/ccl/split_scatter/device/kernels/dataflow/scatter_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

void kernel_main() {
    DPRINT << "receive writer start" << ENDL();
    uint32_t arg_idx = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg_idx++);
    // Different per worker receiver writer
    const uint32_t worker_sender_reader_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t worker_sender_reader_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const bool sender_enabled = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_start_page_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_start_addr_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t col_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t col_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_cols = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_output_page_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_page_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t last_output_addr_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_addr_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_start_ring_idx = get_arg_val<uint32_t>(arg_idx++);
    const bool is_clockwise_direction = get_arg_val<uint32_t>(arg_idx++) == 1;

    // Compile time Args
    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(2);
    // Same per worker receiver writer
    uint32_t sem_addr = get_semaphore(get_compile_time_arg_val(3));
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(4);
    constexpr uint32_t ring_size = get_compile_time_arg_val(5);
    constexpr uint32_t output_tile_size = get_compile_time_arg_val(6);

    ASSERT(half_cb_n_pages > rem_num_pages);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;
#ifdef ROW_MAJOR_LAYOUT
#ifdef INTERLEAVED_MEM_LAYOUT
    InterleavedAddrGen<dst_is_dram> d = {
        .bank_base_address = dst_addr + output_start_addr_offset, .page_size = output_page_size};
#endif
#elif defined TILED_LAYOUT
#ifdef INTERLEAVED_MEM_LAYOUT
    const DataFormat in0_df = get_dataformat(cb_id_in0);

    InterleavedAddrGenFast<dst_is_dram, output_tile_size> d = {
        .bank_base_address = dst_addr, .page_size = output_page_size, .data_format = in0_df};

#endif
#endif

    // Each worker receiver writer matches with a specific worker sender reader
    // Used to signal that data has been committed to memory and can be read
    uint64_t worker_send_reader_semaphore_noc_addr = -1;
    if (sender_enabled) {
        worker_send_reader_semaphore_noc_addr =
            get_noc_addr(worker_sender_reader_noc_x, worker_sender_reader_noc_y, sem_addr);
    }

    uint32_t input_ring_idx = input_start_ring_idx;
    uint32_t output_base_page_idx = output_start_page_idx;
    uint32_t output_page_idx = output_base_page_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;
    DPRINT << "receiver writer " << ENDL();
    DPRINT << "input_ring_idx " << input_ring_idx << ENDL();
    DPRINT << "dst_addr " << dst_addr << ENDL();
    // DPRINT << "input_page_idx " << input_page_idx << ENDL();
    DPRINT << "output_base_page_idx " << output_base_page_idx << ENDL();
    DPRINT << "output_page_idx " << output_page_idx << ENDL();
    DPRINT << "col_idx " << col_idx << ENDL();
    DPRINT << "row_idx " << row_idx << ENDL();
    DPRINT << "row_start_idx " << row_start_idx << ENDL();
    DPRINT << "col_start_idx " << col_start_idx << ENDL();
    DPRINT << "num_rows " << num_rows << ENDL();
    DPRINT << "num_cols " << num_cols << ENDL();
    DPRINT << "num_transfers " << num_transfers << ENDL();
    DPRINT << "num_full_chunks " << num_full_chunks << ENDL();
    DPRINT << "rem_num_pages " << rem_num_pages << ENDL();

    for (uint32_t i = 0; i < num_transfers; ++i) {
        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                write_chunk(
                    output_page_idx,
                    col_idx,
                    row_idx,
                    cb_id_in0,
                    d,
                    num_cols,
                    num_rows,
                    col_offset,
                    row_offset,
                    num_pages,
                    page_size);
                if (sender_enabled) {
                    noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, 1);
                }
            }
        }
        // col_idx++;
        // row_idx++;
        SliceRange sr1 = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        SliceRange sr2 = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 33, .w1 = 64, .ws = 1};
        SliceRange sr3 = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 65, .w1 = 96, .ws = 1};
        SliceRange sr4 = SliceRange{.h0 = 0, .h1 = 1, .hs = 1, .w0 = 97, .w1 = 128, .ws = 1};
        DPRINT << "fs writer " << ENDL();
        DPRINT << TSLICE(tt::CBIndex::c_0, 0, SliceRange::hw0_32_4(), true, true) << ENDL();
        DPRINT << "54h45 writer " << ENDL();
        DPRINT << TSLICE(tt::CBIndex::c_0, 0, SliceRange::hw0_32_8(), true, true) << ENDL();
        DPRINT << "11111 writer " << ENDL();
        DPRINT << TSLICE(tt::CBIndex::c_0, 0, sr1, true, true) << ENDL();
        DPRINT << "11111 writer " << ENDL();
        DPRINT << TSLICE(tt::CBIndex::c_0, 0, sr2, true, true) << ENDL();
        DPRINT << "11111 writer " << ENDL();
        DPRINT << TSLICE(tt::CBIndex::c_0, 0, sr3, true, true) << ENDL();
        DPRINT << "11111 writer " << ENDL();
        DPRINT << TSLICE(tt::CBIndex::c_0, 0, sr4, true, true) << ENDL();
        DPRINT << "22222 writer " << ENDL();
        DPRINT << TSLICE(tt::CBIndex::c_0, 0, SliceRange::h0_w0_32(), true, true) << ENDL();
        DPRINT << "33333 writer " << ENDL();
        DPRINT << "------------------ " << ENDL();
        // pop_filler_pages_from_cb(cb_id_in0, 1);
        // cb_pop_front(cb_id_in0, 1);
        DPRINT << TSLICE(tt::CBIndex::c_0, 0, sr1, true, true) << ENDL();

        DPRINT << "receiver ************************ " << ENDL();
        DPRINT << "input_ring_idx " << input_ring_idx << ENDL();
        DPRINT << "dst_addr " << dst_addr << ENDL();
        // DPRINT << "input_page_idx " << input_page_idx << ENDL();
        DPRINT << "output_base_page_idx " << output_base_page_idx << ENDL();
        DPRINT << "output_page_idx " << output_page_idx << ENDL();
        DPRINT << "col_idx " << col_idx << ENDL();
        DPRINT << "row_idx " << row_idx << ENDL();
        DPRINT << "row_start_idx " << row_start_idx << ENDL();
        DPRINT << "col_start_idx " << col_start_idx << ENDL();
        DPRINT << "num_rows " << num_rows << ENDL();
        DPRINT << "num_cols " << num_cols << ENDL();
        DPRINT << "num_transfers " << num_transfers << ENDL();
        DPRINT << "num_full_chunks " << num_full_chunks << ENDL();
        DPRINT << "half_cb_n_pages " << half_cb_n_pages << ENDL();
        DPRINT << "rem_num_pages " << rem_num_pages << ENDL();
        DPRINT << "page_size " << page_size << ENDL();
        DPRINT << "receiver ************************ " << ENDL();
        if (rem_num_pages > 0) {
            write_chunk(
                output_page_idx,
                col_idx,
                row_idx,
                cb_id_in0,
                d,
                num_cols,
                num_rows,
                col_offset,
                row_offset,
                rem_num_pages,
                page_size);
            if (sender_enabled) {
                noc_semaphore_inc(worker_send_reader_semaphore_noc_addr, 1);
            }
            ASSERT(num_pages == 0 || num_pages > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }

        if (is_clockwise_direction) {
            if (input_ring_idx == 0) {
                input_ring_idx = ring_size - 1;
                if (output_addr_offset != 0) {
                    d.bank_base_address += last_output_addr_offset;
                }
                if (output_page_offset != 0) {
                    output_base_page_idx += last_output_page_offset;
                }
            } else {
                input_ring_idx--;
                if (output_addr_offset != 0) {
                    d.bank_base_address -= output_addr_offset;
                }
                if (output_page_offset != 0) {
                    output_base_page_idx -= output_page_offset;
                }
            }
        } else {
            if (input_ring_idx == ring_size - 1) {
                input_ring_idx = 0;
                if (output_addr_offset != 0) {
                    d.bank_base_address -= last_output_addr_offset;
                }
                if (output_page_offset != 0) {
                    output_base_page_idx -= last_output_page_offset;
                }
            } else {
                input_ring_idx++;
                if (output_addr_offset != 0) {
                    d.bank_base_address += output_addr_offset;
                }
                if (output_page_offset != 0) {
                    output_base_page_idx += output_page_offset;
                }
            }
        }

        output_page_idx = output_base_page_idx;
        col_idx = col_start_idx;
        row_idx = row_start_idx;
    }
    DPRINT << "receive writer end" << ENDL();
}
