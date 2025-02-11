// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/experimental/ccl/split_scatter/device/kernels/dataflow/scatter_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

void kernel_main() {
    DPRINT << "send writer start" << ENDL();
    uint32_t arg_idx = 0;
    const uint32_t dst_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_sender_l1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_sender_l1_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_pages_per_full_chunk = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_start_addr_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t col_start_idx = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t col_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_rows = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_cols = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_sender_noc_y = get_arg_val<uint32_t>(arg_idx++);

    constexpr bool dst_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(2);
    constexpr uint32_t input_start_ring_idx = get_compile_time_arg_val(3);
    volatile uint32_t* const writer_send_sem_ptr =
        reinterpret_cast<volatile uint32_t* const>(get_semaphore(get_compile_time_arg_val(4)));
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(5);
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(6);
    constexpr uint32_t output_tile_size = get_compile_time_arg_val(7);

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

    const InterleavedAddrGenFast<dst_is_dram, output_tile_size> d = {
        .bank_base_address = dst_addr, .page_size = output_page_size, .data_format = in0_df};
#endif
#endif

    ccl::edm::WorkerToEdmSender<ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED> sender(
        ttnn::ccl::WorkerXY(eth_sender_noc_x, eth_sender_noc_y),
        eth_sender_l1_base_addr,
        num_buffers_per_channel,
        eth_sender_l1_sem_addr,
        (num_full_chunks > 0 ? num_pages_per_full_chunk : rem_num_pages) * page_size,
        writer_send_sem_ptr);

    uint32_t output_page_idx = output_start_idx;
    uint32_t col_idx = col_start_idx;
    uint32_t row_idx = row_start_idx;
    // This is different per writer core
    DPRINT << "send writer " << ENDL();
    // DPRINT << "input_ring_idx " << input_ring_idx << ENDL();
    // DPRINT << "input_page_idx " << input_page_idx << ENDL();
    // DPRINT << "output_base_page_idx " << output_base_page_idx << ENDL();
    DPRINT << "output_page_idx " << output_page_idx << ENDL();
    DPRINT << "col_idx " << col_idx << ENDL();
    DPRINT << "row_idx " << row_idx << ENDL();
    DPRINT << "row_start_idx " << row_start_idx << ENDL();
    DPRINT << "col_start_idx " << col_start_idx << ENDL();
    DPRINT << "num_rows " << num_rows << ENDL();
    DPRINT << "num_cols " << num_cols << ENDL();
    uint32_t ID = (my_y[0] << 16) | my_x[0];

    if (num_full_chunks > 0) {
        for (uint32_t c = 0; c < num_full_chunks; ++c) {
            sender.wait_for_empty_write_slot();
            write_and_send_chunk(
                output_page_idx,
                col_idx,
                row_idx,
                cb_id_in0,
                d,
                num_cols,
                num_rows,
                col_offset,
                row_offset,
                num_pages_per_full_chunk,
                page_size,
                sender);
        }
    }

    if (rem_num_pages > 0) {
        sender.wait_for_empty_write_slot();
        write_and_send_chunk(
            output_page_idx,
            ++col_idx,
            ++row_idx,
            cb_id_in0,
            d,
            num_cols,
            num_rows,
            col_offset + 1,
            row_offset + 1,
            rem_num_pages,
            page_size,
            sender);
        ASSERT(num_pages_per_full_chunk == 0 || num_pages_per_full_chunk > rem_num_pages);
        ASSERT(half_cb_n_pages > rem_num_pages);
        pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
    }

    // num_transfers = num_devices - 1
    for (uint32_t i = 1; i < num_transfers; ++i) {
        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                sender.wait_for_empty_write_slot();
                sender.send_payload_blocking(cb_id_in0, num_pages_per_full_chunk, page_size);
            }
        }
        if (rem_num_pages > 0) {
            sender.wait_for_empty_write_slot();
            sender.send_payload_blocking(cb_id_in0, rem_num_pages, page_size);
            ASSERT(num_pages_per_full_chunk == 0 || num_pages_per_full_chunk > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            pop_filler_pages_from_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }
    }

    sender.close();
    DPRINT << "send writer end" << ENDL();
}
