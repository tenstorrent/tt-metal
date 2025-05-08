// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    volatile uint32_t* receiver_read_sem_addr =
        reinterpret_cast<volatile uint32_t*>(get_semaphore(get_compile_time_arg_val(1)));
    constexpr uint32_t half_cb_n_pages = get_compile_time_arg_val(2);
    constexpr uint32_t num_buffers_per_channel = get_compile_time_arg_val(3);

    uint32_t arg_idx = 0;
    const uint32_t eth_receiver_l1_base_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_transfers = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_full_chunks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_pages_per_full_chunk = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rem_num_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_receiver_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_receiver_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_receiver_l1_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);

    ASSERT(half_cb_n_pages > rem_num_pages);

    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0;

    ccl::edm::WorkerToEdmReader<ttnn::ccl::EriscDataMoverTerminationMode::MESSAGE_COUNT_REACHED> reader(
        ttnn::ccl::WorkerXY(eth_receiver_noc_x, eth_receiver_noc_y),
        eth_receiver_l1_base_addr,
        num_buffers_per_channel,
        eth_receiver_l1_semaphore_addr,
        (num_full_chunks > 0 ? num_pages_per_full_chunk : rem_num_pages) * page_size,
        receiver_read_sem_addr);

    for (uint32_t i = 0; i < num_transfers; ++i) {
        if (num_full_chunks > 0) {
            for (uint32_t c = 0; c < num_full_chunks; ++c) {
                reader.wait_for_payload_available();
                reader.fetch_payload_blocking(cb_id_in0, num_pages_per_full_chunk, page_size, false);
            }
        }
        if (rem_num_pages > 0) {
            reader.wait_for_payload_available();
            reader.fetch_payload_blocking(cb_id_in0, rem_num_pages, page_size, false);
            ASSERT(num_pages_per_full_chunk == 0 || num_pages_per_full_chunk > rem_num_pages);
            ASSERT(half_cb_n_pages > rem_num_pages);
            push_filler_pages_to_cb(cb_id_in0, half_cb_n_pages - rem_num_pages);
        }
    }

    reader.close();

    noc_async_full_barrier();
}
