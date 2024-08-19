// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>
#include "dataflow_api.h"

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "tt_metal/hw/inc/ethernet/dataflow_api.h"

namespace ccl {
namespace edm {

template <ttnn::ccl::EriscDataMoverTerminationMode termination_mode>
struct WorkerToEdmReader{
    constexpr WorkerToEdmReader (
        ttnn::ccl::WorkerXY edm_worker_xy,
        std::size_t edm_buffer_base_addr,
        std::size_t num_buffers_per_channel,
        std::size_t edm_l1_sem_addr,
        std::size_t buffer_size_bytes,
        volatile uint32_t * const worker_sem_addr
    ) :
        edm_buffer_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_buffer_base_addr)),
        edm_semaphore_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_l1_sem_addr)),
        worker_sem_addr(worker_sem_addr),
        edm_buffer_base_addr(edm_buffer_base_addr),
        num_buffers_per_channel(num_buffers_per_channel),
        last_buffer_index(num_buffers_per_channel - 1),
        edm_l1_sem_addr(edm_l1_sem_addr),
        buffer_size_bytes(buffer_size_bytes),
        buffer_index(0)
    {}

    FORCE_INLINE void wait_for_payload_available() const {
        noc_semaphore_wait_min(worker_sem_addr, 1);
        if (*worker_sem_addr > 1) {
            DPRINT << "ERROR!!!!!!!!!!!!!!!!!!!!!!!!!\n";
            ASSERT(false);
        }
        noc_semaphore_set(worker_sem_addr, 0);
    }

    FORCE_INLINE void fetch_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size, bool last_message) {
        uint64_t buffer_address = edm_buffer_addr + (buffer_index * (buffer_size_bytes + sizeof(eth_channel_sync_t)));
        fetch_chunk(cb_id, num_pages, page_size, buffer_address);
        if constexpr (termination_mode == ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED) {
            if (!last_message) {
                DPRINT << "fetch_payload_blocking: incrementing semaphore to " << (uint32_t)(edm_semaphore_addr & 0xFFFFFFFF) << "\n";
                noc_semaphore_inc(edm_semaphore_addr, ttnn::ccl::EriscDataMoverWorkerSignal::NEXT_MESSAGE_AVAILABLE);
            }
        } else {
            noc_semaphore_inc(edm_semaphore_addr, ttnn::ccl::EriscDataMoverWorkerSignal::NEXT_MESSAGE_AVAILABLE);
        }
        buffer_index = (buffer_index == last_buffer_index) ? 0 : buffer_index + 1;
    }

    FORCE_INLINE void fetch_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        // With worker initiated termination mode, we must always specify if we are sending the last message or not
        ASSERT(termination_mode != ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED);
        fetch_payload_blocking(cb_id, num_pages, page_size, false);
    }

    FORCE_INLINE void close() {
        if constexpr (termination_mode == ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED) {
            noc_semaphore_inc(edm_semaphore_addr, ttnn::ccl::EriscDataMoverWorkerSignal::TERMINATE_IMMEDIATELY);
        }
    }

    uint64_t edm_buffer_addr;
    uint64_t edm_semaphore_addr;
    volatile uint32_t * const worker_sem_addr;
    std::size_t edm_buffer_base_addr;
    std::size_t num_buffers_per_channel;
    std::size_t last_buffer_index;
    std::size_t edm_l1_sem_addr;
    std::size_t buffer_size_bytes;
    std::size_t buffer_index;
};


template <ttnn::ccl::EriscDataMoverTerminationMode termination_mode>
struct WorkerToEdmSender{
    constexpr WorkerToEdmSender (
        ttnn::ccl::WorkerXY edm_worker_xy,
        std::size_t edm_buffer_base_addr,
        std::size_t num_buffers_per_channel,
        std::size_t edm_l1_sem_addr,
        std::size_t buffer_size_bytes,
        volatile uint32_t * const worker_sem_addr
    ) :
        edm_buffer_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_buffer_base_addr)),
        edm_semaphore_addr(get_noc_addr(edm_worker_xy.x, edm_worker_xy.y, edm_l1_sem_addr)),
        worker_sem_addr(worker_sem_addr),
        edm_buffer_base_addr(edm_buffer_base_addr),
        num_buffers_per_channel(num_buffers_per_channel),
        last_buffer_index(num_buffers_per_channel - 1),
        edm_l1_sem_addr(edm_l1_sem_addr),
        buffer_size_bytes(buffer_size_bytes),
        buffer_index(0)
    {
        ASSERT(buffer_size_bytes > 0);
    }

    FORCE_INLINE void wait_for_empty_write_slot() const {
        noc_semaphore_wait(worker_sem_addr, 1);
        noc_semaphore_set(worker_sem_addr, 0);
    }

    FORCE_INLINE void send_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        uint64_t buffer_address = edm_buffer_addr + (buffer_index * (this->buffer_size_bytes + sizeof(eth_channel_sync_t)));
        DPRINT << "SENDER SEND buffer_size_bytes = " << (uint32_t)(this->buffer_size_bytes) << "\n";
        DPRINT << "SENDER SEND " << (uint32_t)(buffer_address & 0xffffffff) << " -> " << (uint32_t)((buffer_address & 0xffffffff) + (page_size * num_pages)) << "\n";
        send_chunk(cb_id, num_pages, page_size, buffer_address);
        noc_semaphore_inc(edm_semaphore_addr, 1);
        buffer_index = (buffer_index == last_buffer_index) ? 0 : buffer_index + 1;
    }

    FORCE_INLINE void close() {
        if constexpr (termination_mode == ttnn::ccl::EriscDataMoverTerminationMode::WORKER_INITIATED) {
            this->wait_for_empty_write_slot();
            noc_semaphore_inc(edm_semaphore_addr, ttnn::ccl::EriscDataMoverWorkerSignal::TERMINATE_IMMEDIATELY);
        }
    }

    uint64_t edm_buffer_addr;
    uint64_t edm_semaphore_addr;
    volatile uint32_t * const worker_sem_addr;
    std::size_t edm_buffer_base_addr;
    std::size_t num_buffers_per_channel;
    std::size_t last_buffer_index;
    std::size_t edm_l1_sem_addr;
    std::size_t buffer_size_bytes;
    std::size_t buffer_index;
};


} // namespace edm
} // namespace ccl
