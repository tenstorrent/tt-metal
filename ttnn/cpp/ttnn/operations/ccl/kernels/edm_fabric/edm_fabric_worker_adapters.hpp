// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

#include "tt_metal/hw/inc/ethernet/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header_validate.hpp"
#include "debug/assert.h"

#include <cstdint>


namespace tt::fabric {

struct WorkerToFabricEdmSender{

    static constexpr uint32_t open_connection_value = 1;
    static constexpr uint32_t close_connection_value = 0;

    WorkerToFabricEdmSender () : worker_sem_addr(nullptr) {}

    WorkerToFabricEdmSender (
        size_t edm_worker_x,
        size_t edm_worker_y,
        std::size_t edm_buffer_base_addr,
        std::size_t num_buffers_per_channel,
        std::size_t edm_l1_sem_id,
        std::size_t edm_connection_handshake_l1_addr,
        std::size_t edm_worker_location_info_addr, // The EDM's location for `EDMChannelWorkerLocationInfo`
        std::size_t buffer_size_bytes,
        std::size_t edm_buffer_index_addr,
        volatile uint32_t * const worker_sem_addr,
        uint32_t local_buffer_index_addr
    ) :
        edm_buffer_addr(get_noc_addr(edm_worker_x, edm_worker_y, edm_buffer_base_addr)),
        edm_semaphore_addr(get_noc_addr(edm_worker_x, edm_worker_y, get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(edm_l1_sem_id))),
        edm_connection_handshake_l1_addr(edm_connection_handshake_l1_addr),
        edm_worker_location_info_addr(edm_worker_location_info_addr),
        edm_buffer_index_addr(edm_buffer_index_addr),
        worker_sem_addr(worker_sem_addr),
        edm_buffer_base_addr(edm_buffer_base_addr),
        num_buffers_per_channel(num_buffers_per_channel),
        last_buffer_index(num_buffers_per_channel - 1),
        edm_l1_sem_addr(get_semaphore<ProgrammableCoreType::ACTIVE_ETH>(edm_l1_sem_id)),
        buffer_size_bytes(buffer_size_bytes),
        buffer_index_ptr(reinterpret_cast<size_t*>(local_buffer_index_addr))
    {
        ASSERT(buffer_size_bytes > 0);
    }

    [[nodiscard]] FORCE_INLINE bool consumer_has_space() const {
        return *this->worker_sem_addr == 1;
    }
    FORCE_INLINE void clear_flow_control_semaphore() const {
        noc_semaphore_set(this->worker_sem_addr, 0);
    }
    FORCE_INLINE void wait_for_empty_write_slot() const {
        noc_semaphore_wait(this->worker_sem_addr, 1);
    }

    FORCE_INLINE void send_payload_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        send_payload_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING>(cb_id, num_pages, page_size);
    }

    // Does not wait for CB. Assumes caller handles CB data availability
    FORCE_INLINE void send_payload_non_blocking(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        send_payload_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::NON_BLOCKING>(cb_id, num_pages, page_size);
    }

    /*
     * No CB
     */
    FORCE_INLINE void send_payload_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::BLOCKING>(source_address, size_bytes);
    }

    /*
     * No CB
     */
    // Does not wait for CB. Assumes caller handles CB data availability
    FORCE_INLINE void send_payload_non_blocking_from_address(uint32_t source_address, size_t size_bytes) {
        send_payload_from_address_impl<ttnn::ccl::EDM_IO_BLOCKING_MODE::NON_BLOCKING>(source_address, size_bytes);
    }

    // Layout
    // |-----------------------|
    // | EDM Handshake         | 16B
    // |-----------------------|
    // | EDM Ack Channel Sync  | 16B
    // |-----------------------|          -
    // | Connection Semaphore  | 16B        |
    // |-----------------------|            |
    // | Buffer Index          | 16B         >- Per Sender Channel (On EDM)
    // |-----------------------|            |
    // | Worker Connection Info| 16B        |worker
    // |-----------------------|          -/
    // |-----------------------|
    //
    static constexpr size_t edm_sender_channel_field_stride_bytes = 16;

    FORCE_INLINE void open() {
        const auto dest_noc_addr_coord_only = this->edm_semaphore_addr & ~(uint64_t)NOC_COORDINATE_MASK;

        const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_buffer_index_addr;
        ASSERT(remote_buffer_index_addr > 0);
        noc_async_read(remote_buffer_index_addr, reinterpret_cast<size_t>(this->buffer_index_ptr), sizeof(uint32_t));

        const uint64_t dest_edm_location_info_addr = dest_noc_addr_coord_only | edm_worker_location_info_addr;
        // TODO: Need to change byte enable to be word enable
        noc_inline_dw_write(dest_edm_location_info_addr, reinterpret_cast<size_t>(worker_sem_addr));
        noc_inline_dw_write(dest_edm_location_info_addr + sizeof(uint32_t), ttnn::ccl::WorkerXY(my_x[0], my_y[0]).to_uint32());

        const uint64_t edm_connection_handshake_noc_addr = dest_noc_addr_coord_only | edm_connection_handshake_l1_addr;
        noc_inline_dw_write(edm_connection_handshake_noc_addr, open_connection_value);
        noc_async_read_barrier();
    }

    FORCE_INLINE void close() {
        const auto dest_noc_addr_coord_only = this->edm_semaphore_addr & ~(uint64_t)NOC_COORDINATE_MASK;

        const uint64_t dest_edm_connection_state_addr = dest_noc_addr_coord_only | edm_connection_handshake_l1_addr;
        noc_inline_dw_write(dest_edm_connection_state_addr, close_connection_value);

        // buffer index stored at location after handshake addr
        const uint64_t remote_buffer_index_addr = dest_noc_addr_coord_only | edm_buffer_index_addr;
        noc_inline_dw_write(remote_buffer_index_addr, *this->buffer_index_ptr);

        noc_async_write_barrier();
    }

    uint64_t edm_buffer_addr;
    uint64_t edm_semaphore_addr;
    size_t edm_connection_handshake_l1_addr;
    size_t edm_worker_location_info_addr;
    size_t edm_buffer_index_addr;
    volatile uint32_t * const worker_sem_addr;
    std::size_t edm_buffer_base_addr;
    std::size_t num_buffers_per_channel;
    std::size_t last_buffer_index;
    std::size_t edm_l1_sem_addr;
    std::size_t buffer_size_bytes;
    std::size_t *buffer_index_ptr;

    private:
    template<ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_from_address_impl(uint32_t source_address, size_t size_bytes) {
        this->clear_flow_control_semaphore();
        uint64_t buffer_address = this->edm_buffer_addr + (*this->buffer_index_ptr * (this->buffer_size_bytes + sizeof(eth_channel_sync_t)));

        ASSERT(size_bytes <= this->buffer_size_bytes);

        /*{ // For debug purposes only. Useful to permanently backup the packet somewhere we can inspect with ttx-status
            uint32_t dram_noc_x = my_y[0] == 1 ? 0 : 0;
            uint32_t dram_noc_y = my_y[0] == 1 ? 0 : 5;
            // noc_inline_dw_write(get_noc_addr(dram_noc_x, dram_noc_y, storage_offset), 0x0F);
            // noc_async_writes_flushed();
            // noc_inline_dw_write(get_noc_addr(dram_noc_x, dram_noc_y, storage_offset  + 4), 0);
            // auto pkthdr_size_words = sizeof(tt::fabric::PacketHeader) >> 2;
            // for (size_t i = 0; i < pkthdr_size_words; i++) {
            //     reinterpret_cast<volatile uint32_t*>(source_address)[pkthdr_size_words - i] =
            //     reinterpret_cast<volatile uint32_t*>(source_address)[pkthdr_size_words - 1 - i];
            // }
            // reinterpret_cast<volatile uint32_t*>(source_address)[0] = 0xc0ffee;
            // DPRINT << "NEXT STORAGE OFF: " << (uint32_t)storage_offset << "\n";
            noc_async_write(source_address, get_noc_addr(dram_noc_x, dram_noc_y, storage_offset), size_bytes);
            storage_offset += size_bytes;
            storage_offset += 64;
            storage_offset = storage_offset & (~0x1F);
        }*/
        ASSERT(tt::fabric::is_valid(*const_cast<tt::fabric::PacketHeader *>(reinterpret_cast<volatile tt::fabric::PacketHeader*>(source_address))));
        send_chunk_from_address<blocking_mode>(source_address, 1, size_bytes, buffer_address);
        noc_semaphore_inc(edm_semaphore_addr, 1);

        *this->buffer_index_ptr = (*this->buffer_index_ptr == this->last_buffer_index) ? 0 : *this->buffer_index_ptr + 1;
    }

    template<ttnn::ccl::EDM_IO_BLOCKING_MODE blocking_mode>
    FORCE_INLINE void send_payload_impl(uint32_t cb_id, uint32_t num_pages, uint32_t page_size) {
        this->clear_flow_control_semaphore();
        uint64_t buffer_address = this->edm_buffer_addr + (*this->buffer_index_ptr * (this->buffer_size_bytes + sizeof(eth_channel_sync_t)));
        ASSERT(num_pages * page_size <= this->buffer_size_bytes);
        send_chunk<blocking_mode>(cb_id, num_pages, page_size, buffer_address);
        noc_semaphore_inc(edm_semaphore_addr, 1);
        *this->buffer_index_ptr = (*this->buffer_index_ptr == this->last_buffer_index) ? 0 : *this->buffer_index_ptr + 1;
    }
};


} // namespace tt::fabric
