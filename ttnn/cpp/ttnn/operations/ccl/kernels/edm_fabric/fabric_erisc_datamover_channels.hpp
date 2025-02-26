// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "debug/dprint.h"
#include "dataflow_api.h"
#include "tt_metal/hw/inc/ethernet/tunneling.h"
#include "tt_metal/hw/inc/utils/utils.h"
#include "risc_attribs.h"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_flow_control_helpers.hpp"
namespace tt::fabric {


template <typename T>
FORCE_INLINE auto wrap_increment(T val, size_t max) {
    return (val == max - 1) ? 0 : val + 1;
}

template <uint8_t NUM_BUFFERS>
class EthChannelBuffer final {
   public:
    // The channel structure is as follows:
    //              &header->  |----------------| channel_base_address
    //                         |    header      |
    //             &payload->  |----------------|
    //                         |                |
    //                         |    payload     |
    //                         |                |
    //        &channel_sync->  |----------------|
    //                         |  channel_sync  |
    //                         ------------------
    EthChannelBuffer() : buffer_size_in_bytes(0), max_eth_payload_size_in_bytes(0) {}

    /*
     * Expected that *buffer_index_ptr is initialized outside of this object
     */
    EthChannelBuffer(
        size_t channel_base_address,
        size_t buffer_size_bytes,
        size_t header_size_bytes,
        size_t eth_transaction_ack_word_addr,  // Assume for receiver channel, this address points to a chunk of memory
                                               // that can fit 2 eth_channel_syncs cfor ack
        uint8_t channel_id) :
        buffer_size_in_bytes(buffer_size_bytes),
        max_eth_payload_size_in_bytes(buffer_size_in_bytes + sizeof(eth_channel_sync_t)),
        channel_id(channel_id) {
        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            this->buffer_addresses[i] =
                channel_base_address + i * this->max_eth_payload_size_in_bytes;
        }
    }

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(BufferIndex const& buffer_index) const {
        return this->buffer_addresses[buffer_index];
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE volatile T *get_packet_header(BufferIndex const& buffer_index) const {
        return reinterpret_cast<volatile T *>(this->buffer_addresses[buffer_index]);
    }

    template <typename T>
    [[nodiscard]] FORCE_INLINE size_t get_payload_size(BufferIndex const& buffer_index) const {
        return get_packet_header<T>(buffer_index)->get_payload_size_including_header();
    }
    [[nodiscard]] FORCE_INLINE size_t get_channel_buffer_max_size_in_bytes(BufferIndex const& buffer_index) const {
        return this->buffer_size_in_bytes;
    }

    // Doesn't return the message size, only the maximum eth payload size
    [[nodiscard]] FORCE_INLINE size_t get_max_eth_payload_size() const {
        return this->max_eth_payload_size_in_bytes;
    }

    [[nodiscard]] FORCE_INLINE size_t get_id() const { return this->channel_id; }

    [[nodiscard]] FORCE_INLINE bool eth_is_acked_or_completed(BufferIndex const& buffer_index) const {
        return eth_is_receiver_channel_send_acked(buffer_index) || eth_is_receiver_channel_send_done(buffer_index);
    }


    FORCE_INLINE bool needs_to_send_channel_sync() const {
        return this->need_to_send_channel_sync;
    }

    FORCE_INLINE void set_need_to_send_channel_sync(bool need_to_send_channel_sync) {
        this->need_to_send_channel_sync = need_to_send_channel_sync;
    }

    FORCE_INLINE void clear_need_to_send_channel_sync() {
        this->need_to_send_channel_sync = false;
    }

   private:

    std::array<size_t, NUM_BUFFERS> buffer_addresses;

    // header + payload regions only
    const std::size_t buffer_size_in_bytes;
    // Includes header + payload + channel_sync
    const std::size_t max_eth_payload_size_in_bytes;
    uint8_t channel_id;
};


template <uint8_t NUM_BUFFERS, bool USE_STATEFUL_NOC_API = false, uint8_t PTR_UPDATE_NOC_CMD_BUF = write_at_cmd_buf, uint8_t PTR_UPDATE_NOC_INDEX = noc_index>
struct EdmChannelWorkerInterface {
    EdmChannelWorkerInterface() :
        worker_location_info_ptr(nullptr),
        cached_worker_semaphore_address(0),
        remote_producer_wrptr(nullptr),
        connection_live_semaphore(nullptr),
        local_wrptr(),
        local_ackptr(),
        local_rdptr() {}
    EdmChannelWorkerInterface(
        // TODO: PERF: See if we can make this non-volatile and then only
        // mark it volatile when we know we need to reload it (i.e. after we receive a
        // "done" message from sender)
        // Have a volatile update function that only triggers after reading the volatile
        // completion field so that way we don't have to do a volatile read for every
        // packet... Then we'll also be able to cache the uint64_t addr of the worker
        // semaphore directly (saving on regenerating it each time)
        volatile EDMChannelWorkerLocationInfo *worker_location_info_ptr,
        volatile tt_l1_ptr uint32_t *const remote_producer_wrptr,
        volatile tt_l1_ptr uint32_t *const connection_live_semaphore) :
        worker_location_info_ptr(worker_location_info_ptr),
        cached_worker_semaphore_address(0),
        remote_producer_wrptr(remote_producer_wrptr),
        connection_live_semaphore(connection_live_semaphore),
        local_wrptr(),
        local_ackptr(),
        local_rdptr() {
        DPRINT << "EDM  my_x: " << (uint32_t)my_x[0] << ", my_y: " << (uint32_t)my_y[0] << " rdptr set to 0 at " << (uint32_t)(void*)&(worker_location_info_ptr->edm_rdptr) << "\n";
        *reinterpret_cast<volatile uint32_t*>(&(worker_location_info_ptr->edm_rdptr)) = 0;
        }

    // Flow control methods
    //
    // local_wrptr trails from_remote_wrptr
    // we have new data if they aren't equal
    [[nodiscard]] FORCE_INLINE bool has_unsent_payload() {
        return local_wrptr.get_ptr() != *remote_producer_wrptr;
    }
    [[nodiscard]] FORCE_INLINE bool has_unacked_sends() {
        return local_ackptr.get_ptr() != local_wrptr.get_ptr();
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_worker_semaphore_address() const {
        return cached_worker_semaphore_address & 0xFFFFFFFF;
    }

    FORCE_INLINE void update_worker_copy_of_read_ptr(BufferPtr new_ptr_val) {
        if constexpr (USE_STATEFUL_NOC_API) {
            // for producer-sender ack path use the other NoC, so they can be made stateful
            noc_inline_dw_write_with_state(new_ptr_val, PTR_UPDATE_NOC_CMD_BUF, PTR_UPDATE_NOC_INDEX);
        } else {
            noc_inline_dw_write(this->cached_worker_semaphore_address, new_ptr_val, 0xF, PTR_UPDATE_NOC_CMD_BUF, PTR_UPDATE_NOC_INDEX);
        }
    }

    // Connection management methods
    //
    FORCE_INLINE void teardown_connection(uint32_t last_edm_rdptr_value) const {
        auto const &worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_teardown_semaphore_address);

        // Set connection to unused so it's available for next worker
        *this->connection_live_semaphore = tt::fabric::EdmToEdmSender<0>::unused_connection_value;

        *reinterpret_cast<volatile uint32_t*>(&(worker_location_info_ptr->edm_rdptr)) = last_edm_rdptr_value;

        noc_semaphore_inc(worker_semaphore_address, 1);
    }

    FORCE_INLINE void cache_producer_noc_addr() {
        auto const &worker_info = *worker_location_info_ptr;
        // for producer-sender ack path use the other NoC, so they can be made stateful
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x,
            (uint32_t)worker_info.worker_xy.y,
            worker_info.worker_semaphore_address,
            PTR_UPDATE_NOC_INDEX);
        this->cached_worker_semaphore_address = worker_semaphore_address;
        if constexpr (USE_STATEFUL_NOC_API) {
            noc_inline_dw_write_set_state(worker_semaphore_address, 0xF, PTR_UPDATE_NOC_CMD_BUF);
        }
    }

    FORCE_INLINE bool all_eth_packets_acked() const {
        return this->local_ackptr.is_caught_up_to(this->local_wrptr);
    }
    FORCE_INLINE bool all_eth_packets_completed() const {
        return this->local_rdptr.is_caught_up_to(this->local_wrptr);
    }

    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const { return *connection_live_semaphore == tt::fabric::EdmToEdmSender<0>::close_connection_request_value; }
    [[nodiscard]] FORCE_INLINE bool connection_is_live() const { return *connection_live_semaphore == tt::fabric::EdmToEdmSender<0>::open_connection_value; }

    volatile EDMChannelWorkerLocationInfo *worker_location_info_ptr;
    uint64_t cached_worker_semaphore_address = 0;
    volatile tt_l1_ptr uint32_t *const remote_producer_wrptr;
    volatile tt_l1_ptr uint32_t *const connection_live_semaphore;

    ChannelBufferPointer<NUM_BUFFERS> local_wrptr;
    ChannelBufferPointer<NUM_BUFFERS> local_ackptr;
    ChannelBufferPointer<NUM_BUFFERS> local_rdptr; // also used as completion_ptr
};


}  // namespace tt::fabric
