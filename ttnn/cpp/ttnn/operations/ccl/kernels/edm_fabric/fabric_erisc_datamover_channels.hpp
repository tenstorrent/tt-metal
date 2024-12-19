// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "debug/dprint.h"
#include "tt_metal/hw/inc/dataflow_api.h"
#include "tt_metal/hw/inc/ethernet/tunneling.h"
#include "tt_metal/hw/inc/risc_attribs.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_types.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"

namespace tt::fabric {
// Increments val and wraps to 0 if it reaches limit
template <typename T, size_t LIMIT>
auto wrap_increment(T val) -> T {
    static_assert(LIMIT != 0, "wrap_increment called with limit of 0; it must be greater than 0");
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr ((LIMIT > 0) && (LIMIT & (LIMIT - 1)) == 0) {
        return (val + 1) & (LIMIT - 1);
    } else {
        return (val == LIMIT - 1) ? 0 : val + 1;
    }
}

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
    EthChannelBuffer() : buffer_size_in_bytes(0), eth_transaction_ack_word_addr(0), max_eth_payload_size_in_bytes(0) {}

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
        eth_transaction_ack_word_addr(eth_transaction_ack_word_addr),
        max_eth_payload_size_in_bytes(buffer_size_in_bytes + sizeof(eth_channel_sync_t)),
        buff_idx(0),
        channel_id(channel_id) {
        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            this->buffer_addresses[i] =
                channel_base_address + i * this->max_eth_payload_size_in_bytes;  //(this->buffer_size_in_bytes);

            uint32_t channel_sync_addr = this->buffer_addresses[i] + buffer_size_in_bytes;
            auto channel_sync_ptr = reinterpret_cast<eth_channel_sync_t *>(channel_sync_addr);

            channel_bytes_sent_addresses[i] =
                reinterpret_cast<volatile tt_l1_ptr size_t *>(&(channel_sync_ptr->bytes_sent));
            channel_bytes_acked_addresses[i] =
                reinterpret_cast<volatile tt_l1_ptr size_t *>(&(channel_sync_ptr->receiver_ack));
            channel_src_id_addresses[i] = reinterpret_cast<volatile tt_l1_ptr size_t *>(&(channel_sync_ptr->src_id));

            ASSERT((uint32_t)channel_bytes_acked_addresses[i] != (uint32_t)(channel_bytes_sent_addresses[i]));
            *(channel_bytes_sent_addresses[i]) = 0;
            *(channel_bytes_acked_addresses[i]) = 0;
            // Note we don't need to overwrite the `channel_src_id_addresses` except for perhapse
            // debug purposes where we may wish to tag this with a special value
        }
    }

    [[nodiscard]] FORCE_INLINE size_t get_current_buffer_address() const {
        return this->buffer_addresses[this->buffer_index()];
    }

    [[nodiscard]] FORCE_INLINE volatile PacketHeader *get_current_packet_header() const {
        return reinterpret_cast<volatile PacketHeader *>(this->buffer_addresses[this->buffer_index()]);
    }

    [[nodiscard]] FORCE_INLINE size_t get_current_payload_size() const {
        return get_current_packet_header()->get_payload_size_including_header();
    }
    [[nodiscard]] FORCE_INLINE size_t get_current_payload_plus_channel_sync_size() const {
        return get_current_packet_header()->get_payload_size_including_header() + sizeof(eth_channel_sync_t);
    }

    // TODO: Split off into two separate functions:
    //       volatile tt_l1_ptr size_t *get_current_bytes_sent_ptr() const
    //       size_t get_current_bytes_sent_address() const
    [[nodiscard]] FORCE_INLINE volatile tt_l1_ptr size_t *get_current_bytes_sent_address() const {
        return this->channel_bytes_sent_addresses[this->buffer_index()];
    }

    [[nodiscard]] FORCE_INLINE volatile tt_l1_ptr size_t *get_current_bytes_acked_address() const {
        return this->channel_bytes_acked_addresses[this->buffer_index()];
    }

    [[nodiscard]] FORCE_INLINE volatile tt_l1_ptr size_t *get_current_src_id_address() const {
        return this->channel_src_id_addresses[this->buffer_index()];
    }

    [[nodiscard]] FORCE_INLINE size_t get_channel_buffer_max_size_in_bytes() const {
        return this->buffer_size_in_bytes;
    }

    // Doesn't return the message size, only the maximum eth payload size
    [[nodiscard]] FORCE_INLINE size_t get_current_max_eth_payload_size() const {
        return this->max_eth_payload_size_in_bytes;
    }

    [[nodiscard]] FORCE_INLINE size_t get_id() const { return this->channel_id; }

    [[nodiscard]] FORCE_INLINE bool eth_is_receiver_channel_send_done() const {
        return *(this->get_current_bytes_sent_address()) == 0;
    }
    [[nodiscard]] FORCE_INLINE bool eth_bytes_are_available_on_channel() const {
        return *(this->get_current_bytes_sent_address()) != 0;
    }
    [[nodiscard]] FORCE_INLINE bool eth_is_receiver_channel_send_acked() const {
        return *(this->get_current_bytes_acked_address()) != 0;
    }
    FORCE_INLINE void eth_clear_sender_channel_ack() const {
        *(this->channel_bytes_acked_addresses[this->buffer_index()]) = 0;
    }

    [[nodiscard]] FORCE_INLINE size_t get_eth_transaction_ack_word_addr() const {
        return this->eth_transaction_ack_word_addr;
    }

    FORCE_INLINE void advance_buffer_index() {
        this->buff_idx = wrap_increment<decltype(this->buff_idx), NUM_BUFFERS>(this->buff_idx);
    }

    [[nodiscard]] FORCE_INLINE bool all_buffers_drained() const {
        bool drained = true;
        for (size_t i = 0; i < NUM_BUFFERS && drained; i++) {
            drained &= *(channel_bytes_sent_addresses[i]) == 0;
        }
        return drained;
    }

   private:
    FORCE_INLINE auto buffer_index() const {
        ASSERT(this->buff_idx < NUM_BUFFERS);
        return buff_idx;
    }

    std::array<size_t, NUM_BUFFERS> buffer_addresses;
    std::array<volatile tt_l1_ptr size_t *, NUM_BUFFERS> channel_bytes_sent_addresses;
    std::array<volatile tt_l1_ptr size_t *, NUM_BUFFERS> channel_bytes_acked_addresses;
    std::array<volatile tt_l1_ptr size_t *, NUM_BUFFERS> channel_src_id_addresses;

    // header + payload regions only
    const std::size_t buffer_size_in_bytes;
    // Includes header + payload + channel_sync
    const std::size_t eth_transaction_ack_word_addr;
    const std::size_t max_eth_payload_size_in_bytes;
    uint8_t buff_idx;
    uint8_t channel_id;
};

struct EdmChannelWorkerInterface {
    EdmChannelWorkerInterface() :
        worker_location_info_ptr(nullptr), local_semaphore_address(nullptr), connection_live_semaphore(nullptr) {}
    EdmChannelWorkerInterface(
        // TODO: PERF: See if we can make this non-volatile and then only
        // mark it volatile when we know we need to reload it (i.e. after we receive a
        // "done" message from sender)
        // Have a volatile update function that only triggers after reading the volatile
        // completion field so that way we don't have to do a volatile read for every
        // packet... Then we'll also be able to cache the uint64_t addr of the worker
        // semaphore directly (saving on regenerating it each time)
        volatile EDMChannelWorkerLocationInfo *worker_location_info_ptr,
        volatile tt_l1_ptr uint32_t *const local_semaphore_address,
        volatile tt_l1_ptr uint32_t *const connection_live_semaphore) :
        worker_location_info_ptr(worker_location_info_ptr),
        local_semaphore_address(local_semaphore_address),
        connection_live_semaphore(connection_live_semaphore) {}

    // Flow control methods
    //
    [[nodiscard]] FORCE_INLINE auto local_semaphore_value() const { return *local_semaphore_address; }

    [[nodiscard]] FORCE_INLINE bool has_payload() { return *local_semaphore_address != 0; }

    FORCE_INLINE void clear_local_semaphore() { noc_semaphore_set(local_semaphore_address, 0); }

    [[nodiscard]] FORCE_INLINE uint32_t get_worker_semaphore_address() const {
        return worker_location_info_ptr->worker_semaphore_address;
    }

    void increment_worker_semaphore() const {
        auto const &worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_semaphore_address);

        DPRINT << "EDM ntf wrkr sem @" << (uint64_t)worker_semaphore_address << "\n";
        noc_semaphore_inc(worker_semaphore_address, 1);
    }

    // Connection management methods
    //
    FORCE_INLINE void teardown_connection() const { increment_worker_semaphore(); }

    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const { return *connection_live_semaphore == 0; }

    [[nodiscard]] FORCE_INLINE bool connection_is_live() const { return *connection_live_semaphore == 1; }

    volatile EDMChannelWorkerLocationInfo *worker_location_info_ptr;
    volatile tt_l1_ptr uint32_t *const local_semaphore_address;
    volatile tt_l1_ptr uint32_t *const connection_live_semaphore;
};

}  // namespace tt::fabric
