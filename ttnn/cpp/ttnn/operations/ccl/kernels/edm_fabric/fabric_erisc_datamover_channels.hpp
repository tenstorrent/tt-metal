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

namespace tt::fabric {

template <typename T, typename Parameter>
class NamedType
{
public:
    explicit NamedType(T const& value) : value_(value) {}
    explicit NamedType(T&& value) : value_(std::move(value)) {}
    NamedType<T,Parameter> &operator=(NamedType<T,Parameter> const& rhs) = default;
    T& get() { return value_; }
    T const& get() const {return value_; }
    operator T() const { return value_; }
    operator T&() { return value_; }
private:
    T value_;
};

using BufferIndex = NamedType<uint8_t, struct BufferIndexType>;
using BufferPtr = NamedType<uint8_t, struct BufferPtrType>;


// Increments val and wraps to 0 if it reaches limit
template <size_t LIMIT, typename T>
auto wrap_increment(T val) -> T {
    static_assert(LIMIT != 0, "wrap_increment called with limit of 0; it must be greater than 0");
    constexpr bool is_pow2 = is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + 1) & (LIMIT - 1);
    } else {
        return (val == static_cast<T>(LIMIT - 1)) ? static_cast<T>(0) : static_cast<T>(val + 1);
    }
}
template <size_t LIMIT, typename T>
auto wrap_increment_n(T val, uint8_t increment) -> T {
    static_assert(LIMIT != 0, "wrap_increment called with limit of 0; it must be greater than 0");
    constexpr bool is_pow2 = is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + increment) & (LIMIT - 1);
    } else {
        T new_unadjusted_val = val + increment;
        bool wraps = new_unadjusted_val >= LIMIT;
        return wraps ? static_cast<T>(new_unadjusted_val - LIMIT) : static_cast<T>(new_unadjusted_val);
    }
}

template <uint8_t NUM_BUFFERS>
auto normalize_ptr(BufferPtr ptr) -> BufferIndex {
    static_assert(NUM_BUFFERS != 0, "normalize_ptr called with NUM_BUFFERS of 0; it must be greater than 0");
    constexpr bool is_size_pow2 = (NUM_BUFFERS & (NUM_BUFFERS - 1)) == 0;
    constexpr bool is_size_2 = NUM_BUFFERS == 2;
    constexpr bool is_size_1 = NUM_BUFFERS == 1;
    constexpr uint8_t wrap_mask = NUM_BUFFERS - 1;
    if constexpr (is_size_pow2) {
        return BufferIndex{ptr & wrap_mask};
    } else if constexpr (is_size_2) {
        return BufferIndex{(uint8_t)1 - ptr};
    } else if constexpr (is_size_1) {
        return BufferIndex{0};
    } else {
        // note it may make sense to calculate this only when we increment
        // which will save calculations overall (but may add register pressure)
        // and introduce undesirable loads
        bool normalize = ptr >= NUM_BUFFERS;
        uint8_t normalized_ptr = ptr.get() - static_cast<uint8_t>(normalize * NUM_BUFFERS);
        ASSERT(normalized_ptr < NUM_BUFFERS);
        return BufferIndex{normalized_ptr};
    }
}


template <uint8_t NUM_BUFFERS>
class ChannelBufferPointer {
    static_assert(NUM_BUFFERS <= std::numeric_limits<uint8_t>::max() / 2, "NUM_BUFFERS must be less than or half of std::numeric_limits<uint8_t>::max() due to the internal implementation");
    public:
    static constexpr bool is_size_pow2 = (NUM_BUFFERS & (NUM_BUFFERS - 1)) == 0;
    static constexpr bool is_size_2 = NUM_BUFFERS == 2;
    static constexpr bool is_size_1 = NUM_BUFFERS == 1;
    static constexpr uint8_t ptr_wrap_size = 2 * NUM_BUFFERS;

    // Only to use if is_size_pow2
    static constexpr uint8_t ptr_wrap_mask = (2 * NUM_BUFFERS) - 1;
    static constexpr uint8_t buffer_wrap_mask = NUM_BUFFERS - 1;
    ChannelBufferPointer() : ptr(0) {}
    /*
     * Returns the "raw" pointer - not usable to index the buffer channel
     */
    BufferPtr get_ptr() const {
        return this->ptr;
    }

    bool is_caught_up_to(ChannelBufferPointer<NUM_BUFFERS> const& leading_ptr) const {
        return this->is_caught_up_to(leading_ptr.get_ptr());
    }
    uint8_t distance_behind(ChannelBufferPointer<NUM_BUFFERS> const& leading_ptr) const {
        return this->distance_behind(leading_ptr.get_ptr());
    }

    /*
     * Returns the buffer index pointer which is usable to index into the buffer memory
     */
    BufferIndex get_buffer_index() const {
        return BufferIndex{normalize_ptr<NUM_BUFFERS>(this->ptr)};
    }

    void increment_n(uint8_t n) {
        this->ptr = BufferPtr{wrap_increment_n<2*NUM_BUFFERS>(this->ptr.get(), n)};
    }
    void increment() {
        this->ptr = wrap_increment<2*NUM_BUFFERS>(this->ptr);
    }

    private:
    // Make these private to make sure caller doesn't accidentally mix two pointers pointing to
    // different sized channels
    bool is_caught_up_to(BufferPtr const& leading_ptr) const {
        return this->get_ptr() == leading_ptr;
    }
    uint8_t distance_behind(BufferPtr const& leading_ptr) const {
        bool leading_gte_trailing_ptr = leading_ptr >= this->ptr;
        if constexpr (is_size_pow2) {
            return (leading_ptr - this->ptr) & ptr_wrap_mask;
        } else {
            return leading_gte_trailing_ptr ?
                leading_ptr - this->ptr :
                ptr_wrap_size - (this->ptr - leading_ptr);
        }
    }
    BufferPtr ptr = BufferPtr{0};
};


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
        channel_id(channel_id) {
        for (uint8_t i = 0; i < NUM_BUFFERS; i++) {
            this->buffer_addresses[i] =
                channel_base_address + i * this->max_eth_payload_size_in_bytes;

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
            *(channel_src_id_addresses[i]) = 0x1c0ffee1;
            (channel_src_id_addresses[i])[1] = 0x1c0ffee2;

            // Note we don't need to overwrite the `channel_src_id_addresses` except for perhapse
            // debug purposes where we may wish to tag this with a special value
        }
    }

    [[nodiscard]] FORCE_INLINE size_t get_buffer_address(BufferIndex const& buffer_index) const {
        return this->buffer_addresses[buffer_index];
    }

    [[nodiscard]] FORCE_INLINE volatile PacketHeader *get_packet_header(BufferIndex const& buffer_index) const {
        return reinterpret_cast<volatile PacketHeader *>(this->buffer_addresses[buffer_index]);
    }

    [[nodiscard]] FORCE_INLINE size_t get_payload_size(BufferIndex const& buffer_index) const {
        return get_packet_header(buffer_index)->get_payload_size_including_header();
    }
    [[nodiscard]] FORCE_INLINE size_t get_payload_plus_channel_sync_size(BufferIndex const& buffer_index) const {
        return get_packet_header(buffer_index)->get_payload_size_including_header() + sizeof(eth_channel_sync_t);
    }

    [[nodiscard]] FORCE_INLINE volatile tt_l1_ptr size_t *get_bytes_sent_address(BufferIndex const& buffer_index) const {
        return this->channel_bytes_sent_addresses[buffer_index];
    }

    [[nodiscard]] FORCE_INLINE volatile tt_l1_ptr size_t *get_bytes_acked_address(BufferIndex const& buffer_index) const {
        return this->channel_bytes_acked_addresses[buffer_index];
    }

    [[nodiscard]] FORCE_INLINE volatile tt_l1_ptr size_t *get_src_id_address(BufferIndex const& buffer_index) const {
        return this->channel_src_id_addresses[buffer_index];
    }

    [[nodiscard]] FORCE_INLINE size_t get_channel_buffer_max_size_in_bytes(BufferIndex const& buffer_index) const {
        return this->buffer_size_in_bytes;
    }

    // Doesn't return the message size, only the maximum eth payload size
    [[nodiscard]] FORCE_INLINE size_t get_max_eth_payload_size() const {
        return this->max_eth_payload_size_in_bytes;
    }

    [[nodiscard]] FORCE_INLINE size_t get_id() const { return this->channel_id; }

    [[nodiscard]] FORCE_INLINE bool eth_is_receiver_channel_send_done(BufferIndex const& buffer_index) const {
        return *(this->get_bytes_sent_address(buffer_index)) == 0;
    }
    [[nodiscard]] FORCE_INLINE bool eth_bytes_are_available_on_channel(BufferIndex const& buffer_index) const {
        return *(this->get_bytes_sent_address(buffer_index)) != 0;
    }
    [[nodiscard]] FORCE_INLINE bool eth_is_receiver_channel_send_acked(BufferIndex const& buffer_index) const {
        return *(this->get_bytes_acked_address(buffer_index)) != 0;
    }
    FORCE_INLINE void eth_clear_sender_channel_ack(BufferIndex const& buffer_index) const {
        *(this->channel_bytes_acked_addresses[buffer_index]) = 0;
    }
    [[nodiscard]] FORCE_INLINE bool eth_is_acked_or_completed(BufferIndex const& buffer_index) const {
        return eth_is_receiver_channel_send_acked(buffer_index) || eth_is_receiver_channel_send_done(buffer_index);
    }

    [[nodiscard]] FORCE_INLINE size_t get_eth_transaction_ack_word_addr() const {
        return this->eth_transaction_ack_word_addr;
    }

    [[nodiscard]] FORCE_INLINE bool all_buffers_drained() const {
        bool drained = true;
        for (size_t i = 0; i < NUM_BUFFERS && drained; i++) {
            drained &= *(channel_bytes_sent_addresses[i]) == 0;
        }
        return drained;
    }

    bool needs_to_send_channel_sync() const {
        return this->need_to_send_channel_sync;
    }

    void set_need_to_send_channel_sync(bool need_to_send_channel_sync) {
        this->need_to_send_channel_sync = need_to_send_channel_sync;
    }

    void clear_need_to_send_channel_sync() {
        this->need_to_send_channel_sync = false;
    }

   private:

    std::array<size_t, NUM_BUFFERS> buffer_addresses;
    std::array<volatile tt_l1_ptr size_t *, NUM_BUFFERS> channel_bytes_sent_addresses;
    std::array<volatile tt_l1_ptr size_t *, NUM_BUFFERS> channel_bytes_acked_addresses;
    std::array<volatile tt_l1_ptr size_t *, NUM_BUFFERS> channel_src_id_addresses;

    // header + payload regions only
    const std::size_t buffer_size_in_bytes;
    // Includes header + payload + channel_sync
    const std::size_t eth_transaction_ack_word_addr;
    const std::size_t max_eth_payload_size_in_bytes;
    uint8_t channel_id;
};


template <uint8_t NUM_BUFFERS>
struct EdmChannelWorkerInterface {
    EdmChannelWorkerInterface() :
        worker_location_info_ptr(nullptr),
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
        return worker_location_info_ptr->worker_semaphore_address;
    }

    FORCE_INLINE void update_worker_copy_of_read_ptr() {
        auto const &worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_semaphore_address);
        noc_inline_dw_write(worker_semaphore_address, local_ackptr.get_ptr());
    }

    // Connection management methods
    //
    FORCE_INLINE void teardown_connection(uint32_t last_edm_rdptr_value) const {
        auto const &worker_info = *worker_location_info_ptr;
        uint64_t worker_semaphore_address = get_noc_addr(
            (uint32_t)worker_info.worker_xy.x, (uint32_t)worker_info.worker_xy.y, worker_info.worker_teardown_semaphore_address);

        // Set connection to unused so it's available for next worker
        *this->connection_live_semaphore = tt::fabric::WorkerToFabricEdmSender::unused_connection_value;

        *reinterpret_cast<volatile uint32_t*>(&(worker_location_info_ptr->edm_rdptr)) = last_edm_rdptr_value;

        noc_semaphore_inc(worker_semaphore_address, 1);
    }

    bool all_eth_packets_acked() const {
        return this->local_ackptr.is_caught_up_to(this->local_wrptr);
    }
    bool all_eth_packets_completed() const {
        return this->local_rdptr.is_caught_up_to(this->local_wrptr);
    }

    // Call to keep the connection flow control info fresh with worker.
    void propagate_ackptr_to_connection_info() {
        worker_location_info_ptr->edm_rdptr = local_ackptr.get_ptr();
    }

    [[nodiscard]] FORCE_INLINE bool has_worker_teardown_request() const { return *connection_live_semaphore == tt::fabric::WorkerToFabricEdmSender::close_connection_request_value; }
    [[nodiscard]] FORCE_INLINE bool connection_is_live() const { return *connection_live_semaphore == tt::fabric::WorkerToFabricEdmSender::open_connection_value; }

    volatile EDMChannelWorkerLocationInfo *worker_location_info_ptr;
    volatile tt_l1_ptr uint32_t *const remote_producer_wrptr;
    volatile tt_l1_ptr uint32_t *const connection_live_semaphore;

    ChannelBufferPointer<NUM_BUFFERS> local_wrptr;
    ChannelBufferPointer<NUM_BUFFERS> local_ackptr;
    ChannelBufferPointer<NUM_BUFFERS> local_rdptr; // also used as completion_ptr
};


}  // namespace tt::fabric
