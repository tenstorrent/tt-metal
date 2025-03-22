// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/hw/inc/utils/utils.h"
#include <cstdint>
#include <cstddef>
#include <tuple>
#include <array>
#include "debug/dprint.h"

// Increments val and wraps to 0 if it reaches limit
template <size_t LIMIT = 0, typename T>
FORCE_INLINE auto wrap_increment(T val) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
    if constexpr (LIMIT == 1) {
        return val;
    } else if constexpr (LIMIT == 2) {
        return 1 - val;
    } else if constexpr (is_pow2) {
        return (val + 1) & (static_cast<T>(LIMIT - 1));
    } else {
        return (val == static_cast<T>(LIMIT - 1)) ? static_cast<T>(0) : static_cast<T>(val + 1);
    }
}

template <typename T>
FORCE_INLINE auto wrap_increment(T val, T size) -> T {
    return (val == static_cast<T>(size - 1)) ? static_cast<T>(0) : static_cast<T>(val + 1);
}

template <size_t LIMIT, typename T>
FORCE_INLINE auto wrap_increment_n(T val, uint8_t increment) -> T {
    constexpr bool is_pow2 = LIMIT != 0 && is_power_of_2(LIMIT);
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

template <typename T, typename Parameter>
class NamedType {
public:
    FORCE_INLINE explicit NamedType(const T& value) : value_(value) {}
    FORCE_INLINE explicit NamedType(T&& value) : value_(std::move(value)) {}
    FORCE_INLINE NamedType<T, Parameter>& operator=(const NamedType<T, Parameter>& rhs) = default;
    FORCE_INLINE T& get() { return value_; }
    FORCE_INLINE const T& get() const { return value_; }
    FORCE_INLINE operator T() const { return value_; }
    FORCE_INLINE operator T&() { return value_; }

private:
    T value_;
};

using TransactionId = NamedType<uint8_t, struct TransactionIdType>;

template <uint8_t NUM_TRIDS>
struct TransactionIdCounter {
    FORCE_INLINE void increment() { this->next_trid = wrap_increment<NUM_TRIDS>(this->next_trid); }

    FORCE_INLINE uint8_t get() const { return this->next_trid; }

private:
    uint8_t next_trid = 0;
};

template <size_t NUM_TRIDS>
struct TransactionIdTracker {
    static constexpr uint8_t INVALID_TRID = NUM_TRIDS;
    static constexpr bool N_TRIDS_IS_POW2 = is_power_of_2(NUM_TRIDS);
    static_assert(N_TRIDS_IS_POW2, "NUM_TRIDS must be a power of 2");

    TransactionIdTracker(uint32_t cb_id) :
        cb_interface(get_local_cb_interface(cb_id)),
        trid_counter({}),
        cb_id(cb_id),
        open_trids(0),
        oldest_trid(0),
        next_trid(0) {
        // Check for invalid usage
    }

    // Sent but not complete
    FORCE_INLINE bool oldest_write_trid_sent() {
        return ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, oldest_trid);
    }
    // sent and complete
    FORCE_INLINE bool oldest_write_trid_flushed() {
        return ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, oldest_trid);
    }
    FORCE_INLINE bool oldest_read_trid_flushed() {
        return ncrisc_noc_read_with_transaction_id_flushed(noc_index, oldest_trid);
    }

    FORCE_INLINE bool next_cb_write_slot_is_available(size_t num_pages) {
        if constexpr (NUM_TRIDS != 1) {
            return cb_pages_reservable_at_back(cb_id, total_pages_open + num_pages);
        } else {
            return cb_pages_reservable_at_back(cb_id, num_pages);
        }
    }

    FORCE_INLINE bool backpressured() { return open_trids == NUM_TRIDS; }

    FORCE_INLINE bool next_cb_read_slot_is_available(size_t num_pages) {
        if constexpr (NUM_TRIDS != 1) {
            return !this->backpressured() && cb_pages_available_at_front(cb_id, total_pages_open + num_pages);
        } else {
            return !this->backpressured() && cb_pages_available_at_front(cb_id, num_pages);
        }
    }

    FORCE_INLINE std::tuple<size_t, TransactionId> get_next_cb_write_slot(size_t num_pages) {
        auto [chunk_start_address, next_available_trid] = update_for_next_cb_slots(num_pages, cb_interface.fifo_rd_ptr);

        if constexpr (NUM_TRIDS != 1) {
            cb_reserve_back(cb_id, total_pages_open);
        } else {
            cb_reserve_back(cb_id, num_pages);
        }
        if constexpr (NUM_TRIDS != 1) {
            next_trid = TransactionId{wrap_increment<NUM_TRIDS>(next_trid.get())};
        }
        return {chunk_start_address, next_available_trid};
    }
    // How does this work with wrapping?
    // Do I need to know the CB ID too?
    FORCE_INLINE std::tuple<size_t, TransactionId> get_next_cb_read_slot(size_t num_pages) {
        auto [chunk_start_address, next_available_trid] = update_for_next_cb_slots(num_pages, cb_interface.fifo_rd_ptr);

        cb_wait_front(cb_id, total_pages_open);
        if constexpr (NUM_TRIDS != 1) {
            next_trid = TransactionId{wrap_increment<NUM_TRIDS>(next_trid.get())};
        }
        return {chunk_start_address, next_available_trid};
    }

    // Returns the chunk start address
    FORCE_INLINE std::tuple<size_t, TransactionId> update_for_next_cb_slots(size_t num_pages, size_t base_ptr) {
        size_t chunk_start_address = 0;
        if constexpr (NUM_TRIDS != 1) {
            auto chunk_offset_from_base_ptr = total_pages_open * cb_interface.fifo_page_size;

            chunk_start_address = base_ptr + chunk_offset_from_base_ptr;
            bool wraps = chunk_start_address >= cb_interface.fifo_limit;
            if (wraps) {
                ASSERT(chunk_start_address == cb_interface.fifo_limit);
                chunk_start_address -= (wraps * cb_interface.fifo_size);
            }
            open_trids++;
        } else {
            chunk_start_address = base_ptr;
            open_trids = 1;
        }
        auto next_available_trid = next_trid;
        pages_per_trid[next_available_trid] = num_pages;
        if constexpr (NUM_TRIDS != 1) {
            total_pages_open += num_pages;
        }
        return {chunk_start_address, next_available_trid};
    }

    FORCE_INLINE bool has_unflushed_trid() { return open_trids > 0; }

    FORCE_INLINE void push_pages_for_oldest_trid() {
        auto pages_to_push = pages_per_trid[oldest_trid];
        cb_push_back(cb_id, pages_to_push);
        advance_pages_for_oldest_trid_update_counters(pages_to_push);
    }

    FORCE_INLINE void pop_pages_for_oldest_trid() {
        auto pages_to_pop = pages_per_trid[oldest_trid];
        cb_pop_front(cb_id, pages_to_pop);
        advance_pages_for_oldest_trid_update_counters(pages_to_pop);
    }

    FORCE_INLINE void write_barrier() {
        while (open_trids > 0) {
            while (!oldest_write_trid_flushed()) {
            }
            this->pop_pages_for_oldest_trid();
        }
    }

private:
    FORCE_INLINE void advance_pages_for_oldest_trid_update_counters(size_t pages_to_advance) {
        if constexpr (NUM_TRIDS != 1) {
            total_pages_open -= pages_to_advance;
        }
        ASSERT(total_pages_open >= 0);
        if constexpr (NUM_TRIDS != 1) {
            oldest_trid = TransactionId{wrap_increment<NUM_TRIDS>(oldest_trid.get())};
        }
        open_trids--;
    }

    std::array<size_t, NUM_TRIDS> pages_per_trid;
    LocalCBInterface& cb_interface;
    uint16_t total_pages_open = 0;
    // uint8_t cb_n_chunks = 0;

    // Advances with oldest_trid
    TransactionIdCounter<NUM_TRIDS> trid_counter;
    uint8_t cb_id = 0;
    uint8_t open_trids = 0;

    // TODO: cleanup - only used for when both params are pow2, else above are used.
    TransactionId oldest_trid = 0;
    TransactionId next_trid = 0;
};
