// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file slotmap.hpp
 * @brief Implementation of the slotmap data structure
 *
 * This header file provides the implementation of a slotmap, which is a
 * high-performance container that offers constant-time insertion, deletion,
 * and access operations. The slotmap maintains stable references to its
 * elements, making it ideal for scenarios where elements need to be
 * frequently added, removed, or accessed without invalidating existing
 * references. It is great for storing collection of objects with no clear
 * ownership.
 */

#pragma once

#include <cassert>
#include <climits>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <tt_metal/common/assert.hpp>

namespace tt::stl {

template <typename T, size_t INDEX_BITS>
struct Key {
   private:
    static_assert(INDEX_BITS < sizeof(T) * CHAR_BIT, "Index bits must be less than the size of the value type");
    static constexpr size_t VERSION_BITS = sizeof(T) * CHAR_BIT - INDEX_BITS;

    // E.g. if sizeof(T) == 2, INDEX_BITS == 10, then
    // INDEX_MASK   = 0b1111111111000000
    // VERSION_MASK = 0b0000000000111111
    static constexpr T INDEX_MASK = ((T(1) << INDEX_BITS) - 1) << VERSION_BITS;
    static constexpr T VERSION_MASK = ~INDEX_MASK;

    T value = 0;

    template <typename KeyT, typename U>
    friend class SlotMap;

   public:
    using value_type = T;
    static constexpr T max_index = (T(1) << INDEX_BITS) - 1;

    Key() = default;

    // Always sets the LSB to 1 to ensure the key is valid.
    Key(T idx, T ver) : value((idx << VERSION_BITS) | ver | 1) {
        // assert bit counts
        assert(((idx << VERSION_BITS) & INDEX_MASK) == idx << VERSION_BITS);
        assert((ver & VERSION_MASK) == ver);
    }

    explicit Key(T full_value) : value(full_value) {
        assert(value & 1);  // Ensure the key is valid
    }

    T index() const { return value >> VERSION_BITS; }
    T version() const { return value & VERSION_MASK; }

    bool operator<=>(const Key& other) const = default;
};

#define MAKE_SLOTMAP_KEY(NAME, T, N)     \
    struct NAME : ::tt::stl::Key<T, N> { \
        using Key::Key;                  \
    };

template <typename KeyT, typename T>
class SlotMap {
   public:
    using index_type = KeyT::value_type;
    using version_type = KeyT::value_type;
    using value_type = T;

   private:
    class Slot {
       public:
        // Even for vacant, odd for occupied.
        version_type version;

        union {
            T value;
            index_type next_free;
        };

        Slot() : version(0), next_free(0) {}

        template <typename... Args>
        Slot(version_type ver, Args&&... args) : version(ver), value(std::forward<Args>(args)...) {}

        Slot(const Slot& other) = delete;
        Slot& operator=(const Slot& other) = delete;

        Slot(Slot&& other) noexcept : version(other.version) {
            if (other.occupied()) {
                new (&value) T(std::move(other.value));
            } else {
                next_free = other.next_free;
            }
        }

        Slot& operator=(Slot&& other) noexcept {
            if (this != &other) {
                this->~Slot();
                version = other.version;
                if (other.occupied()) {
                    new (&value) T(std::move(other.value));
                } else {
                    next_free = other.next_free;
                }
            }
            return *this;
        }

        [[nodiscard]] bool occupied() const noexcept { return version & 1; }

        ~Slot()
            requires(!std::is_trivially_destructible_v<T>)
        {
            if (occupied()) {
                value.~T();
            }
        }

        ~Slot()
            requires std::is_trivially_destructible_v<T>
        = default;
    };

   private:
    static constexpr index_type max_index = KeyT::max_index;
    static constexpr index_type invalid_index = std::numeric_limits<index_type>::max();
    std::vector<Slot> slots;

    // Index of the first free slot.
    uint32_t free_head;

    // Number of occupied slots.
    uint32_t num_elems;

   public:
    /**
     * @brief Constructs a SlotMap with an optional initial capacity.
     * @param capacity The initial capacity of the SlotMap.
     */
    SlotMap(size_t capacity = 0) : free_head(invalid_index), num_elems(0) { slots.reserve(capacity); }

    /**
     * @brief Constructs an element in-place and returns its key.
     * @param args Arguments to forward to the constructor of T.
     * @return A KeyT object representing the newly inserted element.
     */
    template <typename... Args>
    KeyT emplace(Args&&... args) {
        index_type idx;
        if (free_head != invalid_index && free_head < slots.size()) {
            idx = free_head;
            auto& slot = slots[idx];
            free_head = slot.next_free;
            version_type new_version = slot.version | 1;
            slot = Slot(new_version, std::forward<Args>(args)...);
        } else {
            idx = static_cast<index_type>(slots.size());
            TT_FATAL(idx <= max_index, "SlotMap index out of bounds");
            constexpr version_type version = 1;
            slots.emplace_back(version, std::forward<Args>(args)...);
        }
        ++num_elems;

        return KeyT{idx, slots[idx].version};
    }

    /**
     * @brief Inserts a value into the SlotMap and returns its key.
     * @param value The value to insert.
     * @return A KeyT object representing the newly inserted element.
     */
    template <typename U>
    KeyT insert(U&& value) {
        return emplace(std::forward<U>(value));
    }

    /**
     * @brief Removes an element from the SlotMap.
     * @param key The key of the element to remove.
     */
    void remove(const KeyT& key) {
        if (!contains(key)) {
            return;
        }
        return remove_unchecked(key.index());
    }

    /**
     * @brief Retrieves a pointer to the value associated with the given key.
     * @param key The key to look up.
     * @return A pointer to the value if the key is valid, or nullptr otherwise.
     */
    [[nodiscard]] T* get(const KeyT& key) {
        if (!contains(key)) {
            return nullptr;
        }
        return &slots[key.index()].value;
    }

    /**
     * @brief Retrieves a const pointer to the value associated with the given key.
     * @param key The key to look up.
     * @return A const pointer to the value if the key is valid, or nullptr otherwise.
     */
    [[nodiscard]] const T* get(const KeyT& key) const {
        if (!contains(key)) {
            return nullptr;
        }
        return &slots[key.index()].value;
    }

    /**
     * @brief Returns the number of elements in the SlotMap.
     * @return The number of occupied slots.
     */
    size_t size() const noexcept { return num_elems; }

    /**
     * @brief Returns the current capacity of the SlotMap.
     * @return The total number of slots (occupied and free).
     */
    size_t capacity() const noexcept { return slots.capacity(); }

    /**
     * @brief Increases the capacity of the SlotMap to at least the specified value.
     * @param new_capacity The new capacity to the SlotMap.
     */
    void reserve(size_t new_capacity) {
        TT_FATAL(new_capacity <= max_index ,"SlotMap capacity out of bounds");

        // Technically this can reserve more than max_index, but it's not a problem,
        // since we still check the index bounds when inserting.
        slots.reserve(new_capacity);
    }

    /**
     * @brief Checks if the SlotMap is empty.
     * @return true if the SlotMap contains no elements, false otherwise.
     */
    bool empty() const noexcept { return num_elems == 0; }

    /**
     * @brief Clears the SlotMap, removing all elements.
     */
    void clear() noexcept {
        slots.clear();
        num_elems = 0;
        free_head = invalid_index;
    }

    /**
     * @brief Checks if the SlotMap contains a key.
     * @param key The key to check for.
     * @return true if the key is in the SlotMap, false otherwise.
     */
    bool contains(const KeyT& key) const {
        if (key.index() >= slots.size()) {
            return false;
        }
        // We don't need to check if the slot is occupied because the key's version
        // will always sets the occupied bit.
        return slots[key.index()].version == key.version();
    }

    class iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        iterator(SlotMap* smap, size_t index) : smap_(smap), index_(index) {
            // Advance to the first occupied slot
            if (index_ < smap_->slots.size() && !smap_->slots[index_].occupied()) {
                ++(*this);
            }
        }

        reference operator*() { return smap_->slots[index_].value; }
        pointer operator->() { return &smap_->slots[index_].value; }

        iterator& operator++() {
            do {
                ++index_;
            } while (index_ < smap_->slots.size() && !smap_->slots[index_].occupied());
            return *this;
        }

        iterator operator++(int) {
            iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const iterator& other) const { return index_ == other.index_; }
        bool operator!=(const iterator& other) const { return index_ != other.index_; }

       private:
        SlotMap* smap_;
        size_t index_;
    };

    class const_iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const T;
        using difference_type = std::ptrdiff_t;
        using pointer = const T*;
        using reference = const T&;

        const_iterator(const SlotMap* smap, size_t index) : smap_(smap), index_(index) {
            // Advance to the first occupied slot
            if (index_ < smap_->slots.size() && !smap_->slots[index_].occupied()) {
                ++(*this);
            }
        }

        reference operator*() const { return smap_->slots[index_].value; }
        pointer operator->() const { return &smap_->slots[index_].value; }

        const_iterator& operator++() {
            do {
                ++index_;
            } while (index_ < smap_->slots.size() && !smap_->slots[index_].occupied());
            return *this;
        }

        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(const const_iterator& other) const { return index_ == other.index_; }
        bool operator!=(const const_iterator& other) const { return index_ != other.index_; }

       private:
        const SlotMap* smap_;
        size_t index_;
    };

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, slots.size()); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, slots.size()); }
    const_iterator cbegin() const { return const_iterator(this, 0); }
    const_iterator cend() const { return const_iterator(this, slots.size()); }

   private:
    void remove_unchecked(uint32_t idx) {
        // Remove value from the slot before overwriting union.
        auto& slot = slots[idx];
        slot.~Slot();

        // Maintain freelist.
        slot.next_free = free_head;
        free_head = idx;
        slot.version += 1;
        --num_elems;
    }
};

}  // namespace tt::stl
