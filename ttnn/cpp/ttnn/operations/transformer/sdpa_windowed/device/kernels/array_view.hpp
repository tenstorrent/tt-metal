// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/assert.h"
#include "debug/dprint_tile.h"

enum class CBAccessType : uint8_t { CB_FRONT_RW, CB_BACK_RW, CB_FRONT_RO, CB_BACK_RO };

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_ASSERT) && !defined(FORCE_WATCHER_OFF)
#define WATCHER_OVERHEAD_OK 1
#endif

/**
 * @brief ArrayView struct - Type-safe view into circular buffer L1 memory.
 *
 * Provides a direct, type-safe view into the L1 memory region pointed to by the front (read pointer)
 * or back (write pointer) of a circular buffer (CB). Read/write access is determined at compile time
 * based on the CBAccessType parameter.
 *
 * Lifetime:
 *   The lifetime of any one ArrayView must be contained within the lifetime of its surrounding
 *   cb_push_back/cb_pop_front pair
 *
 * Template parameters:
 *   @tparam T Element type (e.g., uint32_t, float)
 *   @tparam cb_id Circular buffer ID
 *   @tparam _type CB pointer type:
 *     - CB_FRONT: Read/write access to front pointer
 *     - CB_BACK: Read/write access to back pointer
 *     - CB_FRONT_RO: Read-only access to front pointer
 *     - CB_BACK_RO: Read-only access to back pointer
 *
 * Methods:
 *   - operator[]: Access elements by index with bounds checking (read-only for *_RO types)
 *   - size(): Get number of elements available in the buffer
 *
 * Example usage:
 *   ArrayView<uint32_t, cb_id, CBAccessType::CB_FRONT> view_rw;
 *   view_rw[0] = 42; // write to front of CB
 *   uint32_t val = view_rw[1]; // read from front of CB
 *
 *   ArrayView<uint32_t, cb_id, CBAccessType::CB_FRONT_RO> view_ro;
 *   uint32_t val = view_ro[0]; // read-only access
 *   // view_ro[0] = 42; // Compile error - no write access
 */
template <typename T, CBAccessType _type>
struct ArrayView {
    ArrayView(uint32_t cb_id, uint32_t tile_id_offset = 0, uint32_t ntiles = 1) {
        ASSERT(ntiles > 0);

        auto tile_size = get_tile_size(cb_id);
        if constexpr (_type == CBAccessType::CB_FRONT_RW || _type == CBAccessType::CB_FRONT_RO) {
            _ptr = reinterpret_cast<volatile tt_l1_ptr T*>(CB_RD_PTR(cb_id) + tile_id_offset * tile_size);

#if defined(WATCHER_OVERHEAD_OK)
            _base_addr = CB_RD_PTR(cb_id) + tile_id_offset * tile_size;
#endif
        } else {
            _ptr = reinterpret_cast<volatile tt_l1_ptr T*>(CB_WR_PTR(cb_id) + tile_id_offset * tile_size);

#if defined(WATCHER_OVERHEAD_OK)
            _base_addr = CB_WR_PTR(cb_id) + tile_id_offset * tile_size;
#endif
        }
        _size = ntiles * tile_size / sizeof(T);
    }

    // Non-const operator[] - only available for read/write types
    template <
        CBAccessType type = _type,
        typename = std::enable_if_t<type != CBAccessType::CB_FRONT_RO && type != CBAccessType::CB_BACK_RO>>
    volatile T& operator[](size_t i) {
        ASSERT(i < _size);
        return _ptr[i];
    }

    // Const operator[] - always available for reading
    const volatile T& operator[](size_t i) const {
        ASSERT(i < _size);
        return _ptr[i];
    }

    size_t size() const { return _size; }

#if defined(WATCHER_OVERHEAD_OK)
    uint32_t addr(size_t i) const { return _base_addr + i * sizeof(T); }
#endif

private:
    size_t _size = 0;
    volatile tt_l1_ptr T* _ptr = nullptr;
#if defined(WATCHER_OVERHEAD_OK)
    // For debugging purposes, store the L1 address of the array view
    uint32_t _base_addr = 0;
#endif
};
