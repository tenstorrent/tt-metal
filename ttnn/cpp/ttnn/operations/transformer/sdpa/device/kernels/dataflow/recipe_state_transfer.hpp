// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/circular_buffer.h"
#include "../recipe_state_layout.hpp"

// Where a Q block's checkpoint lives: its slot of the DRAM state tensor, or, for a core's first cache_slots Q blocks,
// a slot of the core's L1 state cache (CB recipe_ring_cb::state_cache, laid out as the tensor's slots): local NoC
// copies instead of a DRAM round trip.
template <typename Accessor>
struct RecipeStateStore {
    const Accessor& dram;
    uint32_t cache_address;  // the state cache CB's L1 address
    uint32_t cache_slots;
    uint32_t first_slot;  // the core's first Q block
};

// Page reads and writes of one slot of a RecipeStateStore (page ids as in the DRAM tensor).
template <typename Accessor>
struct RecipeStateSlot {
    Noc& noc;
    const RecipeStateStore<Accessor>& store;
    uint32_t base;  // the slot's first page id
    bool cached;
    uint32_t cache_base;  // L1 address of the slot's first page, if cached
    RecipeStateSlot(Noc& n, const RecipeStateStore<Accessor>& st, uint32_t slot, uint32_t slot_pages) :
        noc(n), store(st), base(slot * slot_pages) {
        const uint32_t local = slot - st.first_slot;
        cached = local < st.cache_slots;
        cache_base = st.cache_address + local * slot_pages * sdpa::streaming::StateTransfer::page_bytes;
    }
    // Out of line, one copy for both directions (the ring writer sits at the kernel config buffer limit). A cached
    // page is a local L1-to-L1 copy on the writer's NoC.
    __attribute__((noinline)) void move(uint32_t l1, uint32_t size, uint32_t page_id, bool restore) {
        if (cached) {
            const uint64_t cache = get_noc_addr(cache_base + (page_id - base) * sdpa::streaming::StateTransfer::page_bytes);
            if (restore) {
                noc_async_read(cache, l1, size);
            } else {
                noc_async_write(l1, cache, size);
            }
        } else if (restore) {
            noc.async_read(store.dram, CoreLocalMem<uint32_t>(l1), size, {.page_id = page_id}, {});
        } else {
            noc.async_write(CoreLocalMem<uint32_t>(l1), store.dram, size, {}, {.page_id = page_id});
        }
    }
    void read(uint32_t l1, uint32_t size, uint32_t page_id) { move(l1, size, page_id, true); }
    void write(uint32_t l1, uint32_t size, uint32_t page_id) { move(l1, size, page_id, false); }
};

template <bool fp32, uint32_t q_tiles, uint32_t request_cb, uint32_t ack_cb, uint32_t d_tiles = 4, typename Accessor>
void transfer_recipe_state(Noc& noc, const RecipeStateStore<Accessor>& backing) {
    using Transfer = sdpa::streaming::StateTransfer;
    // The FP32 recipes' maxima are FP32 tiles.
    constexpr uint32_t max_bytes = fp32 ? 4096 : 2048;
    CircularBuffer request(request_cb), ack(ack_cb);
    request.wait_front(1);
    ack.reserve_back(1);
    const auto* words = reinterpret_cast<const volatile uint32_t*>(request.get_read_ptr());
    const bool restore = words[Transfer::Operation] == Transfer::Restore;
    RecipeStateSlot<Accessor> state(noc, backing, words[Transfer::Slot], Transfer::pages<q_tiles, d_tiles, max_bytes> + 1);
    const uint32_t base = state.base;
    uint32_t page = base + 1;
    for (uint32_t plane = 0; plane < 3; ++plane) {
        const uint32_t cb = words[Transfer::Numerator + plane];
        const uint32_t bytes = Transfer::plane_bytes<q_tiles, d_tiles, max_bytes>(plane);
        // These full-capacity state banks are at their allocation origin at
        // every segment boundary. Dataflow never advances their CB pointers.
        const uint32_t address = CircularBuffer(cb).get_read_ptr();
        // The BF16 recipes' numerator CB holds, per Q row, O (plane 0) then a rescaled group's chunk PV
        // (plane 1, per-chunk scratch). Only plane 0 is state: move each row's d tiles, skipping plane 1.
        const auto l1_offset = [&](uint32_t offset) -> uint32_t {
            if (fp32 || plane != 0) {
                return offset;
            }
            const uint32_t tile = offset / Transfer::page_bytes;
            return ((tile / d_tiles) * 2 * d_tiles + tile % d_tiles) * Transfer::page_bytes + offset % Transfer::page_bytes;
        };
        if constexpr (Transfer::page_aligned<q_tiles, d_tiles, max_bytes>) {
            for (uint32_t offset = 0; offset < bytes; offset += Transfer::page_bytes, ++page) {
                if (restore) {
                    state.read(address + l1_offset(offset), Transfer::page_bytes, page);
                } else {
                    state.write(address + l1_offset(offset), Transfer::page_bytes, page);
                }
            }
        } else {
            // Odd Q tile counts: a half-page maxima plane. Move only the plane's bytes of its last page, so a
            // restore never writes past the plane into the neighbouring CB.
            for (uint32_t offset = 0; offset < bytes; offset += Transfer::page_bytes, ++page) {
                const uint32_t size = bytes - offset < Transfer::page_bytes ? bytes - offset : Transfer::page_bytes;
                if (restore) {
                    state.read(address + l1_offset(offset), size, page);
                } else {
                    state.write(address + l1_offset(offset), size, page);
                }
            }
        }
    }
    if (restore) {
        state.read(ack.get_write_ptr(), Transfer::Words * sizeof(uint32_t), base);
        noc.async_read_barrier();
    } else {
        state.write(request.get_read_ptr(), Transfer::Words * sizeof(uint32_t), base);
        noc.async_write_barrier();
    }
    request.pop_front(1);
    ack.push_back(1);
}

// Streamed checkpoints (BF16 recipes with fused chunks only): see StateTransfer. Services one
// Save/Restore/RestoreStream request, or a run of SaveRows requests through their SaveTail. Save and Restore (compute
// without fused chunks) move the same pages as transfer_recipe_state, through the streamed paths so the writer carries
// one copy of them. Compute never waits for a save: a restore first flushes this core's outstanding save writes out of
// the state banks it overwrites, and drain_resident_ring_iter's end-of-iteration write barrier lands every save before
// the next iteration restores it.
template <uint32_t q_tiles, uint32_t request_cb, uint32_t ack_cb, uint32_t d_tiles = 4, typename Accessor>
void stream_recipe_state(Noc& noc, const RecipeStateStore<Accessor>& backing) {
    using Transfer = sdpa::streaming::StateTransfer;
    constexpr uint32_t page = Transfer::page_bytes;
    constexpr uint32_t o_pages = q_tiles * d_tiles;
    CircularBuffer request(request_cb), ack(ack_cb);
    request.wait_front(1);
    const auto* words = reinterpret_cast<const volatile uint32_t*>(request.get_read_ptr());
    uint32_t op = words[Transfer::Operation];
    // A run of SaveRows through its SaveTail is one block's: one slot.
    RecipeStateSlot<Accessor> state(noc, backing, words[Transfer::Slot], Transfer::pages<q_tiles, d_tiles> + 1);
    // O row r of the numerator CB: d tiles of plane 0 (plane 1 is per-chunk scratch); DRAM keeps plane 0 only.
    const auto move_o_rows = [&](uint32_t base, uint32_t address, uint32_t row0, uint32_t rows, bool restore) {
        for (uint32_t r = row0; r < row0 + rows; ++r) {
            for (uint32_t c = 0; c < d_tiles; ++c) {
                const uint32_t l1 = address + (r * 2 * d_tiles + c) * page;
                const uint32_t page_id = base + 1 + r * d_tiles + c;
                if (restore) {
                    state.read(l1, page, page_id);
                } else {
                    state.write(l1, page, page_id);
                }
            }
        }
    };
    // Maxima (plane 1) and sums (plane 2), page by page (an odd Q's maxima plane ends in a half page).
    const auto move_tail_planes = [&](uint32_t base, bool restore) {
        uint32_t page_id = base + 1 + o_pages;
        for (uint32_t plane = 1; plane < 3; ++plane) {
            const uint32_t address = CircularBuffer(words[Transfer::Numerator + plane]).get_read_ptr();
            const uint32_t bytes = Transfer::plane_bytes<q_tiles, d_tiles>(plane);
            for (uint32_t offset = 0; offset < bytes; offset += page, ++page_id) {
                const uint32_t size = bytes - offset < page ? bytes - offset : page;
                if (restore) {
                    state.read(address + offset, size, page_id);
                } else {
                    state.write(address + offset, size, page_id);
                }
            }
        }
    };
    const uint32_t base = state.base;
    const uint32_t o_address = CircularBuffer(words[Transfer::Numerator]).get_read_ptr();
    if (op == Transfer::Restore || op == Transfer::RestoreStream) {
        const bool whole = op == Transfer::Restore;
        noc.async_writes_flushed();  // the previous block's saves have left the banks this overwrites
        ack.reserve_back(1);
        move_tail_planes(base, true);
        if (whole) {
            move_o_rows(base, o_address, 0, q_tiles, true);
        }
        state.read(ack.get_write_ptr(), Transfer::Words * sizeof(uint32_t), base);
        noc.async_read_barrier();
        ack.push_back(1);
        request.pop_front(1);
        if (whole) {
            return;
        }
        for (uint32_t r = 0; r < q_tiles; ++r) {
            move_o_rows(base, o_address, r, 1, true);
            noc.async_read_barrier();
            ack.reserve_back(1);
            ack.push_back(1);
        }
        return;
    }
    // SaveRows ... SaveTail
    while (op == Transfer::SaveRows) {
        move_o_rows(base, o_address, words[Transfer::Row0], words[Transfer::Rows], false);
        request.pop_front(1);
        request.wait_front(1);
        words = reinterpret_cast<const volatile uint32_t*>(request.get_read_ptr());
        op = words[Transfer::Operation];
    }
    // op == SaveTail, or a whole Save (acknowledged once its writes have landed)
    const bool whole = op == Transfer::Save;
    const uint32_t saved = whole ? 0 : words[Transfer::Row0];
    const bool ack_flushed = whole || words[Transfer::Rows] != 0;
    move_o_rows(base, o_address, saved, q_tiles - saved, false);
    move_tail_planes(base, false);
    state.write(request.get_read_ptr(), Transfer::Words * sizeof(uint32_t), base);
    if (whole) {
        noc.async_write_barrier();
    } else {
        noc.async_writes_flushed();  // the header leaves the request page before it is recycled
    }
    request.pop_front(1);
    if (ack_flushed) {
        ack.reserve_back(1);
        ack.push_back(1);
    }
}
