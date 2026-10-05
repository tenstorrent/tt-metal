// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "api/dataflow/circular_buffer.h"
#include "../recipe_state_layout.hpp"

template <bool fp32, uint32_t q_tiles, uint32_t request_cb, uint32_t ack_cb, uint32_t d_tiles = 4, typename Accessor>
void transfer_recipe_state(Noc& noc, const Accessor& backing) {
    using Transfer = sdpa::streaming::StateTransfer;
    CircularBuffer request(request_cb), ack(ack_cb);
    request.wait_front(1);
    ack.reserve_back(1);
    const auto* words = reinterpret_cast<const volatile uint32_t*>(request.get_read_ptr());
    const bool restore = words[Transfer::Operation] == Transfer::Restore;
    const uint32_t base = words[Transfer::Slot] * (Transfer::pages<fp32, q_tiles, d_tiles> + 1);
    uint32_t page = base + 1;
    for (uint32_t plane = 0; plane < 3; ++plane) {
        const uint32_t cb = words[Transfer::Numerator + plane];
        const uint32_t bytes = Transfer::plane_bytes<fp32, q_tiles, d_tiles>(plane);
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
        if constexpr (Transfer::page_aligned<fp32, q_tiles, d_tiles>) {
            for (uint32_t offset = 0; offset < bytes; offset += Transfer::page_bytes, ++page) {
                if (restore) {
                    noc.async_read(
                        backing, CoreLocalMem<uint32_t>(address + l1_offset(offset)), Transfer::page_bytes, {.page_id = page}, {});
                } else {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(address + l1_offset(offset)), backing, Transfer::page_bytes, {}, {.page_id = page});
                }
            }
        } else {
            // Odd Q tile counts: a half-page maxima plane. Move only the plane's bytes of its last page, so a
            // restore never writes past the plane into the neighbouring CB.
            for (uint32_t offset = 0; offset < bytes; offset += Transfer::page_bytes, ++page) {
                const uint32_t size = bytes - offset < Transfer::page_bytes ? bytes - offset : Transfer::page_bytes;
                if (restore) {
                    noc.async_read(backing, CoreLocalMem<uint32_t>(address + l1_offset(offset)), size, {.page_id = page}, {});
                } else {
                    noc.async_write(CoreLocalMem<uint32_t>(address + l1_offset(offset)), backing, size, {}, {.page_id = page});
                }
            }
        }
    }
    if (restore) {
        noc.async_read(
            backing,
            CoreLocalMem<uint32_t>(ack.get_write_ptr()),
            Transfer::Words * sizeof(uint32_t),
            {.page_id = base},
            {});
        noc.async_read_barrier();
    } else {
        noc.async_write(
            CoreLocalMem<uint32_t>(request.get_read_ptr()),
            backing,
            Transfer::Words * sizeof(uint32_t),
            {},
            {.page_id = base});
        noc.async_write_barrier();
    }
    request.pop_front(1);
    ack.push_back(1);
}

#ifdef SDPA_RING_STREAM_STATE
// Streamed checkpoints (BF16 recipes only): see StateTransfer. Services one Save/Restore/RestoreStream request,
// or a run of SaveRows requests through their SaveTail. Save and Restore (compute without fused chunks) move the
// same pages as transfer_recipe_state, through the streamed paths so the writer carries one copy of them. Compute never waits for a save: a restore first flushes
// this core's outstanding save writes out of the state banks it overwrites, and drain_resident_ring_iter's
// end-of-iteration write barrier lands every save before the next iteration restores it.
template <uint32_t q_tiles, uint32_t request_cb, uint32_t ack_cb, uint32_t d_tiles = 4, typename Accessor>
void stream_recipe_state(Noc& noc, const Accessor& backing) {
    using Transfer = sdpa::streaming::StateTransfer;
    constexpr uint32_t page = Transfer::page_bytes;
    constexpr uint32_t o_pages = q_tiles * d_tiles;
    CircularBuffer request(request_cb), ack(ack_cb);
    request.wait_front(1);
    const auto* words = reinterpret_cast<const volatile uint32_t*>(request.get_read_ptr());
    uint32_t op = words[Transfer::Operation];
    // O row r of the numerator CB: d tiles of plane 0 (plane 1 is per-chunk scratch); DRAM keeps plane 0 only.
    const auto move_o_rows = [&](uint32_t base, uint32_t address, uint32_t row0, uint32_t rows, bool restore) {
        for (uint32_t r = row0; r < row0 + rows; ++r) {
            for (uint32_t c = 0; c < d_tiles; ++c) {
                const uint32_t l1 = address + (r * 2 * d_tiles + c) * page;
                const uint32_t page_id = base + 1 + r * d_tiles + c;
                if (restore) {
                    noc.async_read(backing, CoreLocalMem<uint32_t>(l1), page, {.page_id = page_id}, {});
                } else {
                    noc.async_write(CoreLocalMem<uint32_t>(l1), backing, page, {}, {.page_id = page_id});
                }
            }
        }
    };
    // Maxima (plane 1) and sums (plane 2), page by page (an odd Q's maxima plane ends in a half page).
    const auto move_tail_planes = [&](uint32_t base, bool restore) {
        uint32_t page_id = base + 1 + o_pages;
        for (uint32_t plane = 1; plane < 3; ++plane) {
            const uint32_t address = CircularBuffer(words[Transfer::Numerator + plane]).get_read_ptr();
            const uint32_t bytes = Transfer::plane_bytes<false, q_tiles, d_tiles>(plane);
            for (uint32_t offset = 0; offset < bytes; offset += page, ++page_id) {
                const uint32_t size = bytes - offset < page ? bytes - offset : page;
                if (restore) {
                    noc.async_read(backing, CoreLocalMem<uint32_t>(address + offset), size, {.page_id = page_id}, {});
                } else {
                    noc.async_write(CoreLocalMem<uint32_t>(address + offset), backing, size, {}, {.page_id = page_id});
                }
            }
        }
    };
    const uint32_t base = words[Transfer::Slot] * (Transfer::pages<false, q_tiles, d_tiles> + 1);
    const uint32_t o_address = CircularBuffer(words[Transfer::Numerator]).get_read_ptr();
    if (op == Transfer::Restore || op == Transfer::RestoreStream) {
        const bool whole = op == Transfer::Restore;
        noc.async_writes_flushed();  // the previous block's saves have left the banks this overwrites
        ack.reserve_back(1);
        move_tail_planes(base, true);
        if (whole) {
            move_o_rows(base, o_address, 0, q_tiles, true);
        }
        noc.async_read(
            backing, CoreLocalMem<uint32_t>(ack.get_write_ptr()), Transfer::Words * sizeof(uint32_t), {.page_id = base}, {});
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
    noc.async_write(
        CoreLocalMem<uint32_t>(request.get_read_ptr()), backing, Transfer::Words * sizeof(uint32_t), {}, {.page_id = base});
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
#endif
