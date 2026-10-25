// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"

namespace kda {

// Inputs that do not depend on the value columns are identical for every value block of a head. With
// mcast_shared, value block 0 reads them from DRAM once and multicasts them into its siblings' buffers.
// The slot addresses match on every core because every block makes the same reserve/push sequence on
// identically specified buffers. Handshake: receivers reserve their slots, reset valid and raise ready; the
// sender stages its reads, waits for every receiver, multicasts the data, flushes, then multicasts the valid
// flag before publishing its own copy.
template <typename DFB>
struct SharedInput {
    DFB* buffer;
    uint32_t tiles;
    uint32_t slot;
};
// kda::SharedInput{&buffer, tiles, slot} deduces the buffer's type; an array spells it out:
// kda::SharedInput<decltype(buffer)> inputs[N] = {{&buffer, tiles, slot}, ...}.
template <typename DFB>
SharedInput(DFB*, uint32_t, uint32_t) -> SharedInput<DFB>;

template <typename Accessor, typename DFB>
FORCE_INLINE uint32_t
stage_contiguous_tiles(const Accessor& accessor, DFB& buffer, Noc& noc, uint32_t base, uint32_t count) {
    buffer.reserve_back(count);
    const uint32_t entry_size = buffer.get_entry_size();
    uint32_t tile = 0;
    // Each page carries its address from the iteration, so a read does not resolve its page id again.
    for (const auto& page : accessor.pages(base, base + count)) {
        noc.async_read(page, buffer, entry_size, {}, {.offset_bytes = tile * entry_size});
        ++tile;
    }
    return buffer.get_write_ptr();
}

template <uint32_t N, typename ReadySem, typename ValidSem, typename DFB>
FORCE_INLINE void multicast_shared(
    Noc& noc,
    SharedInput<DFB> (&inputs)[N],
    ReadySem& ready,
    ValidSem& valid,
    uint32_t x0,
    uint32_t y0,
    uint32_t x1,
    uint32_t y1,
    uint32_t receivers) {
    noc.async_read_barrier();
    ready.wait(receivers);
    ready.set(0);
    MulticastEndpoint destination;
    for (auto& input : inputs) {
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(input.slot),
            destination,
            input.tiles * input.buffer->get_entry_size(),
            receivers,
            {},
            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = input.slot},
            /*linked=*/true);
    }
    // The flag multicast issues from a different command buffer than the data, so only this flush orders the data
    // before the flag. It also proves the source slots were read before this core's compute may pop them.
    noc.async_writes_flushed();
    valid.set_multicast(noc, x0, y0, x1, y1, receivers);
    for (auto& input : inputs) {
        input.buffer->push_back(input.tiles);
    }
}

// A stream of payloads (one per chunk) that one value block (the sender) reads and multicasts to its siblings,
// pipelined as deep as the payload buffers. Both semaphores count from the zero dispatch initializes them to and are
// never reset, so no side waits on a round trip per payload:
// - stage(p) on the sender issues payload p's reads; on a receiver it reserves p's slots and raises the sender's ready
//   by one (the slot addresses match on every block, as for multicast_shared);
// - publish(p, stage_next) on the sender waits for p's reads and for ready to reach receivers * (p + 1), issues the
//   data multicast (the chain closed by its last write), publishes its own copy, calls stage_next while the
//   multicast drains, then waits for the data to leave and multicasts VALID = p + 1. On a receiver it calls
//   stage_next, so ready for the next payload is raised before it waits, then waits for VALID to reach p + 1 and
//   publishes the slots.
// A sender's source slots stay intact until VALID: it reuses a slot only after a later payload's reserve, which
// follows. Payloads alternate between two descriptor sets, so staging the next payload keeps the current one's.
template <uint32_t N, typename DFB, typename ReadySem, typename ValidSem>
class SharedInputStream {
public:
    SharedInputStream(
        Noc& noc,
        ReadySem& ready,
        ValidSem& valid,
        bool sender,
        uint32_t x0,
        uint32_t y0,
        uint32_t x1,
        uint32_t y1,
        uint32_t receivers) :
        noc_(noc),
        ready_(ready),
        valid_(valid),
        sender_(sender),
        x0_(x0),
        y0_(y0),
        x1_(x1),
        y1_(y1),
        receivers_(receivers) {}

    // for_each_input(input) calls input(accessor, buffer, first_page, tiles) for each of the payload's N buffers, in
    // the order compute consumes them.
    template <typename ForEachInput>
    FORCE_INLINE void stage(uint32_t payload, ForEachInput&& for_each_input) {
        auto& inputs = inputs_[payload & 1];
        uint32_t count = 0;
        for_each_input([&](const auto& accessor, DFB& buffer, uint32_t first_page, uint32_t tiles) {
            inputs[count++] = {
                &buffer, tiles, sender_ ? stage_contiguous_tiles(accessor, buffer, noc_, first_page, tiles) : 0};
        });
        if (!sender_) {
            // A reservation does not advance the write pointer, so a payload staged behind unpublished ones reserves
            // their slots too.
            ++staged_;
            for (auto& input : inputs) {
                input.buffer->reserve_back(staged_ * input.tiles);
            }
            ready_.up(noc_, x0_, y0_, 1);
        }
    }

    template <typename StageNext>
    FORCE_INLINE void publish(uint32_t payload, StageNext&& stage_next) {
        auto& inputs = inputs_[payload & 1];
        if (sender_) {
            noc_.async_read_barrier();
            ready_.wait_min(receivers_ * (payload + 1));
            MulticastEndpoint destination;
            for (uint32_t index = 0; index < N; ++index) {
                auto& input = inputs[index];
                noc_.async_write_multicast(
                    CoreLocalMem<uint32_t>(input.slot),
                    destination,
                    input.tiles * input.buffer->get_entry_size(),
                    receivers_,
                    {},
                    {.noc_x_start = x0_, .noc_y_start = y0_, .noc_x_end = x1_, .noc_y_end = y1_, .addr = input.slot},
                    /*linked=*/index + 1 < N);
            }
            for (auto& input : inputs) {
                input.buffer->push_back(input.tiles);
            }
            stage_next();
            // The flag multicast issues from a different command buffer than the data, so only this flush orders the
            // data before the flag. It also retires the previous flag's multicast, which sourced the word set here.
            noc_.async_writes_flushed();
            valid_.set(payload + 1);
            valid_.set_multicast(noc_, x0_, y0_, x1_, y1_, receivers_);
        } else {
            stage_next();
            valid_.wait_min(payload + 1);
            for (auto& input : inputs) {
                input.buffer->push_back(input.tiles);
            }
            --staged_;
        }
    }

    // Retire the multicast writes or the ready increments before exit; dispatch re-initializes both semaphores on
    // every launch.
    FORCE_INLINE void finish() {
        if (sender_) {
            noc_.async_write_barrier();
        } else {
            noc_.async_atomic_barrier();
        }
    }

private:
    Noc& noc_;
    ReadySem& ready_;
    ValidSem& valid_;
    bool sender_;
    uint32_t x0_;
    uint32_t y0_;
    uint32_t x1_;
    uint32_t y1_;
    uint32_t receivers_;
    // A receiver's payloads staged and not yet published.
    uint32_t staged_ = 0;
    SharedInput<DFB> inputs_[2][N];
};

template <uint32_t N, typename ReadySem, typename ValidSem, typename DFB>
FORCE_INLINE void receive_shared(
    Noc& noc, SharedInput<DFB> (&inputs)[N], ReadySem& ready, ValidSem& valid, uint32_t sender_x, uint32_t sender_y) {
    for (auto& input : inputs) {
        input.buffer->reserve_back(input.tiles);
    }
    // Reset valid before raising ready: a fast sender may multicast VALID right after the increment.
    valid.set(0);
    ready.up(noc, sender_x, sender_y, 1);
    valid.wait(1);
    for (auto& input : inputs) {
        input.buffer->push_back(input.tiles);
    }
}

}  // namespace kda
