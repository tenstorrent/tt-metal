// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t G>
FORCE_INLINE void synchronize_head_stage(
    uint32_t worker_index,
    uint32_t group,
    uint32_t& completed_stages,
    Noc& noc,
    Semaphore<>& arrival,
    Semaphore<>& release) {
    completed_stages++;
    const uint32_t coordinator = worker_index - group;
    const uint32_t coordinator_x = get_common_vararg(2 * coordinator);
    const uint32_t coordinator_y = get_common_vararg(2 * coordinator + 1);
    arrival.up(noc, coordinator_x, coordinator_y, 1);
    if (group == 0) {
        arrival.wait_min(completed_stages * G);
        for (uint32_t worker = coordinator; worker < coordinator + G; worker++) {
            const uint32_t worker_x = get_common_vararg(2 * worker);
            const uint32_t worker_y = get_common_vararg(2 * worker + 1);
            release.up(noc, worker_x, worker_y, 1);
        }
        noc.async_atomic_barrier();
    }
    release.wait_min(completed_stages);
}

template <uint32_t Kt, uint32_t Vt, uint32_t BH, uint32_t G>
TT_KERNEL void dataflow(uint32_t worker_index, uint32_t group) {
    constexpr uint32_t kk = Kt * Kt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t worker_count = BH * G;
    static_assert(worker_count <= 128, "affine prefix coordinate table exceeds runtime-arg budget");

    const auto a_accessor = TensorAccessor(tensor::a);
    const auto b_accessor = TensorAccessor(tensor::b);
    const auto initial_state_accessor = TensorAccessor(tensor::initial_state);
    const auto output_accessor = TensorAccessor(tensor::output);
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer local_a(dfb::local_a);
    DataflowBuffer local_b(dfb::local_b);
    DataflowBuffer to_remote_a(dfb::to_remote_a);
    DataflowBuffer to_remote_b(dfb::to_remote_b);
    DataflowBuffer from_remote_a(dfb::from_remote_a);
    DataflowBuffer from_remote_b(dfb::from_remote_b);
    DataflowBuffer initial_state(dfb::initial_state);
    DataflowBuffer final(dfb::final);
    Noc noc;
    Semaphore<> ready(sem::ready);
    Semaphore<> arrival(sem::arrival);
    Semaphore<> release(sem::release);

    auto worker_x = [](uint32_t worker) { return get_common_vararg(2 * worker); };
    auto worker_y = [](uint32_t worker) { return get_common_vararg(2 * worker + 1); };
    auto read_tiles = [&](const auto& accessor, DataflowBuffer& buffer, uint32_t page, uint32_t tiles) {
        buffer.reserve_back(tiles);
        for (uint32_t tile = 0; tile < tiles; tile++) {
            noc.async_read(
                accessor,
                buffer,
                buffer.get_entry_size(),
                {.page_id = page + tile},
                {.offset_bytes = tile * buffer.get_entry_size()});
        }
        noc.async_read_barrier();
        buffer.push_back(tiles);
    };
    read_tiles(a_accessor, initial_a, worker_index * kk, kk);
    read_tiles(b_accessor, initial_b, worker_index * kv, kv);
    read_tiles(initial_state_accessor, initial_state, (worker_index / G) * kv, kv);

    uint32_t completed_stages = 0;
    auto send_pair = [&](uint32_t target, DataflowBuffer& current_a, DataflowBuffer& current_b) {
        const uint32_t target_x = worker_x(target);
        const uint32_t target_y = worker_y(target);
        noc.async_write(
            current_a,
            UnicastEndpoint{},
            kk * current_a.get_entry_size(),
            {},
            {.noc_x = target_x, .noc_y = target_y, .addr = from_remote_a.get_write_ptr()});
        noc.async_write(
            current_b,
            UnicastEndpoint{},
            kv * current_b.get_entry_size(),
            {},
            {.noc_x = target_x, .noc_y = target_y, .addr = from_remote_b.get_write_ptr()});
    };
    auto loopback_pair = [&] {
        local_a.reserve_back(kk);
        local_b.reserve_back(kv);
        const uint32_t local_x = worker_x(worker_index);
        const uint32_t local_y = worker_y(worker_index);
        noc.async_write(
            to_remote_a,
            UnicastEndpoint{},
            kk * to_remote_a.get_entry_size(),
            {},
            {.noc_x = local_x, .noc_y = local_y, .addr = local_a.get_write_ptr()});
        noc.async_write(
            to_remote_b,
            UnicastEndpoint{},
            kv * to_remote_b.get_entry_size(),
            {},
            {.noc_x = local_x, .noc_y = local_y, .addr = local_b.get_write_ptr()});
        noc.async_write_barrier();
        local_a.push_back(kk);
        local_b.push_back(kv);
    };

    uint32_t ready_target = 0;
    for (uint32_t distance = 1; distance < G; distance *= 2) {
        to_remote_a.wait_front(kk);
        to_remote_b.wait_front(kv);
        const bool sends = group + distance < G;
        const bool receives = group >= distance;
        if (sends) {
            send_pair(worker_index + distance, to_remote_a, to_remote_b);
        }
        if (receives) {
            // This barrier retires both the remote send and same-core loopback when this worker does both.
            loopback_pair();
        } else if (sends) {
            noc.async_write_barrier();
        }
        if (sends) {
            ready.up(noc, worker_x(worker_index + distance), worker_y(worker_index + distance), 1);
        }
        if (receives) {
            from_remote_a.reserve_back(kk);
            from_remote_b.reserve_back(kv);
            // All NoC writes complete before this receiver frees the old outbound block.
            to_remote_a.pop_front(kk);
            to_remote_b.pop_front(kv);

            ready_target++;
            ready.wait_min(ready_target);
            from_remote_a.push_back(kk);
            from_remote_b.push_back(kv);

            // Compute must publish the replacement before any worker starts the next distance.
            to_remote_a.wait_front(kk);
            to_remote_b.wait_front(kv);
        }
        // Each head is an independent G-worker scan. Do not release its next NoC stage until all G workers have
        // consumed their remote buffers and produced the next prefix; otherwise that head can overwrite a mailbox
        // while compute is still reading it. Arrival and release semaphore targets stay monotonic across stages.
        synchronize_head_stage<G>(worker_index, group, completed_stages, noc, arrival, release);
    }

    to_remote_a.wait_front(kk);
    to_remote_b.wait_front(kv);
    if (group + 1 < G) {
        const uint32_t target = worker_index + 1;
        send_pair(target, to_remote_a, to_remote_b);
        noc.async_write_barrier();
        ready.up(noc, worker_x(target), worker_y(target), 1);
    }
    // The final inclusive prefix is never reused after the exclusive neighbor shift.
    to_remote_a.pop_front(kk);
    to_remote_b.pop_front(kv);
    if (group > 0) {
        from_remote_a.reserve_back(kk);
        from_remote_b.reserve_back(kv);
        ready_target++;
        ready.wait_min(ready_target);
        from_remote_a.push_back(kk);
        from_remote_b.push_back(kv);
    }

    final.wait_front(kv);
    for (uint32_t tile = 0; tile < kv; tile++) {
        noc.async_write(
            final,
            output_accessor,
            final.get_entry_size(),
            {.offset_bytes = tile * final.get_entry_size()},
            {.page_id = worker_index * kv + tile});
    }
    noc.async_write_barrier();
    final.pop_front(kv);
}
