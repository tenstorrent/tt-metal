// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

FORCE_INLINE uint32_t worker_x(uint32_t worker) { return get_common_vararg(2 * worker); }
FORCE_INLINE uint32_t worker_y(uint32_t worker) { return get_common_vararg(2 * worker + 1); }

template <typename Accessor>
FORCE_INLINE void issue_tensor_block_read(
    Noc& noc, const Accessor& accessor, DataflowBuffer& buffer, uint32_t page, uint32_t tiles) {
    for (uint32_t tile = 0; tile < tiles; tile++) {
        noc.async_read(
            accessor,
            buffer,
            buffer.get_entry_size(),
            {.page_id = page + tile},
            {.offset_bytes = tile * buffer.get_entry_size()});
    }
}

template <uint32_t Kt, uint32_t Vt>
FORCE_INLINE void issue_affine_pair_send(
    Noc& noc,
    uint32_t destination_worker,
    DataflowBuffer& send_a,
    DataflowBuffer& send_b,
    DataflowBuffer& remote_affine) {
    const uint32_t target_x = worker_x(destination_worker);
    const uint32_t target_y = worker_y(destination_worker);
    const uint32_t tile_bytes = remote_affine.get_entry_size();
    for (uint32_t row = 0; row < Kt; ++row) {
        const uint32_t remote_row = remote_affine.get_write_ptr() + row * (Kt + Vt) * tile_bytes;
        noc.async_write(
            send_a,
            UnicastEndpoint{},
            Kt * tile_bytes,
            {.offset_bytes = row * Kt * tile_bytes},
            {.noc_x = target_x, .noc_y = target_y, .addr = remote_row});
        noc.async_write(
            send_b,
            UnicastEndpoint{},
            Vt * tile_bytes,
            {.offset_bytes = row * Vt * tile_bytes},
            {.noc_x = target_x, .noc_y = target_y, .addr = remote_row + Kt * tile_bytes});
    }
}

FORCE_INLINE void complete_affine_pair_send(Noc& noc, Semaphore<>& ready, uint32_t destination_worker) {
    noc.async_write_barrier();
    ready.up(noc, worker_x(destination_worker), worker_y(destination_worker), 1);
}

FORCE_INLINE void issue_affine_pair_loopback(
    Noc& noc,
    uint32_t worker_index,
    DataflowBuffer& send_a,
    DataflowBuffer& send_b,
    DataflowBuffer& local_a,
    DataflowBuffer& local_b,
    uint32_t a_tiles,
    uint32_t b_tiles) {
    local_a.reserve_back(a_tiles);
    local_b.reserve_back(b_tiles);
    const uint32_t local_x = worker_x(worker_index);
    const uint32_t local_y = worker_y(worker_index);
    noc.async_write(
        send_a,
        UnicastEndpoint{},
        a_tiles * send_a.get_entry_size(),
        {},
        {.noc_x = local_x, .noc_y = local_y, .addr = local_a.get_write_ptr()});
    noc.async_write(
        send_b,
        UnicastEndpoint{},
        b_tiles * send_b.get_entry_size(),
        {},
        {.noc_x = local_x, .noc_y = local_y, .addr = local_b.get_write_ptr()});
}

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
    const uint32_t coordinator_x = worker_x(coordinator);
    const uint32_t coordinator_y = worker_y(coordinator);
    arrival.up(noc, coordinator_x, coordinator_y, 1);
    if (group == 0) {
        arrival.wait_min(completed_stages * G);
        for (uint32_t worker = coordinator; worker < coordinator + G; worker++) {
            release.up(noc, worker_x(worker), worker_y(worker), 1);
        }
        noc.async_atomic_barrier();
    }
    release.wait_min(completed_stages);
}

template <uint32_t Kt, uint32_t Vt, uint32_t BH, uint32_t G>
TT_KERNEL void dataflow(uint32_t worker_index, uint32_t group) {
    constexpr uint32_t affine_a_tiles = Kt * Kt;
    constexpr uint32_t affine_b_tiles = Kt * Vt;
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
    DataflowBuffer from_remote_affine(dfb::from_remote_affine);
    DataflowBuffer initial_state(dfb::initial_state);
    DataflowBuffer final(dfb::final);
    Noc noc;
    Semaphore<> ready(sem::ready);
    Semaphore<> arrival(sem::arrival);
    Semaphore<> release(sem::release);

    initial_a.reserve_back(affine_a_tiles);
    initial_b.reserve_back(affine_b_tiles);
    initial_state.reserve_back(affine_b_tiles);
    issue_tensor_block_read(noc, a_accessor, initial_a, worker_index * affine_a_tiles, affine_a_tiles);
    issue_tensor_block_read(noc, b_accessor, initial_b, worker_index * affine_b_tiles, affine_b_tiles);
    issue_tensor_block_read(
        noc, initial_state_accessor, initial_state, (worker_index / G) * affine_b_tiles, affine_b_tiles);
    noc.async_read_barrier();
    initial_a.push_back(affine_a_tiles);
    initial_b.push_back(affine_b_tiles);
    initial_state.push_back(affine_b_tiles);

    uint32_t completed_stages = 0;

    uint32_t expected_ready_events = 0;
    for (uint32_t distance = 1; distance < G; distance *= 2) {
        to_remote_a.wait_front(affine_a_tiles);
        to_remote_b.wait_front(affine_b_tiles);
        const bool sends = group + distance < G;
        const bool receives = group >= distance;
        if (sends) {
            issue_affine_pair_send<Kt, Vt>(noc, worker_index + distance, to_remote_a, to_remote_b, from_remote_affine);
        }
        if (receives) {
            issue_affine_pair_loopback(
                noc, worker_index, to_remote_a, to_remote_b, local_a, local_b, affine_a_tiles, affine_b_tiles);
        }
        if (sends) {
            // This retires both the remote send and a same-core loopback when this worker does both. Readiness is
            // published only after the remote writes complete.
            complete_affine_pair_send(noc, ready, worker_index + distance);
        } else if (receives) {
            noc.async_write_barrier();
        }
        if (receives) {
            local_a.push_back(affine_a_tiles);
            local_b.push_back(affine_b_tiles);
            from_remote_affine.reserve_back(affine_a_tiles + affine_b_tiles);
            // All NoC writes complete before this receiver frees the old outbound block.
            to_remote_a.pop_front(affine_a_tiles);
            to_remote_b.pop_front(affine_b_tiles);

            expected_ready_events++;
            ready.wait_min(expected_ready_events);
            from_remote_affine.push_back(affine_a_tiles + affine_b_tiles);

            // Compute must publish the replacement before any worker starts the next distance.
            to_remote_a.wait_front(affine_a_tiles);
            to_remote_b.wait_front(affine_b_tiles);
        }
        // Each head is an independent G-worker scan. Do not release its next NoC stage until all G workers have
        // consumed their remote buffers and produced the next prefix; otherwise that head can overwrite a mailbox
        // while compute is still reading it. Arrival and release semaphore targets stay monotonic across stages.
        synchronize_head_stage<G>(worker_index, group, completed_stages, noc, arrival, release);
    }

    to_remote_a.wait_front(affine_a_tiles);
    to_remote_b.wait_front(affine_b_tiles);
    if (group + 1 < G) {
        const uint32_t destination_worker = worker_index + 1;
        issue_affine_pair_send<Kt, Vt>(noc, destination_worker, to_remote_a, to_remote_b, from_remote_affine);
        complete_affine_pair_send(noc, ready, destination_worker);
    }
    // The final inclusive prefix is never reused after the exclusive neighbor shift.
    to_remote_a.pop_front(affine_a_tiles);
    to_remote_b.pop_front(affine_b_tiles);
    if (group > 0) {
        from_remote_affine.reserve_back(affine_a_tiles + affine_b_tiles);
        expected_ready_events++;
        ready.wait_min(expected_ready_events);
        from_remote_affine.push_back(affine_a_tiles + affine_b_tiles);
    }

    final.wait_front(affine_b_tiles);
    for (uint32_t tile = 0; tile < affine_b_tiles; tile++) {
        noc.async_write(
            final,
            output_accessor,
            final.get_entry_size(),
            {.offset_bytes = tile * final.get_entry_size()},
            {.page_id = worker_index * affine_b_tiles + tile});
    }
    noc.async_write_barrier();
    final.pop_front(affine_b_tiles);
}
