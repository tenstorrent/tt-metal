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

template <typename ReadySem>
FORCE_INLINE void send_affine_pair(
    Noc& noc,
    ReadySem& ready,
    uint32_t target,
    DataflowBuffer& send_a,
    DataflowBuffer& send_b,
    DataflowBuffer& remote_a,
    DataflowBuffer& remote_b,
    uint32_t a_tiles,
    uint32_t b_tiles) {
    const uint32_t target_x = worker_x(target);
    const uint32_t target_y = worker_y(target);
    noc.async_write(
        send_a,
        UnicastEndpoint{},
        a_tiles * send_a.get_entry_size(),
        {},
        {.noc_x = target_x, .noc_y = target_y, .addr = remote_a.get_write_ptr()});
    noc.async_write(
        send_b,
        UnicastEndpoint{},
        b_tiles * send_b.get_entry_size(),
        {},
        {.noc_x = target_x, .noc_y = target_y, .addr = remote_b.get_write_ptr()});
    noc.async_write_barrier();
    ready.up(noc, target_x, target_y, 1);
}

template <uint32_t G, typename ArrivalSem, typename ReleaseSem>
FORCE_INLINE void synchronize_head_stage(
    uint32_t worker_index,
    uint32_t group,
    uint32_t& completed_stages,
    Noc& noc,
    ArrivalSem& arrival,
    ReleaseSem& release) {
    completed_stages++;
    const uint32_t coordinator = worker_index - group;
    const uint32_t coordinator_x = worker_x(coordinator);
    const uint32_t coordinator_y = worker_y(coordinator);
    arrival.up(noc, coordinator_x, coordinator_y, 1);
    if (group == 0) {
        arrival.wait_min(completed_stages * G);
        for (uint32_t worker = coordinator; worker < coordinator + G; worker++) {
            const uint32_t target_x = worker_x(worker);
            const uint32_t target_y = worker_y(worker);
            release.up(noc, target_x, target_y, 1);
        }
        noc.async_atomic_barrier();
    }
    release.wait_min(completed_stages);
}

template <uint32_t Kt, uint32_t Vt, uint32_t G>
TT_KERNEL void dataflow(uint32_t worker_index, uint32_t group) {
    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t b_tiles = Kt * Vt;

    const auto a_accessor = TensorAccessor(tensor::a);
    const auto b_accessor = TensorAccessor(tensor::b);
    const auto output_a_accessor = TensorAccessor(tensor::output_a);
    const auto output_b_accessor = TensorAccessor(tensor::output_b);
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer send_a(dfb::send_a);
    DataflowBuffer send_b(dfb::send_b);
    DataflowBuffer remote_a(dfb::remote_a);
    DataflowBuffer remote_b(dfb::remote_b);
    Noc noc;
    Semaphore ready(sem::ready);
    Semaphore arrival(sem::arrival);
    Semaphore release(sem::release);

    initial_a.reserve_back(a_tiles);
    initial_b.reserve_back(b_tiles);
    issue_tensor_block_read(noc, a_accessor, initial_a, worker_index * a_tiles, a_tiles);
    issue_tensor_block_read(noc, b_accessor, initial_b, worker_index * b_tiles, b_tiles);
    noc.async_read_barrier();
    initial_a.push_back(a_tiles);
    initial_b.push_back(b_tiles);

    uint32_t completed_stages = 0;

    uint32_t ready_target = 0;
    for (uint32_t distance = 1; distance < G; distance *= 2) {
        send_a.wait_front(a_tiles);
        send_b.wait_front(b_tiles);
        if (group + distance < G) {
            send_affine_pair(noc, ready, worker_index + distance, send_a, send_b, remote_a, remote_b, a_tiles, b_tiles);
        }
        if (group >= distance) {
            remote_a.reserve_back(a_tiles);
            remote_b.reserve_back(b_tiles);
            ready_target++;
            ready.wait_min(ready_target);
            remote_a.push_back(a_tiles);
            remote_b.push_back(b_tiles);
            send_a.wait_front(2 * a_tiles);
            send_b.wait_front(2 * b_tiles);
            send_a.pop_front(a_tiles);
            send_b.pop_front(b_tiles);
        }
        // Do not release the next NoC stage until every receiver has consumed the remote buffers and produced its
        // next prefix. Otherwise the following stage can overwrite the remote buffers while compute is reading them.
        synchronize_head_stage<G>(worker_index, group, completed_stages, noc, arrival, release);
    }

    send_a.wait_front(a_tiles);
    send_b.wait_front(b_tiles);
    if (group + 1 == G) {
        const uint32_t head = worker_index / G;
        for (uint32_t tile = 0; tile < a_tiles; tile++) {
            noc.async_write(
                send_a,
                output_a_accessor,
                send_a.get_entry_size(),
                {.offset_bytes = tile * send_a.get_entry_size()},
                {.page_id = head * a_tiles + tile});
        }
        for (uint32_t tile = 0; tile < b_tiles; tile++) {
            noc.async_write(
                send_b,
                output_b_accessor,
                send_b.get_entry_size(),
                {.offset_bytes = tile * send_b.get_entry_size()},
                {.page_id = head * b_tiles + tile});
        }
        noc.async_write_barrier();
    }
}
