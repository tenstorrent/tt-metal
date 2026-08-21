// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Kt, uint32_t Vt, uint32_t BH, uint32_t G>
TT_KERNEL void dataflow(uint32_t worker_index, uint32_t group, uint32_t coordinator_x, uint32_t coordinator_y) {
    constexpr uint32_t kk = Kt * Kt;
    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t worker_count = BH * G;
    static_assert(worker_count <= 128, "affine prefix coordinate table exceeds runtime-arg budget");

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

    uint32_t completed_stages = 0;
    auto stage_barrier = [&] {
        completed_stages++;
        arrival.up(noc, coordinator_x, coordinator_y, 1);
        if (worker_index == 0) {
            arrival.wait_min(completed_stages * worker_count);
            for (uint32_t worker = 0; worker < worker_count; worker++) {
                release.up(noc, worker_x(worker), worker_y(worker), 1);
            }
            noc.async_atomic_barrier();
        }
        release.wait_min(completed_stages);
    };
    auto send_pair = [&](uint32_t target, DataflowBuffer& send_a, DataflowBuffer& send_b) {
        const uint32_t target_x = worker_x(target);
        const uint32_t target_y = worker_y(target);
        noc.async_write(
            send_a,
            UnicastEndpoint{},
            kk * send_a.get_entry_size(),
            {},
            {.noc_x = target_x, .noc_y = target_y, .addr = remote_a.get_write_ptr()});
        noc.async_write(
            send_b,
            UnicastEndpoint{},
            kv * send_b.get_entry_size(),
            {},
            {.noc_x = target_x, .noc_y = target_y, .addr = remote_b.get_write_ptr()});
        noc.async_write_barrier();
        ready.up(noc, target_x, target_y, 1);
    };

    uint32_t ready_target = 0;
    for (uint32_t distance = 1; distance < G; distance *= 2) {
        send_a.wait_front(kk);
        send_b.wait_front(kv);
        if (group + distance < G) {
            send_pair(worker_index + distance, send_a, send_b);
        }
        if (group >= distance) {
            remote_a.reserve_back(kk);
            remote_b.reserve_back(kv);
            ready_target++;
            ready.wait_min(ready_target);
            remote_a.push_back(kk);
            remote_b.push_back(kv);
            send_a.wait_front(2 * kk);
            send_b.wait_front(2 * kv);
            send_a.pop_front(kk);
            send_b.pop_front(kv);
        }
        // Do not release the next NoC stage until every receiver has consumed the remote buffers and produced its
        // next prefix. Otherwise the following stage can overwrite the remote buffers while compute is reading them.
        stage_barrier();
    }

    send_a.wait_front(kk);
    send_b.wait_front(kv);
    if (group + 1 == G) {
        const uint32_t head = worker_index / G;
        for (uint32_t tile = 0; tile < kk; tile++) {
            noc.async_write(
                send_a,
                output_a_accessor,
                send_a.get_entry_size(),
                {.offset_bytes = tile * send_a.get_entry_size()},
                {.page_id = head * kk + tile});
        }
        for (uint32_t tile = 0; tile < kv; tile++) {
            noc.async_write(
                send_b,
                output_b_accessor,
                send_b.get_entry_size(),
                {.offset_bytes = tile * send_b.get_entry_size()},
                {.page_id = head * kv + tile});
        }
        noc.async_write_barrier();
    }
}
