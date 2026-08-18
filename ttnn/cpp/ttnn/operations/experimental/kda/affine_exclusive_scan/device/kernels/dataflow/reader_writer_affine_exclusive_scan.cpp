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
    const auto initial_state_accessor = TensorAccessor(tensor::initial_state);
    const auto output_accessor = TensorAccessor(tensor::output);
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer send_a_ping(dfb::send_a_ping);
    DataflowBuffer send_b_ping(dfb::send_b_ping);
    DataflowBuffer send_a_pong(dfb::send_a_pong);
    DataflowBuffer send_b_pong(dfb::send_b_pong);
    DataflowBuffer remote_a(dfb::remote_a);
    DataflowBuffer remote_b(dfb::remote_b);
    DataflowBuffer initial_state(dfb::initial_state);
    DataflowBuffer final(dfb::final);
    DataflowBuffer stage_token(dfb::stage_token);
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
    auto send_pair = [&](uint32_t target, DataflowBuffer& current_a, DataflowBuffer& current_b) {
        const uint32_t target_x = worker_x(target);
        const uint32_t target_y = worker_y(target);
        noc.async_write(
            current_a,
            UnicastEndpoint{},
            kk * current_a.get_entry_size(),
            {},
            {.noc_x = target_x, .noc_y = target_y, .addr = remote_a.get_write_ptr()});
        noc.async_write(
            current_b,
            UnicastEndpoint{},
            kv * current_b.get_entry_size(),
            {},
            {.noc_x = target_x, .noc_y = target_y, .addr = remote_b.get_write_ptr()});
        noc.async_write_barrier();
        ready.up(noc, target_x, target_y, 1);
    };
    auto receive_pair = [&] {
        remote_a.reserve_back(kk);
        remote_b.reserve_back(kv);
        remote_a.push_back(kk);
        remote_b.push_back(kv);
    };

    uint32_t ready_target = 0;
    bool ping = false;
    for (uint32_t distance = 1; distance < G; distance *= 2) {
        DataflowBuffer& current_a = ping ? send_a_pong : send_a_ping;
        DataflowBuffer& current_b = ping ? send_b_pong : send_b_ping;
        DataflowBuffer& next_a = ping ? send_a_ping : send_a_pong;
        DataflowBuffer& next_b = ping ? send_b_ping : send_b_pong;
        current_a.wait_front(kk);
        current_b.wait_front(kv);
        if (group + distance < G) {
            send_pair(worker_index + distance, current_a, current_b);
        }
        if (group >= distance) {
            ready_target++;
            ready.wait_min(ready_target);
            receive_pair();
            stage_token.reserve_back(1);
            stage_token.push_back(1);
            next_a.wait_front(kk);
            next_b.wait_front(kv);
            current_a.pop_front(kk);
            current_b.pop_front(kv);
            ping = !ping;
        }
        // Do not release the next NoC stage until every receiver has consumed the remote buffers and produced its
        // next prefix. Otherwise the following stage can overwrite the remote buffers while compute is reading them.
        stage_barrier();
    }

    DataflowBuffer& current_a = ping ? send_a_pong : send_a_ping;
    DataflowBuffer& current_b = ping ? send_b_pong : send_b_ping;
    current_a.wait_front(kk);
    current_b.wait_front(kv);
    if (group + 1 < G) {
        send_pair(worker_index + 1, current_a, current_b);
    }
    if (group > 0) {
        ready_target++;
        ready.wait_min(ready_target);
        receive_pair();
    }
    stage_token.reserve_back(1);
    stage_token.push_back(1);

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
