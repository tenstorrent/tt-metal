// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_initial_a = 0;
constexpr uint32_t cb_initial_b = 1;
constexpr uint32_t cb_stage_a_ping = 2;
constexpr uint32_t cb_stage_b_ping = 3;
constexpr uint32_t cb_stage_a_pong = 4;
constexpr uint32_t cb_stage_b_pong = 5;
constexpr uint32_t cb_remote_a = 6;
constexpr uint32_t cb_remote_b = 7;
constexpr uint32_t cb_initial_state = 8;
constexpr uint32_t cb_output = 9;
constexpr uint32_t cb_stage_token = 11;
}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t BH = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr bool compose_only = get_compile_time_arg_val(4) == 1;
    constexpr auto a_args = TensorAccessorArgs<5>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto output_a_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto output_b_args = TensorAccessorArgs<output_a_args.next_compile_time_args_offset()>();
    constexpr uint32_t kk = Kt * Kt;
    constexpr uint32_t kv = Kt * Vt;
    static_assert(BH * G <= 128, "affine prefix coordinate table exceeds runtime-arg budget");

    const uint32_t worker_index = get_arg_val<uint32_t>(0);
    const uint32_t group = get_arg_val<uint32_t>(1);
    const uint32_t worker_count = get_arg_val<uint32_t>(2);
    const uint32_t a_addr = get_arg_val<uint32_t>(3);
    const uint32_t b_addr = get_arg_val<uint32_t>(4);
    const uint32_t s_addr = get_arg_val<uint32_t>(5);
    const uint32_t output_a_addr = get_arg_val<uint32_t>(6);
    const uint32_t output_b_addr = get_arg_val<uint32_t>(7);
    const uint32_t ready_sem = get_arg_val<uint32_t>(8);
    const uint32_t arrival_sem = get_arg_val<uint32_t>(9);
    const uint32_t release_sem = get_arg_val<uint32_t>(10);
    const uint32_t coordinator_x = get_arg_val<uint32_t>(11);
    const uint32_t coordinator_y = get_arg_val<uint32_t>(12);

    const uint32_t tile_bytes = get_tile_size(cb_initial_a);
    const auto a_accessor = TensorAccessor(a_args, a_addr, tile_bytes);
    const auto b_accessor = TensorAccessor(b_args, b_addr, tile_bytes);
    const auto s_accessor = TensorAccessor(s_args, s_addr, tile_bytes);
    const auto output_a_accessor = TensorAccessor(output_a_args, output_a_addr, tile_bytes);
    const auto output_b_accessor = TensorAccessor(output_b_args, output_b_addr, tile_bytes);
    Noc noc;

    auto worker_x = [&](uint32_t worker) { return get_arg_val<uint32_t>(13 + 2 * worker); };
    auto worker_y = [&](uint32_t worker) { return get_arg_val<uint32_t>(14 + 2 * worker); };
    auto read_tiles = [&](const auto& accessor, uint32_t cb_id, uint32_t page, uint32_t tiles) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(tiles);
        for (uint32_t tile = 0; tile < tiles; tile++) {
            noc.async_read(accessor, cb, tile_bytes, {.page_id = page + tile}, {.offset_bytes = tile * tile_bytes});
        }
        noc.async_read_barrier();
        cb.push_back(tiles);
    };
    read_tiles(a_accessor, cb_initial_a, worker_index * kk, kk);
    read_tiles(b_accessor, cb_initial_b, worker_index * kv, kv);
    if constexpr (!compose_only) {
        read_tiles(s_accessor, cb_initial_state, (worker_index / G) * kv, kv);
    }

    uint32_t completed_stages = 0;
    auto stage_barrier = [&] {
        completed_stages++;
        noc_semaphore_inc(get_noc_addr(coordinator_x, coordinator_y, get_semaphore(arrival_sem)), 1);
        if (worker_index == 0) {
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(arrival_sem)),
                completed_stages * worker_count);
            for (uint32_t worker = 0; worker < worker_count; worker++) {
                noc_semaphore_inc(get_noc_addr(worker_x(worker), worker_y(worker), get_semaphore(release_sem)), 1);
            }
            noc_async_atomic_barrier();
        }
        noc_semaphore_wait_min(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(release_sem)), completed_stages);
    };
    auto send_pair = [&](uint32_t target, uint32_t current_a, uint32_t current_b) {
        noc_async_write(
            get_read_ptr(current_a),
            get_noc_addr(worker_x(target), worker_y(target), get_write_ptr(cb_remote_a)),
            kk * tile_bytes);
        noc_async_write(
            get_read_ptr(current_b),
            get_noc_addr(worker_x(target), worker_y(target), get_write_ptr(cb_remote_b)),
            kv * tile_bytes);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(worker_x(target), worker_y(target), get_semaphore(ready_sem)), 1);
    };
    auto receive_pair = [&] {
        CircularBuffer(cb_remote_a).reserve_back(kk);
        CircularBuffer(cb_remote_b).reserve_back(kv);
        CircularBuffer(cb_remote_a).push_back(kk);
        CircularBuffer(cb_remote_b).push_back(kv);
    };

    uint32_t ready_target = 0;
    bool ping = false;
    for (uint32_t distance = 1; distance < G; distance *= 2) {
        const uint32_t current_a = ping ? cb_stage_a_pong : cb_stage_a_ping;
        const uint32_t current_b = ping ? cb_stage_b_pong : cb_stage_b_ping;
        const uint32_t next_a = ping ? cb_stage_a_ping : cb_stage_a_pong;
        const uint32_t next_b = ping ? cb_stage_b_ping : cb_stage_b_pong;
        CircularBuffer(current_a).wait_front(kk);
        CircularBuffer(current_b).wait_front(kv);
        if (group + distance < G) {
            send_pair(worker_index + distance, current_a, current_b);
        }
        if (group >= distance) {
            ready_target++;
            noc_semaphore_wait_min(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem)), ready_target);
            receive_pair();
            CircularBuffer(cb_stage_token).reserve_back(1);
            CircularBuffer(cb_stage_token).push_back(1);
        }
        stage_barrier();
        if (group >= distance) {
            CircularBuffer(next_a).wait_front(kk);
            CircularBuffer(next_b).wait_front(kv);
            ping = !ping;
        }
    }

    const uint32_t current_a = ping ? cb_stage_a_pong : cb_stage_a_ping;
    const uint32_t current_b = ping ? cb_stage_b_pong : cb_stage_b_ping;
    CircularBuffer(current_a).wait_front(kk);
    CircularBuffer(current_b).wait_front(kv);
    if constexpr (compose_only) {
        if (group + 1 == G) {
            CircularBuffer prefix_a(current_a);
            CircularBuffer prefix_b(current_b);
            const uint32_t head = worker_index / G;
            for (uint32_t tile = 0; tile < kk; tile++) {
                noc.async_write(
                    prefix_a,
                    output_a_accessor,
                    tile_bytes,
                    {.offset_bytes = tile * tile_bytes},
                    {.page_id = head * kk + tile});
            }
            for (uint32_t tile = 0; tile < kv; tile++) {
                noc.async_write(
                    prefix_b,
                    output_b_accessor,
                    tile_bytes,
                    {.offset_bytes = tile * tile_bytes},
                    {.page_id = head * kv + tile});
            }
            noc.async_write_barrier();
        }
        return;
    }
    if (group + 1 < G) {
        send_pair(worker_index + 1, current_a, current_b);
    }
    if (group > 0) {
        ready_target++;
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_sem)), ready_target);
        receive_pair();
    }
    CircularBuffer(cb_stage_token).reserve_back(1);
    CircularBuffer(cb_stage_token).push_back(1);

    CircularBuffer output(cb_output);
    output.wait_front(kv);
    for (uint32_t tile = 0; tile < kv; tile++) {
        noc.async_write(
            output,
            output_a_accessor,
            tile_bytes,
            {.offset_bytes = tile * tile_bytes},
            {.page_id = worker_index * kv + tile});
    }
    noc.async_write_barrier();
    output.pop_front(kv);
}
