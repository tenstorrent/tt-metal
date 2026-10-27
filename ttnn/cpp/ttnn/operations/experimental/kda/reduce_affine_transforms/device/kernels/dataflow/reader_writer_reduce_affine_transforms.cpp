// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/dataflow/packed_summary.hpp"

#include <cstdint>

#include "api/tile_constants.h"

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
FORCE_INLINE uint32_t worker_x(uint32_t worker) { return get_common_vararg(2 * worker); }
FORCE_INLINE uint32_t worker_y(uint32_t worker) { return get_common_vararg(2 * worker + 1); }

template <typename ReadySem, typename SendA, typename SendB, typename RemoteA, typename RemoteB>
FORCE_INLINE void send_affine_pair(
    Noc& noc,
    ReadySem& ready,
    uint32_t target,
    SendA& send_a,
    SendB& send_b,
    RemoteA& remote_a,
    RemoteB& remote_b,
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

// Empty ranks contribute the affine identity to the distributed prefix: (0, 0) in complement form.
// The unused remote buffer provides one zeroed scratch tile; compute has exited.
template <uint32_t Kt, uint32_t Vt, typename OutputAccessor, typename DFB>
FORCE_INLINE void write_identity_transform(
    Noc& noc, DFB& scratch, uint32_t head, const OutputAccessor& output, uint32_t page_bytes) {
    constexpr uint32_t head_tiles = Kt * (Kt + Vt);
    scratch.reserve_back(1);
    noc.async_write_zeros(scratch, scratch.get_entry_size());
    noc.write_zeros_l1_barrier();
    for (uint32_t tile = 0; tile < head_tiles; ++tile) {
        noc.async_write(scratch, output, page_bytes, {}, {.page_id = head * head_tiles + tile});
    }
    noc.async_write_barrier();
}

// Write the published head transform, Kt rows of E (Kt tiles each) then Kt rows of B (Vt tiles each), into the
// packed [1, B*H, K, K+V] rows [E | B].
template <uint32_t Kt, uint32_t Vt, typename OutputAccessor, typename DFB>
FORCE_INLINE void write_published_transform(Noc& noc, DFB& published, uint32_t head, const OutputAccessor& output) {
    constexpr uint32_t row_tiles = Kt + Vt;
    const uint32_t tile_bytes = published.get_entry_size();
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t columns = half == 0 ? Kt : Vt;
        const uint32_t first_column = half == 0 ? 0 : Kt;
        for (uint32_t row = 0; row < Kt; ++row) {
            published.wait_front(columns);
            for (uint32_t column = 0; column < columns; ++column) {
                noc.async_write(
                    published,
                    output,
                    tile_bytes,
                    {.offset_bytes = column * tile_bytes},
                    {.page_id = (head * Kt + row) * row_tiles + first_column + column});
            }
            noc.async_write_barrier();
            published.pop_front(columns);
        }
    }
}

template <uint32_t G, typename ArrivalSem, typename ReleaseSem>
FORCE_INLINE void synchronize_head_stage(
    uint32_t worker_index,
    uint32_t group,
    uint32_t active,
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
        arrival.wait_min(completed_stages * active);
        for (uint32_t worker = coordinator; worker < coordinator + active; worker++) {
            const uint32_t target_x = worker_x(worker);
            const uint32_t target_y = worker_y(worker);
            release.up(noc, target_x, target_y, 1);
        }
        noc.async_atomic_barrier();
    }
    release.wait_min(completed_stages);
}

template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t G,
    uint32_t has_actual_end,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void dataflow(uint32_t worker_index, uint32_t group) {
    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t b_tiles = Kt * Vt;

    const auto summaries_accessor = TensorAccessor(tensor::summaries);
    const auto output_accessor = TensorAccessor(tensor::output);
    DataflowBuffer initial_a(dfb::initial_a);
    DataflowBuffer initial_b(dfb::initial_b);
    DataflowBuffer send_a(dfb::send_a);
    DataflowBuffer send_b(dfb::send_b);
    DataflowBuffer remote_a(dfb::remote_a);
    DataflowBuffer remote_b(dfb::remote_b);
    DataflowBuffer identity(dfb::identity);
    DataflowBuffer published(dfb::output);
    Noc noc;
    Semaphore ready(sem::ready);
    Semaphore arrival(sem::arrival);
    Semaphore release(sem::release);

    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        chronology.reserve_back(1);
        const auto actual_start = TensorAccessor(tensor::actual_start);
        noc.async_read(actual_start, chronology, sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chronology.get_write_ptr());
        const uint32_t start = words[0];
        if constexpr (has_actual_end) {
            const auto end = TensorAccessor(*tensor::get_token_if_present<"actual_end">());
            noc.async_read(end, chronology, sizeof(uint32_t), {.page_id = 0}, {});
            noc.async_read_barrier();
            topology = kda_chronology::derive_interval(start, words[0], sp_rank, sp_size, local_rows);
        } else {
            topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
        }
        kda_chronology::store(words, topology);
        chronology.push_back(1);
    }
    const uint32_t active = topology.head_groups(G);
    const uint32_t head = worker_index / G;
    if (active == 0) {
        if (group == 0) {
            write_identity_transform<Kt, Vt>(noc, remote_a, head, output_accessor, published.get_entry_size());
        }
        return;
    }
    if (group >= active) {
        return;
    }
    // The last active group composes the whole head and publishes it, in its last step. Every group forms A = E + I.
    const bool publishes = group + 1 == active;
    kda::seed_identity_tile(noc, identity);

    initial_a.reserve_back(a_tiles);
    initial_b.reserve_back(b_tiles);
    kda::read_summary_transform<Kt, Vt>(noc, summaries_accessor, initial_a, initial_b, worker_index);
    noc.async_read_barrier();
    initial_a.push_back(a_tiles);
    initial_b.push_back(b_tiles);

    uint32_t completed_stages = 0;

    uint32_t ready_target = 0;
    for (uint32_t distance = 1; distance < active; distance *= 2) {
        send_a.wait_front(a_tiles);
        send_b.wait_front(b_tiles);
        if (group + distance < active) {
            send_affine_pair(noc, ready, worker_index + distance, send_a, send_b, remote_a, remote_b, a_tiles, b_tiles);
        }
        if (group >= distance) {
            remote_a.reserve_back(a_tiles);
            remote_b.reserve_back(b_tiles);
            ready_target++;
            ready.wait_min(ready_target);
            remote_a.push_back(a_tiles);
            remote_b.push_back(b_tiles);
            // Compute publishes before it releases the next prefix to send, so drain the published rows first.
            if (publishes && 2 * distance >= active) {
                write_published_transform<Kt, Vt>(noc, published, head, output_accessor);
            }
            send_a.wait_front(2 * a_tiles);
            send_b.wait_front(2 * b_tiles);
            send_a.pop_front(a_tiles);
            send_b.pop_front(b_tiles);
        }
        // Do not release the next NoC stage until every receiver has consumed the remote buffers and produced its
        // next prefix. Otherwise the following stage can overwrite the remote buffers while compute is reading them.
        synchronize_head_stage<G>(worker_index, group, active, completed_stages, noc, arrival, release);
    }

    // A single active group publishes its copied summary; there is no composition stage.
    if (publishes && active == 1) {
        write_published_transform<Kt, Vt>(noc, published, head, output_accessor);
    }
    send_a.wait_front(a_tiles);
    send_b.wait_front(b_tiles);
}
