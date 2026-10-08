// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/dataflow/value_block_multicast.hpp"

// Read Kt rows of Vt tiles starting at row_page, rows row_stride pages apart, into the buffer in row-major order.
template <uint32_t Kt, uint32_t Vt, typename Accessor>
FORCE_INLINE void issue_value_slice_read(
    Noc& noc, const Accessor& accessor, DataflowBuffer& buffer, uint32_t row_page, uint32_t row_stride) {
    const uint32_t tile_bytes = buffer.get_entry_size();
    for (uint32_t row = 0; row < Kt; ++row) {
        for (uint32_t column = 0; column < Vt; ++column) {
            noc.async_read(
                accessor,
                buffer,
                tile_bytes,
                {.page_id = row_page + row * row_stride + column},
                {.offset_bytes = (row * Vt + column) * tile_bytes});
        }
    }
}

// One worker owns one value block of one head's state: it streams each chronological step's A and its own B
// columns to compute and writes its columns of the entry state and final carry that compute publishes.
template <
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Vt_full,
    uint32_t BH,
    uint32_t mcast_shared,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void dataflow(
    uint32_t head,
    uint32_t value_block,
    uint32_t peer_x0,
    uint32_t peer_y0,
    uint32_t peer_x1,
    uint32_t peer_y1,
    uint32_t receivers) {
    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t state_tiles = Kt * Vt;
    constexpr uint32_t row_tiles = Kt + Vt_full;
    // This block's columns of a head's [K, V] state: Kt rows of Vt tiles, strided by the full width.
    const uint32_t state_page = head * Kt * Vt_full + value_block * Vt;

    const auto transforms_accessor = TensorAccessor(tensor::transforms);
    const auto initial_state_accessor = TensorAccessor(tensor::initial_state);
    const auto entry_state_accessor = TensorAccessor(tensor::entry_state);
    const auto final_state_accessor = TensorAccessor(tensor::final_state);
    DataflowBuffer initial(dfb::initial);
    DataflowBuffer a(dfb::a);
    DataflowBuffer b(dfb::b);
    DataflowBuffer out(dfb::out);
    Noc noc;
    Semaphore ready(sem::ready);
    Semaphore valid(sem::valid);
    if constexpr (mcast_shared) {
        if (value_block == 0) {
            // set_multicast sources its payload from this local word; preset it to VALID once.
            valid.set(1);
        }
    }

    const auto write_state = [&](DataflowBuffer& source, const auto& destination) {
        const uint32_t tile_bytes = source.get_entry_size();
        for (uint32_t row = 0; row < Kt; ++row) {
            for (uint32_t column = 0; column < Vt; ++column) {
                noc.async_write(
                    source,
                    destination,
                    tile_bytes,
                    {.offset_bytes = (row * Vt + column) * tile_bytes},
                    {.page_id = state_page + row * Vt_full + column});
            }
        }
        noc.async_write_barrier();
    };

    // The initial state does not depend on the chronology, so its reads share one barrier with actual_start.
    initial.reserve_back(state_tiles);
    issue_value_slice_read<Kt, Vt>(noc, initial_state_accessor, initial, state_page, Vt_full);
    kda_chronology::Topology topology{};
    {
        DataflowBuffer chronology(dfb::chronology_compute);
        chronology.reserve_back(1);
        const auto actual_start = TensorAccessor(tensor::actual_start);
        noc.async_read(actual_start, chronology, sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chronology.get_write_ptr());
        topology = kda_chronology::derive(words[0], sp_rank, sp_size, local_rows);
        kda_chronology::store(words, topology);
        chronology.push_back(1);
    }
    // This rank's chronological index: its entry state is the carry after step entry_step - 1.
    const uint32_t entry_step = (topology.rank + sp_size - topology.first_rank) % sp_size;

    if (entry_step == 0) {
        // initial is used once and holds exactly state_tiles entries, so its read pointer (the NoC source) still
        // equals the write pointer the tiles were read into.
        write_state(initial, entry_state_accessor);
    }
    initial.push_back(state_tiles);

    for (uint32_t step = 0; step < sp_size; ++step) {
        const uint32_t rank = (topology.first_rank + step) % sp_size;
        const uint32_t transform_row = (rank * BH + head) * Kt;
        // B: this block's columns of the B half of the step's [A | B] rows.
        b.reserve_back(state_tiles);
        issue_value_slice_read<Kt, Vt>(
            noc, transforms_accessor, b, transform_row * row_tiles + Kt + value_block * Vt, row_tiles);
        // A: every value block needs the whole A half.
        const auto stage_a = [&]() {
            a.reserve_back(a_tiles);
            issue_value_slice_read<Kt, Kt>(noc, transforms_accessor, a, transform_row * row_tiles, row_tiles);
            return a.get_write_ptr();
        };
        if constexpr (!mcast_shared) {
            stage_a();
            noc.async_read_barrier();
            a.push_back(a_tiles);
        } else {
            SharedInput inputs[1] = {{&a, a_tiles, value_block == 0 ? stage_a() : 0}};
            if (value_block == 0) {
                multicast_shared(noc, inputs, ready, valid, peer_x0, peer_y0, peer_x1, peer_y1, receivers);
            } else {
                receive_shared(noc, inputs, ready, valid, peer_x0, peer_y0);
                noc.async_read_barrier();
            }
        }
        b.push_back(state_tiles);
        // Compute publishes the carry after step entry_step - 1; drain it once the next inputs are queued.
        if (entry_step != 0 && step == entry_step) {
            out.wait_front(state_tiles);
            write_state(out, entry_state_accessor);
            out.pop_front(state_tiles);
        }
    }
    out.wait_front(state_tiles);
    write_state(out, final_state_accessor);
    out.pop_front(state_tiles);
    if constexpr (mcast_shared) {
        // Retire the multicast writes and ready increments before exit; dispatch re-initializes both semaphores on
        // every launch.
        if (value_block == 0) {
            noc.async_write_barrier();
        } else {
            noc.async_atomic_barrier();
        }
    }
}
