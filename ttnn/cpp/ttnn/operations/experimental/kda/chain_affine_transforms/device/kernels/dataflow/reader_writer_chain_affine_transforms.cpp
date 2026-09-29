// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// One worker owns a head's column block of the state: it streams each chronological step's A rows and the
// matching B columns to compute, and writes the entry state and final carry that compute publishes.
template <uint32_t Kt, uint32_t Vt, uint32_t Vc, uint32_t BH, uint32_t sp_rank, uint32_t sp_size, uint32_t local_rows>
TT_KERNEL void dataflow(uint32_t head, uint32_t value_block) {
    constexpr uint32_t a_tiles = Kt * Kt;
    constexpr uint32_t state_tiles = Kt * Vc;
    constexpr uint32_t row_tiles = Kt + Vt;

    const auto transforms = TensorAccessor(tensor::transforms);
    const auto initial_state = TensorAccessor(tensor::initial_state);
    const auto entry_state = TensorAccessor(tensor::entry_state);
    const auto final_state = TensorAccessor(tensor::final_state);
    DataflowBuffer initial(dfb::initial);
    DataflowBuffer a(dfb::a);
    DataflowBuffer b(dfb::b);
    DataflowBuffer out(dfb::out);
    Noc noc;

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
    // This rank's chronological index: its entry state is the carry before its own step.
    const uint32_t entry_step = (sp_rank + sp_size - topology.first_rank) % sp_size;

    const auto state_page = [head, value_block](uint32_t row, uint32_t column) {
        return (head * Kt + row) * Vt + value_block * Vc + column;
    };
    const auto write_state = [&](DataflowBuffer& source, const auto& destination) {
        const uint32_t tile_bytes = source.get_entry_size();
        for (uint32_t row = 0; row < Kt; ++row) {
            for (uint32_t column = 0; column < Vc; ++column) {
                noc.async_write(
                    source,
                    destination,
                    tile_bytes,
                    {.offset_bytes = (row * Vc + column) * tile_bytes},
                    {.page_id = state_page(row, column)});
            }
        }
        noc.async_write_barrier();
    };

    initial.reserve_back(state_tiles);
    {
        const uint32_t tile_bytes = initial.get_entry_size();
        for (uint32_t row = 0; row < Kt; ++row) {
            for (uint32_t column = 0; column < Vc; ++column) {
                noc.async_read(
                    initial_state,
                    initial,
                    tile_bytes,
                    {.page_id = state_page(row, column)},
                    {.offset_bytes = (row * Vc + column) * tile_bytes});
            }
        }
        noc.async_read_barrier();
    }
    if (entry_step == 0) {
        write_state(initial, entry_state);
    }
    initial.push_back(state_tiles);

    for (uint32_t step = 0; step < sp_size; ++step) {
        const uint32_t rank = (topology.first_rank + step) % sp_size;
        const uint32_t transform_row = (rank * BH + head) * Kt;
        a.reserve_back(a_tiles);
        b.reserve_back(state_tiles);
        const uint32_t a_bytes = a.get_entry_size();
        const uint32_t b_bytes = b.get_entry_size();
        for (uint32_t row = 0; row < Kt; ++row) {
            const uint32_t row_page = (transform_row + row) * row_tiles;
            for (uint32_t column = 0; column < Kt; ++column) {
                noc.async_read(
                    transforms,
                    a,
                    a_bytes,
                    {.page_id = row_page + column},
                    {.offset_bytes = (row * Kt + column) * a_bytes});
            }
            for (uint32_t column = 0; column < Vc; ++column) {
                noc.async_read(
                    transforms,
                    b,
                    b_bytes,
                    {.page_id = row_page + Kt + value_block * Vc + column},
                    {.offset_bytes = (row * Vc + column) * b_bytes});
            }
        }
        noc.async_read_barrier();
        a.push_back(a_tiles);
        b.push_back(state_tiles);
        // Compute publishes the carry after step entry_step - 1; drain it once the next inputs are queued.
        if (step != 0 && step == entry_step) {
            out.wait_front(state_tiles);
            write_state(out, entry_state);
            out.pop_front(state_tiles);
        }
    }
    out.wait_front(state_tiles);
    write_state(out, final_state);
    out.pop_front(state_tiles);
}
