// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/semaphore.h"
#include "experimental/kernel_args.h"

// Entry correction reader: the chronology (this core's valid chunks and the reset chunk), the head's seeds, and the
// rows of E = A - I, B and the seed this core forms the final state of.
// final_stride: a head's cores each form every final_stride-th tile row of the final state.
// streamed: interleaved terms, read a chunk row of F(0) and F(I) at a time; otherwise they are in place in the shard.
// seed_receivers: the head's other cores, which the head's first core writes the seeds to (varargs: NoC x | y << 16).
template <
    uint32_t kt,
    uint32_t seed_receivers,
    uint32_t core_chunks,
    uint32_t final_stride,
    uint32_t streamed,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows,
    uint32_t has_actual_end>
TT_KERNEL void reader(
    uint32_t head, uint32_t first_chunk, uint32_t last_chunk, uint32_t final_row, uint32_t seed_sender) {
    constexpr uint32_t state_tiles = kt * kt;
    // final_row: the first tile row of the final state this core forms (kt: none).
    const bool owns_final = final_row < kt;
    DataflowBuffer seed(dfb::seed);
    DataflowBuffer tail(dfb::tail);
    DataflowBuffer terms(dfb::terms);
    DataflowBuffer e(dfb::e);
    DataflowBuffer b(dfb::b);
    DataflowBuffer seed_row(dfb::seed_row);
    DataflowBuffer ctl(dfb::ctl);
    DataflowBuffer ctl_writer(dfb::ctl_writer);
    Noc noc;

    // The chronology: valid chunks and the reset chunk (where the tail's seed takes over).
    ctl.reserve_back(1);
    ctl_writer.reserve_back(1);
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl.get_write_ptr());
    auto* writer_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctl_writer.get_write_ptr());
    // Both bounds in one round trip, each into its own channel's slot.
    noc.async_read(TensorAccessor(tensor::actual_start), ctl, sizeof(uint32_t), {.page_id = 0}, {});
    if constexpr (has_actual_end) {
        noc.async_read(
            TensorAccessor(*tensor::get_token_if_present<"actual_end">()),
            ctl_writer,
            sizeof(uint32_t),
            {.page_id = 0},
            {});
    }
    noc.async_read_barrier();
    const uint32_t actual_start = words[0];
    kda_chronology::Topology topology{};
    if constexpr (has_actual_end) {
        topology = kda_chronology::derive_interval(actual_start, writer_words[0], sp_rank, sp_size, local_rows);
    } else {
        topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    }
    const uint32_t valid = topology.valid_chunks(0, 1);
    const uint32_t reset = topology.reset_chunk(0, 1);
    const uint32_t end_chunk = last_chunk < valid ? last_chunk : valid;
    const uint32_t chunks = end_chunk > first_chunk ? end_chunk - first_chunk : 0;
    // Chunks before the reset take the group's seed, the rest the tail's.
    uint32_t group_chunks = chunks;
    if (reset != 0) {
        group_chunks = reset <= first_chunk ? 0 : (reset - first_chunk < chunks ? reset - first_chunk : chunks);
    }
    const uint32_t final = owns_final && valid != 0;
    const uint32_t final_tail = final && reset != 0;
    // Every core of the head takes the group's seed, and the tail's after a reset.
    const uint32_t need_group = valid != 0;
    const uint32_t need_tail = valid != 0 && reset != 0;
    words[0] = chunks;
    words[1] = group_chunks;
    words[2] = final;
    words[3] = final_tail;
    words[4] = need_group;
    words[5] = need_tail;
    words[6] = final_row;
    words[7] = final_stride;
    writer_words[0] = chunks;
    writer_words[1] = final;
    writer_words[2] = final_row;
    writer_words[3] = final_stride;
    ctl_writer.push_back(1);
    ctl.push_back(1);

    // The head's first core reads the head's seeds and writes them to the head's other cores, at the same buffer
    // address on each. Every core reading its own copy instead makes the K3 layer 17 us slower on Galaxy.
    Semaphore landed(sem::seed_landed);
    const uint32_t seeds = need_group + need_tail;
    if (seed_sender) {
        const auto send = [&](auto& buffer, const auto& states) {
            buffer.reserve_back(state_tiles);
            const uint32_t tile = buffer.get_entry_size();
            for (uint32_t i = 0; i < state_tiles; ++i) {
                noc.async_read(states, buffer, tile, {.page_id = head * state_tiles + i}, {.offset_bytes = i * tile});
            }
            noc.async_read_barrier();
            const uint32_t local = buffer.get_write_ptr();
            for (uint32_t d = 0; d < seed_receivers; ++d) {
                const uint32_t xy = get_vararg(d);
                noc_async_write(local, get_noc_addr(xy & 0xFFFF, xy >> 16, local), state_tiles * tile);
            }
            buffer.push_back(state_tiles);
        };
        if (need_group) {
            send(seed, TensorAccessor(tensor::entry));
        }
        if (need_tail) {
            send(tail, TensorAccessor(tensor::tail_entry));
        }
        if (seeds != 0) {
            noc_async_write_barrier();
            for (uint32_t d = 0; d < seed_receivers; ++d) {
                const uint32_t xy = get_vararg(d);
                landed.up(noc, xy & 0xFFFF, xy >> 16, seeds);
            }
            noc_async_atomic_barrier();
        }
    } else if (seeds != 0) {
        landed.wait_min(seeds);
        if (need_group) {
            seed.reserve_back(state_tiles);
            seed.push_back(state_tiles);
        }
        if (need_tail) {
            tail.reserve_back(state_tiles);
            tail.push_back(state_tiles);
        }
    }
    if constexpr (streamed) {
        // Head h's F(0) rows are tile rows h * 2 * num_chunks + chunk of the terms, its F(I) rows num_chunks further.
        // Each chunk's F(0) row then its F(I) row go to the terms buffer together.
        constexpr uint32_t num_chunks = local_rows / 32;
        const auto accessor = TensorAccessor(tensor::terms);
        const uint32_t tile = terms.get_entry_size();
        for (uint32_t c = 0; c < chunks; ++c) {
            const uint32_t zero_page = (head * 2 * num_chunks + first_chunk + c) * kt;
            terms.reserve_back(2 * kt);
            for (uint32_t j = 0; j < kt; ++j) {
                noc.async_read(accessor, terms, tile, {.page_id = zero_page + j}, {.offset_bytes = j * tile});
                noc.async_read(
                    accessor,
                    terms,
                    tile,
                    {.page_id = zero_page + num_chunks * kt + j},
                    {.offset_bytes = (kt + j) * tile});
            }
            noc.async_read_barrier();
            terms.push_back(2 * kt);
        }
    } else if (chunks != 0) {
        // This core's chunk rows are in place in its shard of the emitted terms.
        terms.reserve_back(2 * core_chunks * kt);
        terms.push_back(2 * core_chunks * kt);
    }
    if (final) {
        // Tile rows of E, of B and of the seed: each row of the final state is seed[row] + (E[row] @ seed + B[row]).
        const auto e_states = TensorAccessor(tensor::final_e);
        const auto b_states = TensorAccessor(tensor::final_b);
        const auto seed_states = TensorAccessor(tensor::entry);
        const auto tail_states = TensorAccessor(tensor::tail_entry);
        const uint32_t tile = b.get_entry_size();
        for (uint32_t row = final_row; row < kt; row += final_stride) {
            e.reserve_back(kt);
            b.reserve_back(kt);
            seed_row.reserve_back(kt);
            for (uint32_t c = 0; c < kt; ++c) {
                const uint32_t page = head * state_tiles + row * kt + c;
                noc.async_read(e_states, e, tile, {.page_id = page}, {.offset_bytes = c * tile});
                noc.async_read(b_states, b, tile, {.page_id = page}, {.offset_bytes = c * tile});
                if (final_tail) {
                    noc.async_read(tail_states, seed_row, tile, {.page_id = page}, {.offset_bytes = c * tile});
                } else {
                    noc.async_read(seed_states, seed_row, tile, {.page_id = page}, {.offset_bytes = c * tile});
                }
            }
            noc.async_read_barrier();
            e.push_back(kt);
            b.push_back(kt);
            seed_row.push_back(kt);
        }
    }
}
