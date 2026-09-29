// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
//
// KDA scan reader: the initial state S [K,V] once, then vector-decay prep
// intermediates v_beta, kd, q_decay, intra, k_dec_t, dl[K,1], t_inv. FP32 by default; selected intermediates may be
// BF16.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <typename Accessor>
FORCE_INLINE void read_and_publish_contiguous_tiles(
    const Accessor& accessor, DataflowBuffer& buffer, Noc& noc, uint32_t base, uint32_t count) {
    buffer.reserve_back(count);
    const uint32_t entry_size = buffer.get_entry_size();
    uint32_t tile = 0;
    for (const auto& page : accessor.pages(base, base + count)) {
        noc.async_read(accessor, buffer, entry_size, {.page_id = page.page_id()}, {.offset_bytes = tile * entry_size});
        ++tile;
    }
    noc.async_read_barrier();
    // push_back publishes this DFB to compute. Complete its reads first; delaying publication to coalesce across
    // buffers would prevent compute from overlapping the next buffer's NOC reads.
    buffer.push_back(count);
}

template <uint32_t Vt, uint32_t VtFull, typename Accessor>
FORCE_INLINE void read_and_publish_value_slice(
    const Accessor& accessor,
    DataflowBuffer& buffer,
    Noc& noc,
    uint32_t row_base,
    uint32_t rows,
    uint32_t value_block) {
    buffer.reserve_back(rows * Vt);
    const uint32_t entry_size = buffer.get_entry_size();
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t source = row_base + row * VtFull + value_block * Vt;
        const uint32_t destination = row * Vt;
        for (uint32_t value_tile = 0; value_tile < Vt; ++value_tile) {
            noc.async_read(
                accessor,
                buffer,
                entry_size,
                {.page_id = source + value_tile},
                {.offset_bytes = (destination + value_tile) * entry_size});
        }
    }
    noc.async_read_barrier();
    // push_back publishes this DFB to compute. Complete its reads first; delaying publication to coalesce across
    // buffers would prevent compute from overlapping the next buffer's NOC reads.
    buffer.push_back(rows * Vt);
}

template <uint32_t Tiles>
FORCE_INLINE void seed_zero(DataflowBuffer& state, Noc& noc) {
    state.reserve_back(Tiles);
    noc.async_write_zeros(state, Tiles * state.get_entry_size());
    noc.write_zeros_l1_barrier();
    state.push_back(Tiles);
}

template <uint32_t Kt, uint32_t Vt>
FORCE_INLINE void seed_identity(DataflowBuffer& buffer, Noc& noc, uint32_t value_block) {
    constexpr uint32_t one_fp32 = __builtin_bit_cast(uint32_t, 1.0F);
    constexpr uint32_t face_elements = tt::constants::FACE_HW;
    constexpr uint32_t tile_elements = tt::constants::TILE_HW;
    constexpr uint32_t tile_count = Kt * Vt;

    buffer.reserve_back(tile_count);
    noc.async_write_zeros(buffer, tile_count * buffer.get_entry_size());
    noc.write_zeros_l1_barrier();
    {
        auto lock = buffer.scoped_write_lock(tile_count);
        auto state_ptr = lock.get_ptr<volatile uint32_t>();
        for (uint32_t local_col = 0; local_col < Vt; ++local_col) {
            const uint32_t global_col = value_block * Vt + local_col;
            if (global_col < Kt) {
                auto tile = state_ptr + (global_col * Vt + local_col) * tile_elements;
                for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; ++row) {
                    tile[row * tt::constants::FACE_WIDTH + row] = one_fp32;
                    tile[3 * face_elements + row * tt::constants::FACE_WIDTH + row] = one_fp32;
                }
            }
        }
    }
    buffer.push_back(tile_count);
}

// Chunk inputs that do not depend on the value columns are identical for every value block of a head.
// With mcast_shared, value block 0 reads them from DRAM once and multicasts them into its siblings' buffers
// (identical buffer addresses on every core of the program), so DRAM serves each shared tile once per head.
// Handshake: receivers reserve their slots, reset valid and raise ready; the sender stages its reads, waits
// for every receiver, multicasts the data then the valid flag, and flushes before publishing its own copy.
struct SharedInput {
    DataflowBuffer* buffer;
    uint32_t tiles;
    uint32_t slot;
};

template <typename Accessor>
FORCE_INLINE uint32_t
stage_contiguous_tiles(const Accessor& accessor, DataflowBuffer& buffer, Noc& noc, uint32_t base, uint32_t count) {
    buffer.reserve_back(count);
    const uint32_t entry_size = buffer.get_entry_size();
    uint32_t tile = 0;
    for (const auto& page : accessor.pages(base, base + count)) {
        noc.async_read(accessor, buffer, entry_size, {.page_id = page.page_id()}, {.offset_bytes = tile * entry_size});
        ++tile;
    }
    return buffer.get_write_ptr();
}

template <uint32_t N, typename ReadySem, typename ValidSem>
FORCE_INLINE void multicast_shared(
    Noc& noc,
    SharedInput (&inputs)[N],
    ReadySem& ready,
    ValidSem& valid,
    uint32_t x0,
    uint32_t y0,
    uint32_t x1,
    uint32_t y1,
    uint32_t receivers) {
    noc.async_read_barrier();
    ready.wait(receivers);
    ready.set(0);
    MulticastEndpoint destination;
    for (auto& input : inputs) {
        noc.async_write_multicast(
            CoreLocalMem<uint32_t>(input.slot),
            destination,
            input.tiles * input.buffer->get_entry_size(),
            receivers,
            {},
            {.noc_x_start = x0, .noc_y_start = y0, .noc_x_end = x1, .noc_y_end = y1, .addr = input.slot},
            /*linked=*/true);
    }
    // The flush orders the flag after the data on Blackhole and proves the source slots were read before
    // compute may pop them.
    noc.async_writes_flushed();
    valid.set_multicast(noc, x0, y0, x1, y1, receivers);
    for (auto& input : inputs) {
        input.buffer->push_back(input.tiles);
    }
}

template <uint32_t N, typename ReadySem, typename ValidSem>
FORCE_INLINE void receive_shared(
    Noc& noc, SharedInput (&inputs)[N], ReadySem& ready, ValidSem& valid, uint32_t sender_x, uint32_t sender_y) {
    for (auto& input : inputs) {
        input.buffer->reserve_back(input.tiles);
    }
    // Reset valid before raising ready: a fast sender may multicast VALID right after the increment.
    valid.set(0);
    ready.up(noc, sender_x, sender_y, 1);
    valid.wait(1);
    for (auto& input : inputs) {
        input.buffer->push_back(input.tiles);
    }
}

template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Vt_full,
    uint32_t mcast_shared,
    uint32_t summary,
    uint32_t groups_per_head,
    uint32_t has_actual_end,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void reader(
    uint32_t head,
    uint32_t value_block,
    uint32_t num_chunks,
    uint32_t peer_x0,
    uint32_t peer_y0,
    uint32_t peer_x1,
    uint32_t peer_y1,
    uint32_t receivers) {
    const auto v_beta_accessor = TensorAccessor(tensor::v_beta);
    const auto kd_accessor = TensorAccessor(tensor::kd);
    const auto k_decay_transposed_accessor = TensorAccessor(tensor::k_decay_transposed);
    const auto final_decay_accessor = TensorAccessor(tensor::final_decay);
    const auto t_inv_accessor = TensorAccessor(tensor::t_inv);

    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer summary_seed(dfb::summary_seed);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer tail_entry_states(dfb::tail_entry_states);
    Noc noc;
    Semaphore ready(sem::ready);
    Semaphore valid(sem::valid);
    if constexpr (mcast_shared) {
        if (value_block == 0) {
            // set_multicast sources its payload from this local word; preset it to VALID once.
            valid.set(1);
        }
    }

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
    const uint32_t reset_chunk = topology.reset_chunk(head % groups_per_head, groups_per_head);
    if constexpr (summary || has_actual_end) {
        DataflowBuffer writer_chronology(*dfb::get_token_if_present<"chronology_writer">());
        writer_chronology.reserve_back(1);
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_chronology.get_write_ptr());
        kda_chronology::store(words, topology);
        writer_chronology.push_back(1);
    }
    const uint32_t valid_chunks = topology.valid_chunks(head % groups_per_head, groups_per_head);
    if (valid_chunks == 0) {
        valid.set(0);  // restore the initial value for the next launch
        return;
    }
    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;
    constexpr uint32_t key_value_tiles = Kt * Vt;

    if constexpr (summary) {
        seed_zero<key_value_tiles>(state, noc);
        seed_identity<Kt, Vt>(summary_seed, noc, value_block);

    } else {
        const auto group_entry_states_accessor = TensorAccessor(tensor::group_entry_states);
        read_and_publish_value_slice<Vt, Vt_full>(
            group_entry_states_accessor, state, noc, head * Kt * Vt_full, Kt, value_block);
    }

    for (uint32_t chunk = 0; chunk < valid_chunks; ++chunk) {
        const uint32_t head_chunk = head * num_chunks + chunk;
        // Publish the restart seed just in time, never before the loop. The state
        // DFB holds one kv payload and compute frees it only via pop_front at the
        // end of chunk 0, so hoisting this deadlocks; pushing it here reuses the
        // same capacity as a queue and costs no extra L1. reset_chunk is >= 1
        // whenever it is non-zero, so chunk 0 has always been consumed by now.
        if constexpr (summary) {
            if (reset_chunk != 0 && chunk == reset_chunk) {
                seed_zero<key_value_tiles>(state, noc);
                seed_identity<Kt, Vt>(summary_seed, noc, value_block);
            }
        } else {
            if (reset_chunk != 0 && chunk == reset_chunk) {
                const auto tail_entry_states_accessor = TensorAccessor(tensor::tail_entry_states);
                read_and_publish_value_slice<Vt, Vt_full>(
                    tail_entry_states_accessor,
                    tail_entry_states,
                    noc,
                    (head / groups_per_head) * Kt * Vt_full,
                    Kt,
                    value_block);
            }
        }
        read_and_publish_value_slice<Vt, Vt_full>(
            v_beta_accessor, v_beta, noc, head_chunk * Ct * Vt_full, Ct, value_block);
        if constexpr (!mcast_shared) {
            if constexpr (summary) {
                read_and_publish_contiguous_tiles(kd_accessor, kd, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
                read_and_publish_contiguous_tiles(
                    t_inv_accessor, t_inv, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
                read_and_publish_contiguous_tiles(
                    k_decay_transposed_accessor,
                    k_decay_transposed,
                    noc,
                    head_chunk * key_chunk_tiles,
                    key_chunk_tiles);
                read_and_publish_contiguous_tiles(final_decay_accessor, final_decay, noc, head_chunk * Kt, Kt);
            } else {
                read_and_publish_contiguous_tiles(kd_accessor, kd, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
                const auto q_decay_accessor = TensorAccessor(tensor::q_decay);
                const auto intra_accessor = TensorAccessor(tensor::intra);
                read_and_publish_contiguous_tiles(
                    q_decay_accessor, q_decay, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
                read_and_publish_contiguous_tiles(
                    intra_accessor, intra, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
                read_and_publish_contiguous_tiles(
                    k_decay_transposed_accessor,
                    k_decay_transposed,
                    noc,
                    head_chunk * key_chunk_tiles,
                    key_chunk_tiles);
                read_and_publish_contiguous_tiles(final_decay_accessor, final_decay, noc, head_chunk * Kt, Kt);
                read_and_publish_contiguous_tiles(
                    t_inv_accessor, t_inv, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
            }
        } else if constexpr (summary) {
            SharedInput inputs[] = {
                {&kd, chunk_key_tiles, 0},
                {&t_inv, chunk_chunk_tiles, 0},
                {&k_decay_transposed, key_chunk_tiles, 0},
                {&final_decay, Kt, 0},
            };
            if (value_block == 0) {
                inputs[0].slot =
                    stage_contiguous_tiles(kd_accessor, kd, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
                inputs[1].slot = stage_contiguous_tiles(
                    t_inv_accessor, t_inv, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
                inputs[2].slot = stage_contiguous_tiles(
                    k_decay_transposed_accessor,
                    k_decay_transposed,
                    noc,
                    head_chunk * key_chunk_tiles,
                    key_chunk_tiles);
                inputs[3].slot = stage_contiguous_tiles(final_decay_accessor, final_decay, noc, head_chunk * Kt, Kt);
                multicast_shared(noc, inputs, ready, valid, peer_x0, peer_y0, peer_x1, peer_y1, receivers);
            } else {
                receive_shared(noc, inputs, ready, valid, peer_x0, peer_y0);
            }
        } else {
            const auto q_decay_accessor = TensorAccessor(tensor::q_decay);
            const auto intra_accessor = TensorAccessor(tensor::intra);
            SharedInput inputs[] = {
                {&kd, chunk_key_tiles, 0},
                {&q_decay, chunk_key_tiles, 0},
                {&intra, chunk_chunk_tiles, 0},
                {&k_decay_transposed, key_chunk_tiles, 0},
                {&final_decay, Kt, 0},
                {&t_inv, chunk_chunk_tiles, 0},
            };
            if (value_block == 0) {
                inputs[0].slot =
                    stage_contiguous_tiles(kd_accessor, kd, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
                inputs[1].slot = stage_contiguous_tiles(
                    q_decay_accessor, q_decay, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
                inputs[2].slot = stage_contiguous_tiles(
                    intra_accessor, intra, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
                inputs[3].slot = stage_contiguous_tiles(
                    k_decay_transposed_accessor,
                    k_decay_transposed,
                    noc,
                    head_chunk * key_chunk_tiles,
                    key_chunk_tiles);
                inputs[4].slot = stage_contiguous_tiles(final_decay_accessor, final_decay, noc, head_chunk * Kt, Kt);
                inputs[5].slot = stage_contiguous_tiles(
                    t_inv_accessor, t_inv, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
                multicast_shared(noc, inputs, ready, valid, peer_x0, peer_y0, peer_x1, peer_y1, receivers);
            } else {
                receive_shared(noc, inputs, ready, valid, peer_x0, peer_y0);
            }
        }
    }
    if constexpr (mcast_shared) {
        if (value_block == 0) {
            // set_multicast reads the local valid word asynchronously; drain before restoring it.
            noc.async_write_barrier();
        } else {
            noc.async_atomic_barrier();
        }
        valid.set(0);
    }
}
