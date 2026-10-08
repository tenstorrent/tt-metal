// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
//
// Phase B (scan) writer, value-parallel. This core produced ONE V-block of one head and writes that slice back into
// the full output tensors using their full-V row stride.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/dataflow/value_block_multicast.hpp"

template <uint32_t Rows, uint32_t Vt, uint32_t RowStride, uint32_t PacketRows, bool Consume = true, typename Accessor>
FORCE_INLINE void write_value_slice(
    const Accessor& accessor, DataflowBuffer& buffer, Noc& noc, uint32_t row_base, uint32_t column_base) {
    static_assert(PacketRows > 0 && Rows % PacketRows == 0);
    static_assert(Consume || PacketRows == Rows, "Retained writes require one complete buffer packet");
    constexpr uint32_t packet_tiles = PacketRows * Vt;
    const uint32_t entry_size = buffer.get_entry_size();
    for (uint32_t packet = 0; packet < Rows; packet += PacketRows) {
        buffer.wait_front(packet_tiles);
        for (uint32_t row = 0; row < PacketRows; ++row) {
            const uint32_t destination = row_base + (packet + row) * RowStride + column_base;
            for (uint32_t value_tile = 0; value_tile < Vt; ++value_tile) {
                noc.async_write(
                    buffer,
                    accessor,
                    entry_size,
                    {.offset_bytes = (row * Vt + value_tile) * entry_size},
                    {.page_id = destination + value_tile});
            }
        }
        noc.async_write_barrier();
        if constexpr (Consume) {
            buffer.pop_front(packet_tiles);
        }
    }
}

// Consume a published matrix this mode does not emit.
template <uint32_t Rows, uint32_t Vt>
FORCE_INLINE void drain(DataflowBuffer& buffer) {
    buffer.wait_front(Rows * Vt);
    buffer.pop_front(Rows * Vt);
}

// A rank with no head contributes the affine identity (I, 0): this value block writes its columns of I and of 0
// into the packed [A | B] rows. One BF16 scratch tile is zeroed, written to every off-diagonal tile, then given
// its diagonal and written to the diagonal tiles.
template <uint32_t Kt, uint32_t Vt, uint32_t VtFull, typename Accessor>
FORCE_INLINE void write_identity_summary(
    const Accessor& packed, DataflowBuffer& scratch, Noc& noc, uint32_t head, uint32_t value_block) {
    constexpr uint32_t row_stride = Kt + VtFull;
    constexpr uint16_t bf16_one_bits = 0x3F80;
    constexpr uint32_t face_rows = tt::constants::FACE_HEIGHT;
    constexpr uint32_t face_cols = tt::constants::FACE_WIDTH;
    constexpr uint32_t bottom_right_face_offset = (tt::constants::TILE_WIDTH / face_cols + 1) * tt::constants::FACE_HW;
    const uint32_t row_base = head * Kt * row_stride;
    const uint32_t first_column = value_block * Vt;

    scratch.reserve_back(1);
    const uint32_t tile_bytes = scratch.get_entry_size();
    noc.async_write_zeros(scratch, tile_bytes);
    noc.write_zeros_l1_barrier();
    for (uint32_t row = 0; row < Kt; ++row) {
        for (uint32_t column = first_column; column < first_column + Vt; ++column) {
            noc.async_write(scratch, packed, tile_bytes, {}, {.page_id = row_base + row * row_stride + Kt + column});
            if (row != column) {
                noc.async_write(scratch, packed, tile_bytes, {}, {.page_id = row_base + row * row_stride + column});
            }
        }
    }
    noc.async_write_barrier();
    auto* halfwords = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scratch.get_write_ptr());
    for (uint32_t r = 0; r < face_rows; ++r) {
        halfwords[r * face_cols + r] = bf16_one_bits;
        halfwords[bottom_right_face_offset + r * face_cols + r] = bf16_one_bits;
    }
    for (uint32_t row = first_column; row < first_column + Vt; ++row) {
        noc.async_write(scratch, packed, tile_bytes, {}, {.page_id = row_base + row * row_stride + row});
    }
    noc.async_write_barrier();
}

template <uint32_t Kt, uint32_t Vt, uint32_t VtFull, bool PackedHead>
FORCE_INLINE void write_summary(
    uint32_t head,
    uint32_t value_block,
    uint32_t group,
    uint32_t split_group,
    uint32_t split_in_group,
    bool local_split) {
    const auto head_a_accessor = TensorAccessor(tensor::output);
    const auto head_b_accessor = TensorAccessor(tensor::final_state);
    const auto tail_a_accessor = TensorAccessor(tensor::tail_output);
    const auto tail_b_accessor = TensorAccessor(tensor::tail_final_state);
    DataflowBuffer full_a(dfb::output);
    DataflowBuffer full_b(dfb::final_state);
    DataflowBuffer split_head_a(dfb::summary_head_output);
    DataflowBuffer split_head_b(dfb::summary_head_state);
    Noc noc;

    const bool straddles = local_split && split_in_group != 0 && group == split_group;
    const bool head_active = !local_split || group < split_group || straddles;
    const bool tail_active = local_split && group >= split_group;
    // Packed rows hold [A | B]: A at column 0 and B at column Kt of a (Kt + VtFull)-tile row.
    constexpr uint32_t row_stride = PackedHead ? Kt + VtFull : VtFull;
    const uint32_t row_base = head * Kt * row_stride;
    const uint32_t a_column = value_block * Vt;
    const uint32_t b_column = (PackedHead ? Kt : 0) + value_block * Vt;

    if (head_active) {
        auto& head_a = straddles ? split_head_a : full_a;
        auto& head_b = straddles ? split_head_b : full_b;
        // Split-head buffers hold one tile-row each. Drain the same packets
        // compute publishes instead of waiting for a full matrix to fit.
        const auto write_head = [&](const auto& b_accessor) {
            if (straddles) {
                write_value_slice<Kt, Vt, row_stride, 1>(head_a_accessor, head_a, noc, row_base, a_column);
                write_value_slice<Kt, Vt, row_stride, 1>(b_accessor, head_b, noc, row_base, b_column);
            } else {
                write_value_slice<Kt, Vt, row_stride, Kt>(head_a_accessor, head_a, noc, row_base, a_column);
                write_value_slice<Kt, Vt, row_stride, Kt>(b_accessor, head_b, noc, row_base, b_column);
            }
        };
        if constexpr (PackedHead) {
            write_head(head_a_accessor);
        } else {
            write_head(head_b_accessor);
        }
    } else if constexpr (PackedHead) {
        DataflowBuffer scratch(dfb::identity_scratch);
        write_identity_summary<Kt, Vt, VtFull>(head_a_accessor, scratch, noc, head, value_block);
    }

    if (tail_active) {
        if constexpr (PackedHead) {
            // The packed head serves single-group ranks, whose split tail no caller reads.
            drain<Kt, Vt>(full_a);
            drain<Kt, Vt>(full_b);
        } else {
            write_value_slice<Kt, Vt, VtFull, Kt>(tail_a_accessor, full_a, noc, row_base, a_column);
            write_value_slice<Kt, Vt, VtFull, Kt>(tail_b_accessor, full_b, noc, row_base, a_column);
        }
    }
}

// The peers' segment and receiver count for the writer's multicast of the state update's inputs.
struct LatePeers {
    uint32_t x0;
    uint32_t y0;
    uint32_t x1;
    uint32_t y1;
    uint32_t receivers;
};

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t VtFull, uint32_t late_on_writer>
FORCE_INLINE void write_recurrent(
    uint32_t head,
    uint32_t value_block,
    uint32_t num_chunks,
    uint32_t valid_chunks,
    uint32_t final_head,
    const LatePeers& peers) {
    const auto output_accessor = TensorAccessor(tensor::output);
    const auto final_state_accessor = TensorAccessor(tensor::final_state);
    DataflowBuffer output(dfb::output);
    DataflowBuffer final_state(dfb::final_state);
    Noc noc;

    const auto write_output = [&](uint32_t chunk) {
        const uint32_t row_base = (head * num_chunks + chunk) * Ct * VtFull;
        write_value_slice<Ct, Vt, VtFull, Ct>(output_accessor, output, noc, row_base, value_block * Vt);
    };
    if constexpr (late_on_writer) {
        // Value block 0 multicasts each chunk's k_dec_t and dl on this NoC, one chunk ahead of the output it
        // drains, while the reader multicasts the other inputs on its own.
        constexpr uint32_t key_chunk_tiles = Kt * Ct;
        DataflowBuffer k_decay_transposed(*dfb::get_token_if_present<"k_decay_transposed">());
        DataflowBuffer final_decay(*dfb::get_token_if_present<"final_decay">());
        const auto k_decay_transposed_accessor = TensorAccessor(tensor::k_decay_transposed);
        const auto final_decay_accessor = TensorAccessor(tensor::final_decay);
        Semaphore ready(sem::ready_late);
        Semaphore valid(sem::valid_late);
        const bool sender = value_block == 0;
        if (sender) {
            // set_multicast sources its payload from this local word; preset it to VALID once.
            valid.set(1);
        }
        const auto publish = [&](uint32_t chunk) {
            const uint32_t head_chunk = head * num_chunks + chunk;
            SharedInput inputs[2] = {
                {&k_decay_transposed,
                 key_chunk_tiles,
                 sender ? stage_contiguous_tiles(
                              k_decay_transposed_accessor,
                              k_decay_transposed,
                              noc,
                              head_chunk * key_chunk_tiles,
                              key_chunk_tiles)
                        : 0},
                {&final_decay,
                 Kt,
                 sender ? stage_contiguous_tiles(final_decay_accessor, final_decay, noc, head_chunk * Kt, Kt) : 0}};
            if (sender) {
                multicast_shared(noc, inputs, ready, valid, peers.x0, peers.y0, peers.x1, peers.y1, peers.receivers);
            } else {
                receive_shared(noc, inputs, ready, valid, peers.x0, peers.y0);
            }
        };
        publish(0);
        for (uint32_t chunk = 0; chunk < valid_chunks; ++chunk) {
            if (chunk + 1 < valid_chunks) {
                publish(chunk + 1);
            }
            write_output(chunk);
        }
        // Retire the multicast writes and ready increments before exit; dispatch re-initializes both semaphores
        // on every launch.
        if (sender) {
            noc.async_write_barrier();
        } else {
            noc.async_atomic_barrier();
        }
    } else {
        for (uint32_t chunk = 0; chunk < valid_chunks; ++chunk) {
            write_output(chunk);
        }
    }
    // Preserve every valid group's state at its own index. Also publish the last
    // valid state in the final physical slot used by the layer's carry selection.
    if (final_head != head) {
        write_value_slice<Kt, Vt, VtFull, Kt, false>(
            final_state_accessor, final_state, noc, final_head * Kt * VtFull, value_block * Vt);
    }
    write_value_slice<Kt, Vt, VtFull, Kt>(final_state_accessor, final_state, noc, head * Kt * VtFull, value_block * Vt);
}

template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t Vt_full,
    uint32_t summary,
    uint32_t packed_head,
    uint32_t late_on_writer,
    uint32_t has_actual_end>
TT_KERNEL void writer(
    uint32_t head,
    uint32_t value_block,
    uint32_t num_chunks,
    uint32_t group,
    uint32_t peer_x0,
    uint32_t peer_y0,
    uint32_t peer_x1,
    uint32_t peer_y1,
    uint32_t receivers) {
    kda_chronology::Topology topology{};
    uint32_t groups = 0;
    uint32_t valid_chunks = num_chunks;
    if constexpr (summary || has_actual_end) {
        DataflowBuffer chronology(*dfb::get_token_if_present<"chronology_writer">());
        chronology.wait_front(1);
        topology = kda_chronology::load(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chronology.get_read_ptr()));
        chronology.pop_front(1);
        groups = topology.local_rows / tt::constants::TILE_HEIGHT / num_chunks;
        valid_chunks = topology.valid_chunks(group, groups);
        if (valid_chunks == 0) {
            if constexpr (packed_head) {
                const auto packed = TensorAccessor(tensor::output);
                DataflowBuffer scratch(dfb::identity_scratch);
                Noc noc;
                write_identity_summary<Kt, Vt, Vt_full>(packed, scratch, noc, head, value_block);
            }
            return;
        }
    }
    if constexpr (summary) {
        write_summary<Kt, Vt, Vt_full, packed_head != 0>(
            head,
            value_block,
            group,
            topology.split_group(groups),
            topology.split_in_group(groups),
            topology.has_valid_tail());
    } else {
        uint32_t final_head = head;
        if constexpr (has_actual_end) {
            if (group + 1 == topology.active_groups(groups)) {
                final_head = head - group + groups - 1;
            }
        }
        write_recurrent<Ct, Kt, Vt, Vt_full, late_on_writer>(
            head, value_block, num_chunks, valid_chunks, final_head, {peer_x0, peer_y0, peer_x1, peer_y1, receivers});
    }
}
