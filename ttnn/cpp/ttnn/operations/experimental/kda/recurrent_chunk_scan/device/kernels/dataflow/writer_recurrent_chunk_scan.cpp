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

template <uint32_t Rows, uint32_t Vt, uint32_t VtFull, uint32_t PacketRows, typename Accessor>
FORCE_INLINE void write_value_slice(
    const Accessor& accessor, DataflowBuffer& buffer, Noc& noc, uint32_t row_base, uint32_t value_block) {
    static_assert(PacketRows > 0 && Rows % PacketRows == 0);
    constexpr uint32_t packet_tiles = PacketRows * Vt;
    const uint32_t entry_size = buffer.get_entry_size();
    for (uint32_t packet = 0; packet < Rows; packet += PacketRows) {
        buffer.wait_front(packet_tiles);
        for (uint32_t row = 0; row < PacketRows; ++row) {
            const uint32_t destination = row_base + (packet + row) * VtFull + value_block * Vt;
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
        buffer.pop_front(packet_tiles);
    }
}

template <uint32_t Kt, uint32_t Vt, uint32_t VtFull>
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
    const uint32_t row_base = head * Kt * VtFull;

    if (head_active) {
        auto& head_a = straddles ? split_head_a : full_a;
        auto& head_b = straddles ? split_head_b : full_b;
        if (straddles) {
            write_value_slice<Kt, Vt, VtFull, 1>(head_a_accessor, head_a, noc, row_base, value_block);
            write_value_slice<Kt, Vt, VtFull, 1>(head_b_accessor, head_b, noc, row_base, value_block);
        } else {
            write_value_slice<Kt, Vt, VtFull, Kt>(head_a_accessor, head_a, noc, row_base, value_block);
            write_value_slice<Kt, Vt, VtFull, Kt>(head_b_accessor, head_b, noc, row_base, value_block);
        }
    }

    if (tail_active) {
        write_value_slice<Kt, Vt, VtFull, Kt>(tail_a_accessor, full_a, noc, row_base, value_block);
        write_value_slice<Kt, Vt, VtFull, Kt>(tail_b_accessor, full_b, noc, row_base, value_block);
    }
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t VtFull>
FORCE_INLINE void write_recurrent(
    uint32_t head, uint32_t value_block, uint32_t num_chunks, uint32_t valid_chunks, uint32_t final_head) {
    const auto output_accessor = TensorAccessor(tensor::output);
    const auto final_state_accessor = TensorAccessor(tensor::final_state);
    DataflowBuffer output(dfb::output);
    DataflowBuffer final_state(dfb::final_state);
    Noc noc;

    for (uint32_t chunk = 0; chunk < valid_chunks; ++chunk) {
        const uint32_t row_base = (head * num_chunks + chunk) * Ct * VtFull;
        write_value_slice<Ct, Vt, VtFull, Ct>(output_accessor, output, noc, row_base, value_block);
    }
    const uint32_t state_row_base = final_head * Kt * VtFull;
    write_value_slice<Kt, Vt, VtFull, Kt>(final_state_accessor, final_state, noc, state_row_base, value_block);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t Vt_full, uint32_t summary, uint32_t has_actual_end>
TT_KERNEL void writer(uint32_t head, uint32_t value_block, uint32_t num_chunks, uint32_t group) {
    uint32_t split_group = 0;
    uint32_t split_in_group = 0;
    bool local_split = false;
    uint32_t valid_chunks = num_chunks;
    uint32_t final_head = head;
    if constexpr (summary || has_actual_end) {
        DataflowBuffer chronology(*dfb::get_token_if_present<"chronology_writer">());
        chronology.wait_front(1);
        auto topology = kda_chronology::load(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chronology.get_read_ptr()));
        chronology.pop_front(1);
        uint32_t groups = topology.local_rows / 32 / num_chunks;
        split_group = topology.split_group(groups);
        split_in_group = topology.split_in_group(groups);
        local_split = topology.has_valid_tail();
        valid_chunks = topology.valid_chunks(group, groups);
        if (valid_chunks == 0) {
            return;
        }
        if (group + 1 == topology.active_groups(groups)) {
            final_head = head - group + groups - 1;
        }
    }
    if constexpr (summary) {
        write_summary<Kt, Vt, Vt_full>(head, value_block, group, split_group, split_in_group, local_split);
    } else {
        write_recurrent<Ct, Kt, Vt, Vt_full>(head, value_block, num_chunks, valid_chunks, final_head);
    }
}
