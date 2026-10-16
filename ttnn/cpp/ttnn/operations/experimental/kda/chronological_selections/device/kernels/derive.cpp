// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
using namespace kda_chronology;
template <
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows,
    uint32_t BH,
    uint32_t K,
    uint32_t V,
    uint32_t has_actual_end>
TT_KERNEL void derive() {
    Noc noc;
    Scratchpad<volatile uint32_t> scratch(scratch::scratch);
    const auto actual_start = TensorAccessor(tensor::actual_start);
    const auto output = TensorAccessor(tensor::output);
    const auto address = scratch.get_base_address();
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    noc.async_read(actual_start, CoreLocalMem<uint32_t>(address), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t start = words[0];
    auto topology = derive(start, sp_rank, sp_size, local_rows);
    if constexpr (has_actual_end) {
        const auto actual_end = TensorAccessor(*tensor::get_token_if_present<"actual_end">());
        noc.async_read(actual_end, CoreLocalMem<uint32_t>(address), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        topology = derive_interval(start, words[0], sp_rank, sp_size, local_rows);
    }
    for (uint32_t row = 0; row < selection::record_count; ++row) {
        for (uint32_t i = 0; i < selection::record_width; ++i) {
            words[i] = 0;
        }
        if (row == selection::local_final_history) {
            // The end segment is the separated tail when it holds valid rows, else the
            // rank's prefix. With fewer than three valid rows, the history reaches into
            // the tokens preceding that segment: the layer carry before the logical
            // start, or the exchanged predecessor history otherwise.
            const bool tail = topology.has_valid_tail();
            const uint32_t segment_begin = tail ? topology.head_rows : 0;
            const uint32_t segment_rows = topology.valid_rows - segment_begin;
            const uint32_t carried = segment_rows < selection::history_rows ? segment_rows : selection::history_rows;
            // Words [0, history_rows): local rows starting at the earliest valid row kept.
            // With fewer than three valid rows, rows past the end are never chosen below.
            const uint32_t local_begin = topology.valid_rows - carried;
            for (uint32_t i = 0; i < selection::history_rows; ++i) {
                words[i] = local_begin + i;
            }
            // Words [history_rows, 2 * history_rows): rows of the candidate table
            // [layer history; predecessor history; local rows].
            const uint32_t preceding = topology.rank == topology.first_rank && !tail ? 0 : selection::history_rows;
            for (uint32_t i = 0; i < selection::history_rows; ++i) {
                const uint32_t position = carried + i;
                words[selection::history_rows + i] =
                    position < selection::history_rows
                        ? preceding + position
                        : 2 * selection::history_rows + position - selection::history_rows;
            }
        } else if (row < selection::final_state) {
            uint32_t base;
            if (row == selection::outgoing_history) {
                base = (topology.local_split ? topology.head_rows : local_rows) - selection::history_rows;
            } else if (row == selection::predecessor_history) {
                base = ((topology.rank + sp_size - 1) % sp_size) * selection::history_rows;
            } else {
                base = topology.final_owner * selection::history_rows;
            }
            for (uint32_t i = 0; i < selection::history_rows; ++i) {
                words[i] = base + i;
            }
        } else {
            // Candidates contain one final state per rank, followed by the
            // completed distributed prefix at index sp_size for unsplit execution.
            const uint32_t selected = topology.split ? topology.final_owner : sp_size;
            const bool is_end_record = row != selection::final_state;
            words[0] = selected + uint32_t(is_end_record);
            if (is_end_record) {
                words[1] = BH;
                words[2] = K;
                words[3] = V;
            }
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(address), output, selection::record_width * sizeof(uint32_t), {}, {.page_id = row});
        noc.async_write_barrier();
    }
}
