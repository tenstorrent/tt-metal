// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
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
    auto address = scratch.get_base_address();
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    noc.async_read(actual_start, CoreLocalMem<uint32_t>(address), 4, {.page_id = 0}, {});
    noc.async_read_barrier();
    const uint32_t start = words[0];
    auto t = derive(start, sp_rank, sp_size, local_rows);
    if constexpr (has_actual_end) {
        const auto end = TensorAccessor(tensor::actual_end);
        noc.async_read(end, CoreLocalMem<uint32_t>(address), 4, {.page_id = 0}, {});
        noc.async_read_barrier();
        t = derive_interval(start, words[0], sp_rank, sp_size, local_rows);
    }
    for (uint32_t row = 0; row < selection::record_count(sp_size); ++row) {
        for (uint32_t i = 0; i < selection::record_width; ++i) {
            words[i] = 0;
        }
        if (row == selection::local_final_history(sp_size)) {
            const uint32_t end = t.valid_rows == 0 ? selection::history_rows : t.valid_rows;
            for (uint32_t i = 0; i < selection::history_rows; ++i) {
                words[i] = end - selection::history_rows + i;
            }
        } else if (row < selection::local_entry_state) {
            const uint32_t base =
                row == selection::outgoing_history
                    ? (t.local_split ? t.head_rows : local_rows) - selection::history_rows
                    : (row == selection::predecessor_history ? (t.rank + sp_size - 1) % sp_size : t.final_owner) *
                          selection::history_rows;
            for (uint32_t i = 0; i < selection::history_rows; ++i) {
                words[i] = base + i;
            }
        } else {
            uint32_t selected;
            if (row < selection::final_state) {
                selected = (t.rank + sp_size - t.first_rank) % sp_size;
            } else if (row < selection::affine_transforms) {
                selected = t.split ? t.final_owner : sp_size;
            } else {
                selected = (t.first_rank + (row - selection::affine_transforms) / 2) % sp_size;
            }
            const bool end = (row - selection::local_entry_state) % 2 != 0;
            words[0] = selected + uint32_t(end);
            if (end) {
                words[1] = BH;
                words[2] = K;
                words[3] = row >= selection::affine_transforms ? K + V : V;
            }
        }
        noc.async_write(
            CoreLocalMem<uint32_t>(address), output, selection::record_width * sizeof(uint32_t), {}, {.page_id = row});
        noc.async_write_barrier();
    }
}
