// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "chronology.hpp"

// Fuse the local-row gather and candidate selection. In particular, a fresh
// one/two-token prompt must not retain any bytes from the previous request.
template <uint32_t row_bytes>
TT_KERNEL void select_request_history() {
    Noc noc;
    Scratchpad<volatile uint32_t> scratch(scratch::scratch);
    const auto qkv = TensorAccessor(tensor::projected_qkv);
    const auto layer = TensorAccessor(tensor::layer_history);
    const auto predecessor = TensorAccessor(tensor::predecessor_history);
    const auto selections = TensorAccessor(tensor::selection_records);
    const auto start = TensorAccessor(tensor::actual_start);
    const auto output = TensorAccessor(tensor::output);
    const auto address = scratch.get_base_address();
    auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(address);
    // Keep DRAM reads aligned at the scratch base, including the four-byte scalar.
    noc.async_read(start, CoreLocalMem<uint32_t>(address), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    const bool fresh = words[0] == 0;
    noc.async_read(
        selections,
        CoreLocalMem<uint32_t>(address),
        32,
        {.page_id = kda_chronology::selection::local_final_history},
        {});
    noc.async_read_barrier();
    const auto data = address + 64;
    constexpr uint32_t history_rows = kda_chronology::selection::history_rows;
    for (uint32_t row = 0; row < history_rows; ++row) {
        const uint32_t selected = words[history_rows + row];
        if (selected < history_rows && fresh) {
            auto* zeros = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(data);
            for (uint32_t i = 0; i < row_bytes / sizeof(uint32_t); ++i) {
                zeros[i] = 0;
            }
        } else if (selected < history_rows) {
            noc.async_read(layer, CoreLocalMem<uint32_t>(data), row_bytes, {.page_id = selected}, {});
            noc.async_read_barrier();
        } else if (selected < 2 * history_rows) {
            noc.async_read(
                predecessor, CoreLocalMem<uint32_t>(data), row_bytes, {.page_id = selected - history_rows}, {});
            noc.async_read_barrier();
        } else {
            const uint32_t local_row = words[selected - 2 * history_rows];
            noc.async_read(qkv, CoreLocalMem<uint32_t>(data), row_bytes, {.page_id = local_row}, {});
            noc.async_read_barrier();
        }
        noc.async_write(CoreLocalMem<uint32_t>(data), output, row_bytes, {}, {.page_id = row});
        noc.async_write_barrier();
    }
}
