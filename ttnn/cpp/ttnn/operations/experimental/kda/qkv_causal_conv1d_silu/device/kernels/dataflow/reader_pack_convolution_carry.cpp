// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t history_rows>
TT_KERNEL void reader(uint32_t ct_start, uint32_t ct_count, uint32_t sequence, uint32_t wrap_row) {
    const auto input = TensorAccessor(tensor::input);
    const auto wrap_indicator = TensorAccessor(tensor::wrap_indicator);
    DataflowBuffer packed_rm(dfb::packed_rm);
    Noc noc;

    noc.async_read(
        wrap_indicator, CoreLocalMem<uint32_t>(packed_rm.get_write_ptr()), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    const bool is_wrap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(packed_rm.get_write_ptr())[0] != 0;

    constexpr uint32_t row_bytes = 32 * sizeof(uint16_t);
    const uint32_t tile_bytes = packed_rm.get_entry_size();
    const uint32_t outgoing_end = is_wrap ? wrap_row : sequence;
    for (uint32_t item = 0; item < ct_count; ++item) {
        const uint32_t ct = ct_start + item;
        packed_rm.reserve_back(1);
        noc.async_write_zeros(packed_rm, tile_bytes);
        noc.write_zeros_l1_barrier();
        for (uint32_t row = 0; row < history_rows; ++row) {
            noc.async_read(
                input,
                packed_rm,
                row_bytes,
                {.page_id = outgoing_end - history_rows + row, .offset_bytes = ct * row_bytes},
                {.offset_bytes = row * row_bytes});
        }
        if (is_wrap) {
            for (uint32_t row = 0; row < history_rows; ++row) {
                noc.async_read(
                    input,
                    packed_rm,
                    row_bytes,
                    {.page_id = sequence - history_rows + row, .offset_bytes = ct * row_bytes},
                    {.offset_bytes = (history_rows + row) * row_bytes});
            }
        }
        noc.async_read_barrier();
        packed_rm.push_back(1);
    }
}
