// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Phase B (scan) writer, value-parallel. This core produced ONE V-block of one head and writes that slice back into
// the full output tensors using their full-V row stride.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t Vt_full, uint32_t summary_pair>
TT_KERNEL void writer(uint32_t head, uint32_t value_block, uint32_t num_chunks) {
    const auto output_accessor = TensorAccessor(tensor::output);
    const auto final_state_accessor = TensorAccessor(tensor::final_state);
    DataflowBuffer output(dfb::output);
    DataflowBuffer final_state(dfb::final_state);
    Noc noc;

    constexpr uint32_t cv = Ct * Vt;
    constexpr uint32_t kv = Kt * Vt;
    const uint32_t output_entry_size = output.get_entry_size();
    const uint32_t final_state_entry_size = final_state.get_entry_size();

    if constexpr (summary_pair) {
        output.wait_front(kv);
        const uint32_t row_base = head * Kt * Vt_full;
        for (uint32_t row = 0; row < Kt; row++) {
            const uint32_t destination = row_base + row * Vt_full + value_block * Vt;
            for (uint32_t value_tile = 0; value_tile < Vt; value_tile++) {
                noc.async_write(
                    output,
                    output_accessor,
                    output_entry_size,
                    {.offset_bytes = (row * Vt + value_tile) * output_entry_size},
                    {.page_id = destination + value_tile});
            }
        }
        noc.async_write_barrier();
        output.pop_front(kv);

        final_state.wait_front(kv);
        for (uint32_t row = 0; row < Kt; row++) {
            const uint32_t destination = row_base + row * Vt_full + value_block * Vt;
            for (uint32_t value_tile = 0; value_tile < Vt; value_tile++) {
                noc.async_write(
                    final_state,
                    final_state_accessor,
                    final_state_entry_size,
                    {.offset_bytes = (row * Vt + value_tile) * final_state_entry_size},
                    {.page_id = destination + value_tile});
            }
        }
        noc.async_write_barrier();
        final_state.pop_front(kv);
    } else {
        for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
            output.wait_front(cv);
            const uint32_t row_base = (head * num_chunks + chunk) * Ct * Vt_full;
            for (uint32_t row = 0; row < Ct; row++) {
                const uint32_t destination = row_base + row * Vt_full + value_block * Vt;
                for (uint32_t value_tile = 0; value_tile < Vt; value_tile++) {
                    noc.async_write(
                        output,
                        output_accessor,
                        output_entry_size,
                        {.offset_bytes = (row * Vt + value_tile) * output_entry_size},
                        {.page_id = destination + value_tile});
                }
            }
            noc.async_write_barrier();
            output.pop_front(cv);
        }

        final_state.wait_front(kv);
        const uint32_t row_base = head * Kt * Vt_full;
        for (uint32_t row = 0; row < Kt; row++) {
            const uint32_t destination = row_base + row * Vt_full + value_block * Vt;
            for (uint32_t value_tile = 0; value_tile < Vt; value_tile++) {
                noc.async_write(
                    final_state,
                    final_state_accessor,
                    final_state_entry_size,
                    {.offset_bytes = (row * Vt + value_tile) * final_state_entry_size},
                    {.page_id = destination + value_tile});
            }
        }
        noc.async_write_barrier();
        final_state.pop_front(kv);
    }
}
