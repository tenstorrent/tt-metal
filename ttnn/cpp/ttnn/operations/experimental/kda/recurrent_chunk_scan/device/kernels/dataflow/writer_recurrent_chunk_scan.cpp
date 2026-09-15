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

template <uint32_t Rows, uint32_t Vt, uint32_t VtFull, typename Accessor>
FORCE_INLINE void write_value_slice(
    const Accessor& accessor, DataflowBuffer& buffer, Noc& noc, uint32_t row_base, uint32_t value_block) {
    constexpr uint32_t tile_count = Rows * Vt;
    buffer.wait_front(tile_count);
    const uint32_t entry_size = buffer.get_entry_size();
    for (uint32_t row = 0; row < Rows; ++row) {
        const uint32_t destination = row_base + row * VtFull + value_block * Vt;
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
    buffer.pop_front(tile_count);
}

template <uint32_t Kt, uint32_t Vt, uint32_t VtFull>
FORCE_INLINE void write_summary(uint32_t head, uint32_t value_block) {
    const auto output_accessor = TensorAccessor(tensor::output);
    const auto final_state_accessor = TensorAccessor(tensor::final_state);
    DataflowBuffer output(dfb::output);
    DataflowBuffer final_state(dfb::final_state);
    Noc noc;

    const uint32_t row_base = head * Kt * VtFull;
    write_value_slice<Kt, Vt, VtFull>(output_accessor, output, noc, row_base, value_block);
    write_value_slice<Kt, Vt, VtFull>(final_state_accessor, final_state, noc, row_base, value_block);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t VtFull>
FORCE_INLINE void write_recurrent(uint32_t head, uint32_t value_block, uint32_t num_chunks) {
    const auto output_accessor = TensorAccessor(tensor::output);
    const auto final_state_accessor = TensorAccessor(tensor::final_state);
    DataflowBuffer output(dfb::output);
    DataflowBuffer final_state(dfb::final_state);
    Noc noc;

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t row_base = (head * num_chunks + chunk) * Ct * VtFull;
        write_value_slice<Ct, Vt, VtFull>(output_accessor, output, noc, row_base, value_block);
    }
    const uint32_t state_row_base = head * Kt * VtFull;
    write_value_slice<Kt, Vt, VtFull>(final_state_accessor, final_state, noc, state_row_base, value_block);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t Vt_full, uint32_t summary_pair>
TT_KERNEL void writer(uint32_t head, uint32_t value_block, uint32_t num_chunks) {
    if constexpr (summary_pair) {
        write_summary<Kt, Vt, Vt_full>(head, value_block);
    } else {
        write_recurrent<Ct, Kt, Vt, Vt_full>(head, value_block, num_chunks);
    }
}
