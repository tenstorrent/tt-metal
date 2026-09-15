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

template <uint32_t Rows, uint32_t Vt>
FORCE_INLINE void discard_value_slice(DataflowBuffer& buffer) {
    constexpr uint32_t tile_count = Rows * Vt;
    buffer.wait_front(tile_count);
    buffer.pop_front(tile_count);
}

template <uint32_t Rows, uint32_t Vt, uint32_t VtFull, typename Accessor>
FORCE_INLINE void write_streamed_value_slice(
    const Accessor& accessor, DataflowBuffer& buffer, Noc& noc, uint32_t row_base, uint32_t value_block) {
    const uint32_t entry_size = buffer.get_entry_size();
    for (uint32_t row = 0; row < Rows; ++row) {
        buffer.wait_front(Vt);
        const uint32_t destination = row_base + row * VtFull + value_block * Vt;
        for (uint32_t value_tile = 0; value_tile < Vt; ++value_tile) {
            noc.async_write(
                buffer,
                accessor,
                entry_size,
                {.offset_bytes = value_tile * entry_size},
                {.page_id = destination + value_tile});
        }
        noc.async_write_barrier();
        buffer.pop_front(Vt);
    }
}

template <uint32_t Rows, uint32_t Vt>
FORCE_INLINE void discard_streamed_value_slice(DataflowBuffer& buffer) {
    for (uint32_t row = 0; row < Rows; ++row) {
        buffer.wait_front(Vt);
        buffer.pop_front(Vt);
    }
}

template <uint32_t Kt, uint32_t Vt, uint32_t VtFull, typename AAccessor, typename BAccessor>
FORCE_INLINE void write_identity_pair(
    const AAccessor& a_accessor,
    const BAccessor& b_accessor,
    DataflowBuffer& identity_tile,
    DataflowBuffer& zero_tile,
    Noc& noc,
    uint32_t row_base,
    uint32_t value_block) {
    identity_tile.wait_front(1);
    zero_tile.wait_front(1);
    const uint32_t tile_bytes = zero_tile.get_entry_size();
    for (uint32_t row = 0; row < Kt; ++row) {
        for (uint32_t local_col = 0; local_col < Vt; ++local_col) {
            const uint32_t global_col = value_block * Vt + local_col;
            const auto& source = row == global_col ? identity_tile : zero_tile;
            const uint32_t destination = row_base + row * VtFull + global_col;
            noc.async_write(source, a_accessor, tile_bytes, {}, {.page_id = destination});
            noc.async_write(zero_tile, b_accessor, tile_bytes, {}, {.page_id = destination});
        }
    }
    noc.async_write_barrier();
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

template <uint32_t Kt, uint32_t Vt, uint32_t VtFull>
FORCE_INLINE void write_segmented_summary(
    uint32_t head, uint32_t value_block, uint32_t group, uint32_t wrap_group, uint32_t split_in_group) {
    const auto head_a_accessor = TensorAccessor(tensor::output);
    const auto head_b_accessor = TensorAccessor(tensor::final_state);
    const auto tail_a_accessor = TensorAccessor(tensor::tail_output);
    const auto tail_b_accessor = TensorAccessor(tensor::tail_final_state);
    DataflowBuffer full_a(dfb::output);
    DataflowBuffer full_b(dfb::final_state);
    DataflowBuffer split_head_a(dfb::summary_head_output);
    DataflowBuffer split_head_b(dfb::summary_head_state);
    DataflowBuffer identity_tile(dfb::summary_identity_tile);
    DataflowBuffer zero_tile(dfb::summary_zero_tile);
    DataflowBuffer wrap_control(dfb::wrap_control);
    Noc noc;

    wrap_control.wait_front(1);
    const bool device_wrap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wrap_control.get_read_ptr())[0] != 0;
    wrap_control.pop_front(1);
    const bool straddles = device_wrap && split_in_group != 0 && group == wrap_group;
    const bool head_active = !device_wrap || group < wrap_group || straddles;
    const bool tail_active = device_wrap && group >= wrap_group;
    const uint32_t row_base = head * Kt * VtFull;

    if (!device_wrap && split_in_group != 0 && group == wrap_group) {
        discard_streamed_value_slice<Kt, Vt>(split_head_a);
        discard_streamed_value_slice<Kt, Vt>(split_head_b);
    }

    if (head_active) {
        auto& head_a = straddles ? split_head_a : full_a;
        auto& head_b = straddles ? split_head_b : full_b;
        if (straddles) {
            write_streamed_value_slice<Kt, Vt, VtFull>(head_a_accessor, head_a, noc, row_base, value_block);
            write_streamed_value_slice<Kt, Vt, VtFull>(head_b_accessor, head_b, noc, row_base, value_block);
        } else {
            write_value_slice<Kt, Vt, VtFull>(head_a_accessor, head_a, noc, row_base, value_block);
            write_value_slice<Kt, Vt, VtFull>(head_b_accessor, head_b, noc, row_base, value_block);
        }
    } else {
        write_identity_pair<Kt, Vt, VtFull>(
            head_a_accessor, head_b_accessor, identity_tile, zero_tile, noc, row_base, value_block);
    }

    if (tail_active) {
        write_value_slice<Kt, Vt, VtFull>(tail_a_accessor, full_a, noc, row_base, value_block);
        write_value_slice<Kt, Vt, VtFull>(tail_b_accessor, full_b, noc, row_base, value_block);
    } else {
        write_identity_pair<Kt, Vt, VtFull>(
            tail_a_accessor, tail_b_accessor, identity_tile, zero_tile, noc, row_base, value_block);
    }
    identity_tile.pop_front(1);
    zero_tile.pop_front(1);
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

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t Vt_full, uint32_t summary_pair, uint32_t emit_tail_summaries>
TT_KERNEL void writer(
    uint32_t head,
    uint32_t value_block,
    uint32_t num_chunks,
    uint32_t group,
    uint32_t wrap_group,
    uint32_t split_in_group) {
    if constexpr (summary_pair) {
        if constexpr (emit_tail_summaries) {
            write_segmented_summary<Kt, Vt, Vt_full>(head, value_block, group, wrap_group, split_in_group);
        } else {
            write_summary<Kt, Vt, Vt_full>(head, value_block);
        }
    } else {
        write_recurrent<Ct, Kt, Vt, Vt_full>(head, value_block, num_chunks);
    }
}
