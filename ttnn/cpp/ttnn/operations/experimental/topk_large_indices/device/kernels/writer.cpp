// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

namespace {

FORCE_INLINE void add_logical_index_offset(uint32_t addr, uint32_t elements, uint32_t offset) {
    if (offset == 0) {
        return;
    }
    auto* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t i = 0; i < elements; ++i) {
        const uint32_t index = indices[i];
        if (index != 0xFFFFFFFFu) {
            indices[i] = index + offset;
        }
    }
}

FORCE_INLINE void remap_carried_indices(uint32_t addr, uint32_t elements, uint32_t carried_indices_addr) {
    if (carried_indices_addr == 0) {
        return;
    }
    auto* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    const auto* carried = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(carried_indices_addr);
    for (uint32_t i = 0; i < elements; ++i) {
        const uint32_t ordinal = indices[i];
        if (ordinal != 0xFFFFFFFFu) {
            indices[i] = carried[ordinal];
        }
    }
}

template <uint32_t source_slices_per_row, uint32_t output_slices_per_row, uint32_t slice_bytes>
FORCE_INLINE void copy_row_to_scratch(CircularBuffer& src_cb, CircularBuffer& scratch_cb, const Noc& noc) {
    static_assert(source_slices_per_row == 64 || source_slices_per_row == 128);
    static_assert(output_slices_per_row >= 1 && output_slices_per_row <= source_slices_per_row);
    constexpr uint32_t transfer_bytes = slice_bytes;
    static_assert(transfer_bytes <= NOC_MAX_BURST_SIZE);

    const uint32_t src_base = src_cb.get_read_ptr();
    const uint32_t dst_base = scratch_cb.get_write_ptr();
    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]), .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{}, transfer_bytes, local_src(src_base));

    for (uint32_t dst_slice = 0; dst_slice < output_slices_per_row; ++dst_slice) {
        // pack_untilize emits 16-element slices in face-pair order:
        // [top-left, bottom-left, top-right, bottom-right] for each tile column.
        const uint32_t tile_col = dst_slice >> 2;
        const uint32_t face_col = dst_slice & 0x1;
        const uint32_t face_row_offset = (dst_slice & 0x2) ? source_slices_per_row / 2 : 0;
        const uint32_t src_slice = (2 * tile_col) + face_col + face_row_offset;
        const uint32_t src_addr = src_base + src_slice * slice_bytes;
        const uint32_t dst_addr = dst_base + dst_slice * slice_bytes;
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(dst_addr),
            transfer_bytes,
            local_src(src_addr),
            {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}

template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename TensorAccessorT>
FORCE_INLINE void issue_reordered_row_write(
    CircularBuffer& src_cb,
    CircularBuffer& scratch_cb,
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t row,
    uint32_t row_bytes,
    uint32_t index_offset = 0,
    uint32_t carried_indices_addr = 0) {
    src_cb.wait_front(1);
    scratch_cb.reserve_back(1);
    copy_row_to_scratch<source_slices_per_row, output_slices_per_row, slice_bytes>(src_cb, scratch_cb, noc);
    src_cb.pop_front(1);

    scratch_cb.push_back(1);
    scratch_cb.wait_front(1);
    remap_carried_indices(scratch_cb.get_read_ptr(), row_bytes / sizeof(uint32_t), carried_indices_addr);
    add_logical_index_offset(scratch_cb.get_read_ptr(), row_bytes / sizeof(uint32_t), index_offset);
    noc.async_write(scratch_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

template <typename TensorAccessorT>
FORCE_INLINE void issue_contiguous_row_write(
    CircularBuffer& src_cb,
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t row,
    uint32_t row_bytes,
    uint32_t index_offset = 0,
    uint32_t carried_indices_addr = 0) {
    src_cb.wait_front(1);
    remap_carried_indices(src_cb.get_read_ptr(), row_bytes / sizeof(uint32_t), carried_indices_addr);
    add_logical_index_offset(src_cb.get_read_ptr(), row_bytes / sizeof(uint32_t), index_offset);
    noc.async_write(src_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

}  // namespace

void kernel_main() {
    const uint32_t values_addr = get_arg_val<uint32_t>(0);
    const uint32_t indices_addr = get_arg_val<uint32_t>(1);
    const uint32_t start_row = get_arg_val<uint32_t>(2);
    const uint32_t num_rows = get_arg_val<uint32_t>(3);
    const uint32_t index_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_values_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t cb_indices = get_compile_time_arg_val(2);
    constexpr uint32_t cb_indices_scratch = get_compile_time_arg_val(3);
    constexpr uint32_t values_page_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t source_slices_per_row = get_compile_time_arg_val(6);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(7);
    constexpr uint32_t values_slice_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t indices_slice_bytes = get_compile_time_arg_val(9);
    constexpr bool emit_values = get_compile_time_arg_val(10) != 0;
    constexpr uint32_t cb_carried_indices = get_compile_time_arg_val(11);
    constexpr bool carry_indices = get_compile_time_arg_val(12) != 0;
    constexpr auto values_args = TensorAccessorArgs<13>();
    constexpr auto indices_args = TensorAccessorArgs<values_args.next_compile_time_args_offset()>();

    const auto values = TensorAccessor(values_args, values_addr, values_page_bytes);
    const auto indices = TensorAccessor(indices_args, indices_addr, indices_page_bytes);
    CircularBuffer values_cb(cb_values);
    CircularBuffer indices_cb(cb_indices);
    CircularBuffer carried_indices_cb(cb_carried_indices);
    Noc noc;

    if constexpr (source_slices_per_row == 32) {
        for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
            const uint32_t row = start_row + local_row;
            uint32_t carried_indices_addr = 0;
            if constexpr (carry_indices) {
                carried_indices_cb.wait_front(1);
                carried_indices_addr = carried_indices_cb.get_read_ptr();
            }
            if constexpr (emit_values) {
                issue_contiguous_row_write(values_cb, noc, values, row, values_page_bytes);
            }
            issue_contiguous_row_write(
                indices_cb, noc, indices, row, indices_page_bytes, index_offset, carried_indices_addr);
            noc.async_writes_flushed();
            if constexpr (carry_indices) {
                carried_indices_cb.pop_front(1);
            }
            if constexpr (emit_values) {
                values_cb.pop_front(1);
            }
            indices_cb.pop_front(1);
        }
    } else {
        CircularBuffer values_scratch_cb(cb_values_scratch);
        CircularBuffer indices_scratch_cb(cb_indices_scratch);
        for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
            const uint32_t row = start_row + local_row;
            uint32_t carried_indices_addr = 0;
            if constexpr (carry_indices) {
                carried_indices_cb.wait_front(1);
                carried_indices_addr = carried_indices_cb.get_read_ptr();
            }
            if constexpr (emit_values) {
                issue_reordered_row_write<source_slices_per_row, output_slices_per_row, values_slice_bytes>(
                    values_cb, values_scratch_cb, noc, values, row, values_page_bytes);
            }
            issue_reordered_row_write<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                indices_cb,
                indices_scratch_cb,
                noc,
                indices,
                row,
                indices_page_bytes,
                index_offset,
                carried_indices_addr);

            noc.async_writes_flushed();
            if constexpr (carry_indices) {
                carried_indices_cb.pop_front(1);
            }
            if constexpr (emit_values) {
                values_scratch_cb.pop_front(1);
            }
            indices_scratch_cb.pop_front(1);
        }
    }

    noc.async_write_barrier();
}
