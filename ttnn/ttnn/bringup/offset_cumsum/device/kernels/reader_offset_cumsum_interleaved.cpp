// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Offset Cumsum Kernel
//
// Computes every device's global dispatch offset into each expert's token buffer.
//
// The global offset for a device combines two components:
//   1. Local offset: the elementwise sum of the input rows before that device's own row
//   2. Expert region offset: exclusive prefix sum of tile-aligned total token counts within
//      each chip's expert group (experts_per_chip stride)
//
// Inputs:
//   - input [H, W]: UINT32 interleaved tensor of per-device expert histograms
//     (H = devices along cluster_axis, W = n_routed_experts). Produced by all_gather of each
//     device's masked_bincount output.
//
// Outputs:
//   - offsets         [1, W]: this device's global dispatch offsets
//   - totals          [1, W]: sum of all H input rows (total tokens per expert)
//   - expert_region   [1, W]: expert region offsets only (shared component — exclusive
//                             prefix sum of tile-aligned totals within each chip group)
//   - all_offsets     [H, W]: row r is device r's `offsets`. A run is the span of an expert's
//                             buffer that one source device writes; any device can size any run.
//                             Rows include expert_region, so run r's length is row[r+1] - row[r],
//                             and the last run's is totals + expert_region - row[H-1],
//                             NOT totals - row[H-1].
//
// Runtime args:
//   - src_addr, dst_offsets_addr, dst_totals_addr, dst_expert_region_addr, dst_all_offsets_addr
//   - row_idx: which row of the prefix sum this device keeps in `offsets`

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/constants.hpp>

// The kernel is bound by scalar L1 accesses, not NOC traffic. Both helpers walk N columns at a time so
// the running sums stay in registers while each row is still read contiguously.

// sums[i0 + j] = sum over the H rows of in[.][i0 + j]
template <uint32_t N, uint32_t H, uint32_t in_stride>
FORCE_INLINE void sum_columns(volatile tt_l1_ptr uint32_t* in, volatile tt_l1_ptr uint32_t* sums, uint32_t i0) {
    uint32_t acc[N] = {};
    for (uint32_t h = 0; h < H; h++) {
        volatile tt_l1_ptr uint32_t* src = in + h * in_stride + i0;
#pragma GCC unroll N
        for (uint32_t j = 0; j < N; j++) {
            acc[j] += src[j];
        }
    }
#pragma GCC unroll N
    for (uint32_t j = 0; j < N; j++) {
        sums[i0 + j] = acc[j];
    }
}

// rows[r][i0 + j] = base[i0 + j] + sum over rows 0..r-1 of in[.][i0 + j]
template <uint32_t N, uint32_t H, uint32_t in_stride, uint32_t row_stride>
FORCE_INLINE void emit_rows(
    volatile tt_l1_ptr uint32_t* in,
    volatile tt_l1_ptr uint32_t* base,
    volatile tt_l1_ptr uint32_t* rows,
    uint32_t i0) {
    uint32_t acc[N];
#pragma GCC unroll N
    for (uint32_t j = 0; j < N; j++) {
        acc[j] = base[i0 + j];
    }
    for (uint32_t r = 0; r < H; r++) {
        volatile tt_l1_ptr uint32_t* src = in + r * in_stride + i0;
        volatile tt_l1_ptr uint32_t* dst = rows + r * row_stride + i0;
#pragma GCC unroll N
        for (uint32_t j = 0; j < N; j++) {
            dst[j] = acc[j];
            acc[j] += src[j];
        }
    }
}

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_offsets_addr = get_arg_val<uint32_t>(1);
    uint32_t dst_totals_addr = get_arg_val<uint32_t>(2);
    uint32_t dst_expert_region_addr = get_arg_val<uint32_t>(3);
    uint32_t dst_all_offsets_addr = get_arg_val<uint32_t>(4);
    uint32_t row_idx = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_rows = get_compile_time_arg_val(2);

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_out0(cb_id_out0);
    CircularBuffer cb_rows(cb_id_rows);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t totals_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t expert_region_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t all_offsets_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);
    constexpr uint32_t H = get_compile_time_arg_val(9);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(10);

    constexpr uint32_t src_accessor_offset = 11;
    constexpr auto src_args = TensorAccessorArgs<src_accessor_offset>();
    const auto src_accessor = TensorAccessor(src_args, src_addr);

    constexpr uint32_t dst_offsets_args_offset = src_args.next_compile_time_args_offset();
    constexpr auto dst_offsets_args = TensorAccessorArgs<dst_offsets_args_offset>();
    const auto dst_offsets_accessor = TensorAccessor(dst_offsets_args, dst_offsets_addr);

    constexpr uint32_t dst_totals_args_offset = dst_offsets_args.next_compile_time_args_offset();
    constexpr auto dst_totals_args = TensorAccessorArgs<dst_totals_args_offset>();
    const auto dst_totals_accessor = TensorAccessor(dst_totals_args, dst_totals_addr);

    constexpr uint32_t dst_expert_region_args_offset = dst_totals_args.next_compile_time_args_offset();
    constexpr auto dst_expert_region_args = TensorAccessorArgs<dst_expert_region_args_offset>();
    const auto dst_expert_region_accessor = TensorAccessor(dst_expert_region_args, dst_expert_region_addr);

    constexpr uint32_t dst_all_offsets_args_offset = dst_expert_region_args.next_compile_time_args_offset();
    constexpr auto dst_all_offsets_args = TensorAccessorArgs<dst_all_offsets_args_offset>();
    const auto dst_all_offsets_accessor = TensorAccessor(dst_all_offsets_args, dst_all_offsets_addr);

    // These CBs are plain L1 scratch, never pushed or popped: everything is staged and sent through
    // the write pointer.

    constexpr uint32_t COLS = 8;
    constexpr uint32_t W_blocked = W / COLS * COLS;

    // All H input rows, at input_page_size stride, so the input is read from DRAM once.
    constexpr uint32_t in_stride = input_page_size / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_in0.get_write_ptr());
    for (uint32_t h = 0; h < H; h++) {
        noc.async_read(src_accessor, cb_in0, input_page_size, {.page_id = h}, {.offset_bytes = h * input_page_size});
    }
    noc.async_read_barrier();

    // running_sum holds totals, then becomes the expert region offsets
    volatile tt_l1_ptr uint32_t* running_sum = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_out0.get_write_ptr());
    for (uint32_t i0 = 0; i0 < W_blocked; i0 += COLS) {
        sum_columns<COLS, H, in_stride>(in, running_sum, i0);
    }
    for (uint32_t i0 = W_blocked; i0 < W; i0++) {
        sum_columns<1, H, in_stride>(in, running_sum, i0);
    }

    // 1. Write totals to DRAM. The barrier is required: step 2 overwrites running_sum in place.
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out0),
        dst_totals_accessor,
        totals_page_size,
        {},
        {.page_id = 0});
    noc.async_write_barrier();

    // 2. Compute exclusive prefix sum of tile-aligned totals, grouped by experts_per_chip
    //    Pad each expert's count to TILE_HEIGHT so each expert starts at a tile boundary
    //    For group [a, b, c, d] -> [0, align(a), align(a)+align(b), align(a)+align(b)+align(c)]
    // One group per chip model-wide, which is not H: with several dispatch groups there are more
    // expert groups than devices on the cluster axis.
    constexpr uint32_t num_expert_groups = W / experts_per_chip;
    for (uint32_t g = 0; g < num_expert_groups; g++) {
        uint32_t prefix = 0;
        for (uint32_t i = 0; i < experts_per_chip; i++) {
            uint32_t idx = g * experts_per_chip + i;
            uint32_t val = running_sum[idx];
            running_sum[idx] = prefix;
            prefix += (val + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT * tt::constants::TILE_HEIGHT;
        }
    }

    // 3. Write expert region offsets (shared component, before adding any local offset) to DRAM.
    //    running_sum is only read from here on, so this write shares the final barrier.
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_out0),
        dst_expert_region_accessor,
        expert_region_page_size,
        {},
        {.page_id = 0});

    // 4. Row r is expert_region + (rows 0..r-1). `offsets` is written from row row_idx of the same
    //    buffer, so the two outputs cannot disagree.
    constexpr uint32_t row_stride = all_offsets_page_size / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* rows = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_rows.get_write_ptr());
    for (uint32_t i0 = 0; i0 < W_blocked; i0 += COLS) {
        emit_rows<COLS, H, in_stride, row_stride>(in, running_sum, rows, i0);
    }
    for (uint32_t i0 = W_blocked; i0 < W; i0++) {
        emit_rows<1, H, in_stride, row_stride>(in, running_sum, rows, i0);
    }
    for (uint32_t r = 0; r < H; r++) {
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_rows),
            dst_all_offsets_accessor,
            all_offsets_page_size,
            {.offset_bytes = r * all_offsets_page_size},
            {.page_id = r});
    }
    noc.async_write(
        use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_rows),
        dst_offsets_accessor,
        offsets_page_size,
        {.offset_bytes = row_idx * all_offsets_page_size},
        {.page_id = 0});
    noc.async_write_barrier();
}
