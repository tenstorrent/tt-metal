// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Offset Cumsum Kernel
//
// Computes this device's dispatch offset into each expert's token buffer by
// taking a shifted (exclusive) prefix sum over a subgroup-scoped window of
// gathered per-device histograms.
//
// Inputs:
//   - input [full_H, W]: UINT32 interleaved tensor of per-device expert histograms
//     (full_H = num_devices along cluster_axis, W = n_routed_experts). Produced
//     by all_gather across the full mesh.
//
// Outputs:
//   - offsets [1, W]: the starting positions where this device writes into each
//     expert's buffer, computed as the shifted prefix sum across this chip's
//     subgroup only. Equivalent to sum of input rows [row_start .. row_start +
//     row_idx - 1] (row_idx == 0 means all zeros).
//   - totals  [1, W]: sum of input rows [row_start .. row_start + H - 1] where
//     H is the compile-time subgroup row count (= dispatch_group_size).
//
// Runtime args:
//   - src_addr, dst_offsets_addr, dst_totals_addr: buffer addresses
//   - row_idx: subgroup-local row index for this chip (0..H-1)
//   - row_start: global row of the gathered tensor where this subgroup begins
//
// With num_dispatch_subgroups == 1, row_start == 0 and the behavior reduces
// exactly to the pre-subgroup implementation.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_offsets_addr = get_arg_val<uint32_t>(1);
    uint32_t dst_totals_addr = get_arg_val<uint32_t>(2);
    uint32_t row_idx = get_arg_val<uint32_t>(3);
    uint32_t row_start = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool dst_offsets_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool dst_totals_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t totals_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);
    // H is the subgroup row count (= dispatch_group_size). Under subgroups this
    // is smaller than the gathered tensor's full height.
    constexpr uint32_t H = get_compile_time_arg_val(9);

    constexpr uint32_t src_accessor_offset = 10;
    constexpr auto src_args = TensorAccessorArgs<src_accessor_offset>();
    const auto src_accessor = TensorAccessor(src_args, src_addr);

    constexpr uint32_t dst_offsets_args_offset = src_args.next_compile_time_args_offset();
    constexpr auto dst_offsets_args = TensorAccessorArgs<dst_offsets_args_offset>();
    const auto dst_offsets_accessor = TensorAccessor(dst_offsets_args, dst_offsets_addr);

    constexpr uint32_t dst_totals_args_offset = dst_offsets_args.next_compile_time_args_offset();
    constexpr auto dst_totals_args = TensorAccessorArgs<dst_totals_args_offset>();
    const auto dst_totals_accessor = TensorAccessor(dst_totals_args, dst_totals_addr);

    uint32_t out_cb_addr = get_write_ptr(cb_id_out0);
    volatile tt_l1_ptr uint32_t* running_sum = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_cb_addr);

    for (uint32_t i = 0; i < W; i++) {
        running_sum[i] = 0;
    }

    // row_idx == 0: offset is all zeros — write immediately
    if (row_idx == 0) {
        noc_async_write(out_cb_addr, dst_offsets_accessor.get_noc_addr(0), offsets_page_size);
        noc_async_write_barrier();
    }

    uint32_t in_cb_addr = get_write_ptr(cb_id_in0);

    // Iterate over the subgroup's H rows of the gathered tensor starting at row_start.
    // Offsets output is written after accumulating `row_idx` rows (subgroup-local
    // prefix sum); totals output is written after the full subgroup sum.
    for (uint32_t i = 0; i < H; i++) {
        uint32_t h = row_start + i;
        noc_async_read_page(h, src_accessor, in_cb_addr);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_cb_addr);
        for (uint32_t j = 0; j < W; j++) {
            running_sum[j] += stick[j];
        }

        // Write offsets when we've accumulated exactly the first `row_idx` rows of the subgroup
        if (i + 1 == row_idx) {
            noc_async_write(out_cb_addr, dst_offsets_accessor.get_noc_addr(0), offsets_page_size);
            noc_async_write_barrier();
        }

        // Write totals on the final iteration (last row of the subgroup)
        if (i == H - 1) {
            noc_async_write(out_cb_addr, dst_totals_accessor.get_noc_addr(0), totals_page_size);
            noc_async_write_barrier();
        }
    }
}
