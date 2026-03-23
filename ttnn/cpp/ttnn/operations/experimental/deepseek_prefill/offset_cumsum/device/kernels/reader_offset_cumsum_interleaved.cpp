// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Offset Cumsum Kernel
//
// Computes per-device dispatch offsets into each expert's token buffer by
// taking a shifted (exclusive) prefix sum over gathered per-device histograms.
//
// Inputs:
//   - input [H, W]: UINT32 interleaved tensor of per-device expert histograms
//     (H = num_devices, W = n_routed_experts). Produced by all_gather of each
//     device's masked_bincount output.
//
// Outputs:
//   - offsets [H, W]: row k = sum of input rows 0..k-1 (row 0 is all zeros).
//     These are the starting positions where each device writes into each
//     expert's buffer — hence the name "offsets" rather than prefix sum.
//   - totals  [1, W]: sum of all H input rows (total tokens per expert).
//
// The kernel loops over H rows. Before the loop it writes a zeroed row to
// offsets[0]. On each iteration it reads input row h, adds it element-wise into
// the running sum, then writes the updated sum to offsets[h+1] (or to totals
// on the final iteration).
//
// Design choices:
//  - This is a single data-movement kernel that reads, computes,
// and writes — there is no separate compute or writer kernel. W is typically
// n_routed_experts (e.g. 256 for DeepSeek-V3), so the element-wise add is a
// short scalar loop on UINT32 values in L1 and does not warrant SFPU/FPU
// compute.
//  - Because there is only one kernel there is no producer/consumer
// relationship, so CBs are used as raw L1 scratch (no cb_push_back /
// cb_pop_front).
//  - The partial sums are called "offsets" because in the MoE
// dispatch context they are the starting write positions for each device into
// each expert's token buffer.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_offsets_addr = get_arg_val<uint32_t>(1);
    uint32_t dst_totals_addr = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool dst_offsets_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool dst_totals_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t totals_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t W = get_compile_time_arg_val(8);
    constexpr uint32_t H = get_compile_time_arg_val(9);

    constexpr uint32_t src_accessor_offset = 10;
    constexpr auto src_args = TensorAccessorArgs<src_accessor_offset>();
    const auto src_accessor = TensorAccessor(src_args, src_addr, input_page_size);

    constexpr uint32_t dst_offsets_args_offset = src_args.next_compile_time_args_offset();
    constexpr auto dst_offsets_args = TensorAccessorArgs<dst_offsets_args_offset>();
    const auto dst_offsets_accessor = TensorAccessor(dst_offsets_args, dst_offsets_addr, offsets_page_size);

    constexpr uint32_t dst_totals_args_offset = dst_offsets_args.next_compile_time_args_offset();
    constexpr auto dst_totals_args = TensorAccessorArgs<dst_totals_args_offset>();
    const auto dst_totals_accessor = TensorAccessor(dst_totals_args, dst_totals_addr, totals_page_size);

    uint32_t out_cb_addr = get_write_ptr(cb_id_out0);
    volatile tt_l1_ptr uint32_t* running_sum = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_cb_addr);

    for (uint32_t i = 0; i < W; i++) {
        running_sum[i] = 0;
    }

    noc_async_write(out_cb_addr, dst_offsets_accessor.get_noc_addr(0), offsets_page_size);
    noc_async_write_barrier();

    uint32_t in_cb_addr = get_write_ptr(cb_id_in0);

    // The three tensors are allocated independently and could in principle have
    // different aligned page sizes (even though in practice they'll likely be the
    // same, since they all share W and dtype). This doesn't cause correctness
    // issues because:
    //  - The inner loop always operates on exactly W elements and ignores any
    //    padding bytes beyond the W-th element.
    //  - NOC writes use the destination tensor's own page size (offsets_page_size
    //    or totals_page_size)

    for (uint32_t h = 0; h < H; h++) {
        noc_async_read_page(h, src_accessor, in_cb_addr);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_cb_addr);
        for (uint32_t i = 0; i < W; i++) {
            running_sum[i] += stick[i];
        }

        if (h < H - 1) {
            noc_async_write(out_cb_addr, dst_offsets_accessor.get_noc_addr(h + 1), offsets_page_size);
        } else {
            noc_async_write(out_cb_addr, dst_totals_accessor.get_noc_addr(0), totals_page_size);
        }
        noc_async_write_barrier();
    }
}
