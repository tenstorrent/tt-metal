// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Offset Cumsum Kernel
//
// Computes this device's global dispatch offset into each expert's token buffer.
//
// The global offset for each expert combines two components:
//   1. Local offset: shifted prefix sum across devices (sum of rows 0..row_idx-1)
//   2. Expert region offset: exclusive prefix sum of total token counts within each
//      chip's expert group (experts_per_chip stride)
//
// Inputs:
//   - input [H, W]: UINT32 interleaved tensor of per-device expert histograms
//     (H = num_devices, W = n_routed_experts). Produced by all_gather of each
//     device's masked_bincount output.
//
// Outputs:
//   - offsets         [1, W]: global dispatch offsets (local_offset + expert_region_offset)
//   - totals          [1, W]: sum of all H input rows (total tokens per expert)
//   - expert_region   [1, W]: expert region offsets only (shared component — exclusive
//                             prefix sum of tile-aligned totals within each chip group)
//
// Runtime args:
//   - src_addr, dst_offsets_addr, dst_totals_addr, dst_expert_region_addr: buffer addresses
//   - row_idx: which row of the prefix sum this device keeps

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_offsets_addr = get_arg_val<uint32_t>(1);
    uint32_t dst_totals_addr = get_arg_val<uint32_t>(2);
    uint32_t dst_expert_region_addr = get_arg_val<uint32_t>(3);
    uint32_t row_idx = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_local = get_compile_time_arg_val(2);
    constexpr bool src_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool dst_offsets_is_dram = (bool)get_compile_time_arg_val(4);
    constexpr bool dst_totals_is_dram = (bool)get_compile_time_arg_val(5);
    constexpr bool dst_expert_region_is_dram = (bool)get_compile_time_arg_val(6);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t totals_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t expert_region_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t W = get_compile_time_arg_val(11);
    constexpr uint32_t H = get_compile_time_arg_val(12);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(13);

    constexpr uint32_t src_accessor_offset = 14;
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

    // running_sum accumulates totals across all H rows
    uint32_t out_cb_addr = get_write_ptr(cb_id_out0);
    volatile tt_l1_ptr uint32_t* running_sum = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_cb_addr);

    // local_off stores the local offsets (sum of rows 0..row_idx-1) for later combination
    uint32_t local_cb_addr = get_write_ptr(cb_id_local);
    volatile tt_l1_ptr uint32_t* local_off = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_cb_addr);

    for (uint32_t i = 0; i < W; i++) {
        running_sum[i] = 0;
        local_off[i] = 0;
    }

    uint32_t in_cb_addr = get_write_ptr(cb_id_in0);

    for (uint32_t h = 0; h < H; h++) {
        noc_async_read_page(h, src_accessor, in_cb_addr);
        noc_async_read_barrier();

        volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_cb_addr);
        for (uint32_t i = 0; i < W; i++) {
            running_sum[i] += stick[i];
        }

        // Save local offsets when we've accumulated exactly rows 0..row_idx-1
        if (h + 1 == row_idx) {
            for (uint32_t i = 0; i < W; i++) {
                local_off[i] = running_sum[i];
            }
        }
    }

    // --- Post-loop: running_sum now contains totals ---

    // 1. Write totals to DRAM
    noc_async_write(out_cb_addr, dst_totals_accessor.get_noc_addr(0), totals_page_size);
    noc_async_write_barrier();

    // 2. Compute exclusive prefix sum of tile-aligned totals, grouped by experts_per_chip
    //    Pad each expert's count to TILE_HEIGHT so each expert starts at a tile boundary
    //    For group [a, b, c, d] -> [0, align(a), align(a)+align(b), align(a)+align(b)+align(c)]
    constexpr uint32_t num_chips = W / experts_per_chip;
    for (uint32_t g = 0; g < num_chips; g++) {
        uint32_t prefix = 0;
        for (uint32_t i = 0; i < experts_per_chip; i++) {
            uint32_t idx = g * experts_per_chip + i;
            uint32_t val = running_sum[idx];
            running_sum[idx] = prefix;
            prefix += (val + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT * tt::constants::TILE_HEIGHT;
        }
    }

    // 3. Write expert region offsets (shared component, before adding local offset) to DRAM
    noc_async_write(out_cb_addr, dst_expert_region_accessor.get_noc_addr(0), expert_region_page_size);
    noc_async_write_barrier();

    // 4. Add saved local offsets to get global offsets
    for (uint32_t i = 0; i < W; i++) {
        running_sum[i] += local_off[i];
    }

    // 5. Write global offsets to DRAM
    noc_async_write(out_cb_addr, dst_offsets_accessor.get_noc_addr(0), offsets_page_size);
    noc_async_write_barrier();
}
