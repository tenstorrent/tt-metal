// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Offset Cumsum Kernel
//
// Computes every device's global dispatch offset into each expert's token buffer.
//
// The global offset for a device combines two components:
//   1. Local offset: the elementwise sum of the input rows before that device's own row
//   2. Expert region offset: exclusive prefix sum of total token counts within each
//      chip's expert group (experts_per_chip stride)
//
// Inputs:
//   - input [H, W]: UINT32 interleaved tensor of per-device expert histograms
//     (H = num_devices, W = n_routed_experts). Produced by all_gather of each
//     device's masked_bincount output.
//
// Outputs:
//   - offsets         [1, W]: this device's global dispatch offsets
//   - totals          [1, W]: sum of all H input rows (total tokens per expert)
//   - expert_region   [1, W]: expert region offsets only (shared component — exclusive
//                             prefix sum of tile-aligned totals within each chip group)
//   - all_offsets     [H, W]: every device's global dispatch offsets. Row r is what device r
//                             receives in `offsets`, so a consumer that must size a run it neither
//                             produced nor receives reads the boundary directly instead of
//                             exchanging it. Rows are absolute buffer positions, so they carry the
//                             expert_region component: a run's length is
//                             row[r+1] - row[r], and for the last row totals + expert_region -
//                             row[H-1] -- NOT totals - row[H-1].
//
// Runtime args:
//   - src_addr, dst_offsets_addr, dst_totals_addr, dst_expert_region_addr, dst_all_addr
//   - row_idx: which row of the prefix sum this device keeps in `offsets`

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include <tt-metalium/constants.hpp>

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_offsets_addr = get_arg_val<uint32_t>(1);
    uint32_t dst_totals_addr = get_arg_val<uint32_t>(2);
    uint32_t dst_expert_region_addr = get_arg_val<uint32_t>(3);
    uint32_t dst_all_addr = get_arg_val<uint32_t>(4);
    uint32_t row_idx = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_prefix = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_row = get_compile_time_arg_val(3);

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
    CircularBuffer cb_out0(cb_id_out0);
    CircularBuffer cb_prefix(cb_id_prefix);
    CircularBuffer cb_row(cb_id_row);
    constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(5);
    constexpr uint32_t totals_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t expert_region_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t all_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t W = get_compile_time_arg_val(9);
    constexpr uint32_t H = get_compile_time_arg_val(10);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(11);

    constexpr uint32_t src_accessor_offset = 12;
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

    constexpr uint32_t dst_all_args_offset = dst_expert_region_args.next_compile_time_args_offset();
    constexpr auto dst_all_args = TensorAccessorArgs<dst_all_args_offset>();
    const auto dst_all_accessor = TensorAccessor(dst_all_args, dst_all_addr);

    // These CBs are plain L1 scratch: nothing is ever pushed or popped, so read and write pointers
    // both stay at the fifo base and an async_write (which resolves the read pointer) sees what was
    // staged through the write pointer.
    //
    // running_sum accumulates totals across all H rows, then becomes the expert region offsets
    volatile tt_l1_ptr uint32_t* running_sum = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_out0.get_write_ptr());

    // prefix accumulates rows 0..r-1 during the second pass; row holds the row being emitted
    volatile tt_l1_ptr uint32_t* prefix = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_prefix.get_write_ptr());
    volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_row.get_write_ptr());

    for (uint32_t i = 0; i < W; i++) {
        running_sum[i] = 0;
    }

    for (uint32_t h = 0; h < H; h++) {
        noc.async_read(src_accessor, cb_in0, input_page_size, {.page_id = h}, {.offset_bytes = 0});
        noc.async_read_barrier();

        volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_in0.get_write_ptr());
        for (uint32_t i = 0; i < W; i++) {
            running_sum[i] += stick[i];
        }
    }

    // --- Post-loop: running_sum now contains totals ---

    // 1. Write totals to DRAM
    noc.async_write(cb_out0, dst_totals_accessor, totals_page_size, {}, {.page_id = 0});
    noc.async_write_barrier();

    // 2. Compute exclusive prefix sum of tile-aligned totals, grouped by experts_per_chip
    //    Pad each expert's count to TILE_HEIGHT so each expert starts at a tile boundary
    //    For group [a, b, c, d] -> [0, align(a), align(a)+align(b), align(a)+align(b)+align(c)]
    // Expert groups model-wide, which is not H: with more than one dispatch group there are more
    // groups than there are devices on the cluster axis.
    constexpr uint32_t num_expert_groups = W / experts_per_chip;
    for (uint32_t g = 0; g < num_expert_groups; g++) {
        uint32_t p = 0;
        for (uint32_t i = 0; i < experts_per_chip; i++) {
            uint32_t idx = g * experts_per_chip + i;
            uint32_t val = running_sum[idx];
            running_sum[idx] = p;
            p += (val + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT * tt::constants::TILE_HEIGHT;
        }
    }

    // 3. Write expert region offsets (shared component, before adding any local offset) to DRAM
    noc.async_write(cb_out0, dst_expert_region_accessor, expert_region_page_size, {}, {.page_id = 0});
    noc.async_write_barrier();

    // 4. Second pass over the rows, emitting every device's global offsets. Row r is
    //    expert_region + (rows 0..r-1), and `offsets` is the r == row_idx row of that same stream,
    //    so the two outputs cannot disagree.
    volatile tt_l1_ptr uint32_t* expert_region = running_sum;
    for (uint32_t i = 0; i < W; i++) {
        prefix[i] = 0;
    }

    for (uint32_t r = 0; r < H; r++) {
        for (uint32_t i = 0; i < W; i++) {
            row[i] = expert_region[i] + prefix[i];
        }
        noc.async_write(cb_row, dst_all_accessor, all_page_size, {}, {.page_id = r});
        if (r == row_idx) {
            noc.async_write(cb_row, dst_offsets_accessor, offsets_page_size, {}, {.page_id = 0});
        }
        noc.async_write_barrier();

        // Row r's own contribution belongs to iteration r+1, which is what keeps `prefix` exclusive.
        if (r + 1 < H) {
            noc.async_read(src_accessor, cb_in0, input_page_size, {.page_id = r}, {.offset_bytes = 0});
            noc.async_read_barrier();
            volatile tt_l1_ptr uint32_t* stick = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_in0.get_write_ptr());
            for (uint32_t i = 0; i < W; i++) {
                prefix[i] += stick[i];
            }
        }
    }
}
