// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp (height-sharded conv2d).
// Algorithm body identical to the legacy kernel; only the host-binding surface is migrated:
//   - CB-index CTAs -> dfb:: tokens (act / act_sharded / reader_indices)
//   - remaining positional CTAs -> get_arg(args::name)
//   - RTAs (core_index, remaining_tiles_to_push) -> get_arg(args::name)
//   - DRAM config-tensor read uses tensor::reader_indices (CONFIG_TENSOR_IN_DRAM path)
//   - conv_reader_common.hpp helpers are templated on the CB-object type, so the DataflowBuffer
//     constructed here from the dfb:: constexpr index is passed to them directly.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "conv_reader_common.hpp"

void kernel_main() {
    constexpr uint32_t dilation_h = get_arg(args::dilation_h);
    constexpr uint32_t dilation_w = get_arg(args::dilation_w);
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t conv_act_c_read_bytes = get_arg(args::conv_act_c_read_bytes);
    // need to have these as compile-time, they are inner loop bounds / unroll loops / constexpr conditionals based on
    // them
    constexpr uint32_t window_outer = get_arg(args::window_outer);
    constexpr uint32_t act_block_num_tiles = get_arg(args::act_block_num_tiles);
    constexpr uint32_t weight_size_h = get_arg(args::weight_size_h);
    constexpr uint32_t weight_size_w = get_arg(args::weight_size_w);
    constexpr uint32_t conv_act_size_w_padded = get_arg(args::conv_act_size_w_padded);
    constexpr uint32_t act_block_w_extra_align_bytes = get_arg(args::act_block_w_extra_align_bytes);
    constexpr uint32_t act_num_blocks_h = get_arg(args::act_num_blocks_h);

    constexpr bool needs_act_block_zero_out = get_arg(args::needs_act_block_zero_out) == 1;
    constexpr uint32_t cb_id_act = dfb::act;

    constexpr bool split_reader_enabled = get_arg(args::split_reader_enabled);
    constexpr bool activation_reuse_enabled = get_arg(args::activation_reuse_enabled);

    DataflowBuffer cb_act(cb_id_act);

    uint32_t core_index = get_arg(args::core_index);

    // Reader-indices base. On the resident (L1) path the config slice already lives in L1, so it is
    // reached by base address from a local TensorAccessor (tensor::reader_indices) — no borrowed CB.
    // On the DRAM-config path the slice is DMA'd into a fresh L1 DFB (dfb::reader_indices) first.
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr;
#ifdef CONFIG_TENSOR_IN_DRAM
    DataflowBuffer cb_reader_idx(dfb::reader_indices);
    packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_idx.get_write_ptr());
    {
        const auto config_accessor = TensorAccessor(tensor::reader_indices);
        constexpr uint32_t config_page_size = get_arg(args::config_page_size);
        Noc().async_read(config_accessor, cb_reader_idx, config_page_size, {.page_id = core_index}, {});
        Noc().async_read_barrier();
        cb_reader_idx.push_back(1);
    }
#else
    packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::reader_indices).get_noc_addr(0)));
    (void)core_index;
#endif

    // Activation reuse args
    constexpr uint32_t act_reuse_cb_tiles = get_arg(args::act_reuse_cb_tiles);
    constexpr uint32_t act_block_w_tiles = get_arg(args::act_block_w_tiles);
    constexpr bool readers_process_full_image_widths = get_arg(args::readers_process_full_image_widths) == 1;
    constexpr uint32_t image_width_tiles = get_arg(args::image_width_tiles);
    constexpr uint32_t output_image_width = get_arg(args::output_image_width);
    constexpr uint32_t window_reuse_offset = get_arg(args::window_reuse_offset);
    constexpr bool need_to_push_remaining_tiles = get_arg(args::need_to_push_remaining_tiles) == 1;
    constexpr bool single_core_processes_multiple_batches = get_arg(args::single_core_processes_multiple_batches) == 1;

    uint32_t remaining_tiles_to_push = get_arg(args::remaining_tiles_to_push);

    Noc noc;

    if constexpr (needs_act_block_zero_out) {
        zero_out_tiles<cb_id_act>(noc, cb_act);
    }

    constexpr uint32_t window_outer_offset = conv_act_size_w_padded * conv_act_c_read_bytes * dilation_h;

    // LOOP TO FILL READER INDICES

    uint32_t reader_idx = 0;

    // TODO: need to make the read coalescing optimization cleaner
    // pass coalesce_window_inner_reads as a compile time arg and num_coalesced_reads so we can constexpr the if
    // currently works for the case of num_coalesced_reads == weight_size_w since these reads are contiguous on both
    // src/dst side we check if window_inner == weight_size_w to make sure coalescing is legal along full window_inner
    // so the loop can be removed
    constexpr uint32_t num_coalesced_reads = weight_size_w;
    constexpr uint32_t coalesced_read_bytes =
        ((dilation_w == 1) ? num_coalesced_reads * conv_act_c_read_bytes : conv_act_c_read_bytes);
    // the conditional selecting between coalescing and no-colescing must be constexpr to that compiler can optimized
    // the other path away this has shown to be a big perf win

    // coalesce reads along weight_size_w. The resident activation shard is reached by L1 base address
    // from a local TensorAccessor (tensor::act_sharded), not a borrowed self-loop CB.
    uint32_t act_l1_read_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::act_sharded).get_noc_addr(0));

    static_assert(coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
    experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    uint32_t start_reader_idx = 0;
    uint32_t l1_write_addr_act = 0;
    const uint32_t cb_start_addr = cb_act.get_write_ptr();
    for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
        if constexpr (activation_reuse_enabled) {
            l1_write_addr_act = cb_start_addr;
            cb_act.evil_set_write_ptr(l1_write_addr_act);
        }
        uint32_t reader_offset = act_l1_read_addr;
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            reader_idx = start_reader_idx;

            if constexpr (!activation_reuse_enabled) {
                cb_act.reserve_back(act_block_num_tiles);
                l1_write_addr_act = cb_act.get_write_ptr();

                read_sticks<
                    dilation_w,
                    coalesced_read_bytes,
                    conv_act_c_read_bytes,
                    act_block_w_extra_align_bytes,
                    stride_w_bytes,
                    weight_size_w,
                    stride_w>(noc, packed_reader_indices_ptr, reader_offset, l1_write_addr_act, reader_idx);

                noc.async_read_barrier();
                cb_act.push_back(act_block_num_tiles);
                reader_offset += window_outer_offset;
            } else {
                read_sticks_activation_reuse<
                    coalesced_read_bytes,
                    conv_act_c_read_bytes,
                    act_block_w_extra_align_bytes,
                    window_outer_offset,
                    weight_size_w,
                    stride_w,
                    weight_size_h,
                    cb_id_act,
                    act_reuse_cb_tiles,
                    act_block_w_tiles,
                    readers_process_full_image_widths,
                    image_width_tiles,
                    output_image_width,
                    window_reuse_offset,
                    single_core_processes_multiple_batches>(
                    noc,
                    cb_act,
                    packed_reader_indices_ptr,
                    act_l1_read_addr,
                    l1_write_addr_act,
                    reader_idx,
                    cb_start_addr);
            }
        }

        start_reader_idx = reader_idx;
        if constexpr (split_reader_enabled) {
            // Increment reader index for the next number of segments (number of segments for other reader)
            start_reader_idx += (static_cast<uint32_t>(packed_reader_indices_ptr[reader_idx] & 0xffff) + 1);
        }
    }

    if constexpr (activation_reuse_enabled) {
        // Last core sometimes has less work to do, but we still need to push the same number of tiles
        // to avoid blocking compute kernels
        if constexpr (need_to_push_remaining_tiles) {
            push_remaining_tiles<cb_id_act, act_block_w_tiles, image_width_tiles>(
                cb_act, remaining_tiles_to_push, cb_start_addr);
        }
    }

    // Drain outstanding NOC reads/writes/atomics before returning (Metal 2.0 FW epilogue does not).
    noc.async_full_barrier();
}
