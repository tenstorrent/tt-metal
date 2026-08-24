// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Height-sharded Conv2D activation reader using the Metalium 2.0 kernel-binding surface:
//   - CB-index CTAs -> dfb:: tokens (act / act_sharded / reader_indices)
//   - compile-time choices -> TT_KERNEL template arguments
//   - RTAs (core_index, remaining_tiles_to_push) -> TT_KERNEL function arguments
//   - DRAM config-tensor read uses tensor::reader_indices (CONFIG_TENSOR_IN_DRAM path)
//   - conv_reader_common.hpp helpers are templated on the CB-object type, so the DataflowBuffer
//     constructed here from the dfb:: constexpr index is passed to them directly.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "conv_reader_common.hpp"

template <
    uint32_t dilation_h,
    uint32_t dilation_w,
    uint32_t stride_w,
    uint32_t conv_act_c_read_bytes,
    uint32_t window_outer,
    uint32_t act_block_num_tiles,
    uint32_t weight_size_h,
    uint32_t weight_size_w,
    uint32_t conv_act_size_w_padded,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t act_num_blocks_h,
    uint32_t needs_act_block_zero_out,
    uint32_t split_reader_enabled,
    uint32_t activation_reuse_enabled,
    uint32_t config_tensor_in_dram,
    uint32_t config_page_size,
    uint32_t act_reuse_cb_tiles,
    uint32_t act_block_w_tiles,
    uint32_t readers_process_full_image_widths,
    uint32_t image_width_tiles,
    uint32_t output_image_width,
    uint32_t window_reuse_offset,
    uint32_t need_to_push_remaining_tiles,
    uint32_t single_core_processes_multiple_batches>
TT_KERNEL void kernel_main(uint32_t core_index, uint32_t remaining_tiles_to_push) {
    constexpr uint32_t cb_id_act = dfb::act;

    DataflowBuffer cb_act(cb_id_act);

    // On the resident path this borrowed DFB exposes the existing L1 config slice. On the DRAM-config
    // path it owns the destination into which the per-core page is read.
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr;
    DataflowBuffer cb_reader_idx(dfb::reader_indices);
    packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_idx.get_write_ptr());
    if constexpr (config_tensor_in_dram) {
        const auto config_accessor = TensorAccessor(tensor::reader_indices);
        Noc().async_read(config_accessor, cb_reader_idx, config_page_size, {.page_id = core_index}, {});
        Noc().async_read_barrier();
        cb_reader_idx.push_back(1);
    } else {
        (void)core_index;
    }

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

    // Coalesce reads along weight_size_w from the node-local input shard.
    LocalTensorAccessor<uint8_t> sharded_act(tensor::input);
    uint32_t act_l1_read_addr = sharded_act.get_bank_base_address();

    static_assert(coalesced_read_bytes <= NOC_MAX_BURST_SIZE);
    experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);

    constexpr uint32_t stride_w_bytes = dilation_w * conv_act_c_read_bytes;
    // Vertical (kernel-row) stride in the halo'd L1 shard: one padded input row. Same value as
    // window_outer_offset; used by the full-window gather (read_channels<weight_size_h>) to step between
    // kernel rows within a single K-block.
    constexpr uint32_t stride_h_bytes = window_outer_offset;
    // window_outer == 1  <=> the whole reduction window is kept in one K-block (full_inner_dim): the full
    //                        window (all weight_size_h kernel rows) must be
    //                        gathered per stick here.
    // window_outer > 1   <=> sliced per kernel row (normal spilling path): read one kernel row per outer
    //                        block, unchanged from before.
    // Mirrors the block-sharded reader, which derives the same flag.
    constexpr bool sliced_inner_dim = window_outer > 1;
    uint32_t start_reader_idx = 0;
    uint32_t l1_write_addr_act = 0;
    const uint32_t cb_start_addr = cb_act.get_write_ptr();
    for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
        if constexpr (activation_reuse_enabled) {
            l1_write_addr_act = cb_start_addr;
            get_local_cb_interface(dfb::act).fifo_wr_ptr = l1_write_addr_act;
        }
        uint32_t reader_offset = act_l1_read_addr;
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            reader_idx = start_reader_idx;

            if constexpr (!activation_reuse_enabled) {
                cb_act.reserve_back(act_block_num_tiles);
                l1_write_addr_act = cb_act.get_write_ptr();

                // read_activation_data branches on sliced_inner_dim:
                //   sliced_inner_dim == true  -> per-kernel-row read_sticks (unchanged spilling path;
                //                                the outer loop supplies filter_h blocks).
                //   sliced_inner_dim == false -> full-window read_channels<weight_size_h> gather: for each
                //                                stick it reads all weight_size_h kernel rows (each a
                //                                coalesced weight_size_w * Cin burst, stepping by
                //                                stride_h_bytes), laying the K columns out as [r][s][c] to
                //                                match the reuse=true full-window weight layout
                //                                (to_weight_special_padding_tile_layout).
                // In both cases read_activation_data issues the async_read_barrier and advances reader_offset
                // by window_outer_offset internally.
                read_activation_data<
                    sliced_inner_dim,
                    dilation_w,
                    coalesced_read_bytes,
                    conv_act_c_read_bytes,
                    act_block_w_extra_align_bytes,
                    stride_w_bytes,
                    weight_size_w,
                    stride_w,
                    weight_size_h,
                    window_outer_offset>(
                    noc,
                    packed_reader_indices_ptr,
                    reader_offset,
                    l1_write_addr_act,
                    reader_idx,
                    act_l1_read_addr,
                    stride_h_bytes);

                cb_act.push_back(act_block_num_tiles);
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
            push_remaining_tiles<act_block_w_tiles, image_width_tiles>(cb_act, remaining_tiles_to_push, cb_start_addr);
        }
    }

    // Drain outstanding NOC reads/writes/atomics before returning (Metal 2.0 FW epilogue does not).
    noc.async_full_barrier();
}
