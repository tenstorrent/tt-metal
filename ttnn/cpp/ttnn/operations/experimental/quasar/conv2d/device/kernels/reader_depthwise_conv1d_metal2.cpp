// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_depthwise_conv1d.cpp (1D depthwise conv2d, height-sharded).
// Algorithm body identical to the legacy kernel; only the host-binding surface is migrated:
//   - CB-index CTAs -> dfb:: tokens (act / act_sharded / reader_indices)
//   - remaining positional CTAs -> get_arg(args::name)
//   - experimental::CB -> DataflowBuffer
// The legacy kernel has no runtime args; this fork keeps zero RTAs. No CONFIG_TENSOR_IN_DRAM path
// (the depthwise height-sharded indices are always L1-resident).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

template <uint32_t read_bytes>
FORCE_INLINE void read_activation_stick(Noc noc, uint32_t l1_write_addr, uint32_t l1_read_addr) {
    if constexpr (read_bytes <= NOC_MAX_BURST_SIZE) {
        experimental::read_with_state<read_bytes>(noc, l1_write_addr, l1_read_addr);
    } else {
        UnicastEndpoint self_ep;
        noc.async_read(
            self_ep,
            CoreLocalMem<uint32_t>(l1_write_addr),
            read_bytes,
            experimental::local_addr(l1_read_addr, noc.get_noc_id()),
            {});
    }
}

// conv1D reader kernel
void kernel_main() {
    constexpr uint32_t stride_w = get_arg(args::stride_w);
    constexpr uint32_t conv_act_c_read_bytes = get_arg(args::conv_act_c_read_bytes);
    // need to have these as compile-time, they are inner loop bounds / unroll loops / constexpr conditionals based on
    // them
    constexpr uint32_t window_outer = get_arg(args::window_outer);
    constexpr uint32_t window_inner = get_arg(args::window_inner);
    constexpr uint32_t act_block_num_tiles = get_arg(args::act_block_num_tiles);
    constexpr uint32_t weight_size_h = get_arg(args::weight_size_h);
    constexpr uint32_t weight_size_w = get_arg(args::weight_size_w);
    constexpr uint32_t conv_act_size_w_padded = get_arg(args::conv_act_size_w_padded);
    constexpr uint32_t act_block_w_extra_align_bytes = get_arg(args::act_block_w_extra_align_bytes);
    constexpr uint32_t act_num_blocks_h = get_arg(args::act_num_blocks_h);
    // Depthwise reuses the common reader arg slot that non-depthwise height-sharded conv uses for
    // activation reuse. Activation reuse is unsupported for the 1D depthwise path.
    constexpr bool coalesce_kw_reads = get_arg(args::coalesce_kw_reads) == 1;

    // LOOP TO FILL READER OFFSETS
    /* We can add another loop to read chunks of a stick as well.
     * - Duplicate reader_offset for same stick X times (window_inner must be 1)
     * - New loop between outer and inner that loops X times reading from same stick
     * - Read conv_act_c_read_bytes / X each time
     * - Update l1_write_addr_act by conv_act_c_read_bytes
     */
    uint32_t reader_offsets[weight_size_w * weight_size_h];
    uint32_t reader_offset = 0;  // Constant offset for each pixel within filter window
    uint32_t reader_offset_idx = 0;
    for (uint32_t channel_stick_h = 0; channel_stick_h < weight_size_h; channel_stick_h++) {
        uint32_t reader_offset_row = reader_offset;
        for (uint32_t channel_stick_w = 0; channel_stick_w < weight_size_w; channel_stick_w++) {
            reader_offsets[reader_offset_idx++] = reader_offset_row++;
        }
        // -1 to go back to previous reader_offset
        reader_offset += conv_act_size_w_padded;
    }

    DataflowBuffer act_cb(dfb::act);
    Noc noc;

    // LOOP TO FILL READER INDICES
    // The resident activation shard and reader-indices config are L1-resident (the 1D depthwise path
    // has no DRAM-config variant); both are reached by L1 base address from local TensorAccessors
    // (tensor::act_sharded / tensor::reader_indices), not borrowed self-loop CBs.
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
        (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::reader_indices).get_noc_addr(0)));

    uint32_t reader_idx = 0;

    constexpr uint32_t num_coalesced_reads = coalesce_kw_reads ? weight_size_w : 1;
    constexpr uint32_t coalesced_read_bytes = num_coalesced_reads * conv_act_c_read_bytes;
    static_assert(!coalesce_kw_reads || weight_size_h == 1);
    static_assert(!coalesce_kw_reads || window_outer == 1);
    static_assert(!coalesce_kw_reads || window_inner == weight_size_w);

    reader_offset_idx = 0;
    uint32_t act_l1_offset = 0;
    uint32_t act_l1_read_addr = (uint32_t)NOC_LOCAL_ADDR_OFFSET(TensorAccessor(tensor::act_sharded).get_noc_addr(0));

    if constexpr (coalesced_read_bytes <= NOC_MAX_BURST_SIZE) {
        experimental::set_read_state<coalesced_read_bytes>(noc, act_l1_read_addr);
    }
    uint32_t start_reader_idx = 0;
    for (uint32_t bh = 0; bh < act_num_blocks_h; bh++) {
        for (uint32_t outer = 0; outer < window_outer; outer++) {
            // Reset reader_idx to finish act_block_h_datums
            reader_idx = start_reader_idx;

            act_cb.reserve_back(act_block_num_tiles);
            uint32_t l1_write_addr_act = act_cb.get_write_ptr();
            uint32_t reader_offset = act_l1_read_addr + (reader_offsets[reader_offset_idx] * conv_act_c_read_bytes);
            // #pragma GCC unroll 4 // unroll didn't help, but act_block_h_datums (loop bound) being const does help
            uint32_t two_reader_indices = packed_reader_indices_ptr[reader_idx];

            uint16_t num_elems = two_reader_indices & 0xffff;

            while (num_elems--) {
                reader_idx++;
                two_reader_indices = packed_reader_indices_ptr[reader_idx];

                uint16_t start_ind = two_reader_indices & 0xffff;
                uint16_t end_ind = two_reader_indices >> 16;

                for (uint16_t ind = start_ind; ind <= end_ind; ind += stride_w) {
                    act_l1_offset = reader_offset + (ind * conv_act_c_read_bytes);
                    read_activation_stick<coalesced_read_bytes>(noc, l1_write_addr_act, act_l1_offset);
                    l1_write_addr_act += (coalesced_read_bytes + act_block_w_extra_align_bytes);
                }
            }
            noc.async_read_barrier();
            act_cb.push_back(act_block_num_tiles);

            reader_offset_idx += window_inner;
        }
        reader_offset_idx = 0;

        start_reader_idx = reader_idx;
    }
}
