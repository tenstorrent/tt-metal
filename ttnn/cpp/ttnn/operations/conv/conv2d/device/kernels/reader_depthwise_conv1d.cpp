// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Height-sharded depthwise Conv1D reader using the Metalium 2.0 kernel-binding surface:
//   - CB-index CTAs -> dfb:: tokens (act / act_sharded / reader_indices)
//   - compile-time choices -> TT_KERNEL template arguments
//   - experimental::CB -> DataflowBuffer
// This kernel has no runtime args. There is no CONFIG_TENSOR_IN_DRAM path
// (the depthwise height-sharded indices are always L1-resident).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/local_tensor_accessor.h"
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
template <
    uint32_t stride_w,
    uint32_t conv_act_c_read_bytes,
    uint32_t window_outer,
    uint32_t window_inner,
    uint32_t act_block_num_tiles,
    uint32_t weight_size_h,
    uint32_t weight_size_w,
    uint32_t conv_act_size_w_padded,
    uint32_t act_block_w_extra_align_bytes,
    uint32_t act_num_blocks_h,
    uint32_t coalesce_kw_reads>
TT_KERNEL void kernel_main() {
    // Depthwise reuses the common reader arg slot that non-depthwise height-sharded conv uses for
    // activation reuse. Activation reuse is unsupported for the 1D depthwise path.

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
    // The resident activation shard and reader-indices config are L1-resident; the depthwise path
    // has no DRAM-config variant.
    DataflowBuffer cb_reader_indices(dfb::reader_indices);
    volatile tt_l1_ptr uint32_t* packed_reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_reader_indices.get_write_ptr());

    uint32_t reader_idx = 0;

    constexpr uint32_t num_coalesced_reads = coalesce_kw_reads ? weight_size_w : 1;
    constexpr uint32_t coalesced_read_bytes = num_coalesced_reads * conv_act_c_read_bytes;
    static_assert(!coalesce_kw_reads || weight_size_h == 1);
    static_assert(!coalesce_kw_reads || window_outer == 1);
    static_assert(!coalesce_kw_reads || window_inner == weight_size_w);

    reader_offset_idx = 0;
    uint32_t act_l1_offset = 0;
    LocalTensorAccessor<uint8_t> sharded_act(tensor::input);
    uint32_t act_l1_read_addr = sharded_act.get_bank_base_address();

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

        // Advance past the last segment word to the next block's count word. The loop above stops
        // on the last segment, while the shared read_sticks() helper performs this increment itself.
        start_reader_idx = reader_idx + 1;
    }
}
