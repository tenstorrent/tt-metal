// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

#include "convert_to_hwc_writer_impl.hpp"

template <
    uint32_t num_output_channels_padded,  // padded output channels (min 8)
    uint32_t num_full_tiles,
    uint32_t total_tiles_per_block,
    uint32_t initial_write_stick_offset,
    uint32_t element_size_bytes,
    uint32_t is_input_in_dram,
    uint32_t input_block_size_sticks_per_core,
    uint32_t l1_write_output_addr_stride>
TT_KERNEL void reader_writer() {
    uint32_t dram_base_read_addr = 0;
    if constexpr (is_input_in_dram != 0) {
        const TensorAccessor input(tensor::input);
        dram_base_read_addr = input.get_bank_base_address();
    }
    uint32_t args_idx = 0;
    const uint32_t num_blocks = get_vararg(args_idx++);

    Noc noc;
    uint32_t l1_input_read_addr = 0;
    if constexpr (is_input_in_dram == 0) {
        DataflowBuffer input(dfb::input);
        l1_input_read_addr = input.get_read_ptr();
    }
    DataflowBuffer batch(dfb::batch);
    DataflowBuffer transpose(dfb::transpose);
    DataflowBuffer output(dfb::output);
    uint32_t l1_output_write_addr =
        output.get_write_ptr() + initial_write_stick_offset * num_output_channels_padded * element_size_bytes;

    // Process each blocked transfer group
    for (uint32_t block_id = 0; block_id < num_blocks; block_id++) {
        batch.reserve_back(input_block_size_sticks_per_core);

        // Process all transfers in this group.
        const uint32_t group_size = get_vararg(args_idx++);
        for (uint32_t transfer_idx = 0; transfer_idx < group_size; transfer_idx++) {
            uint32_t src_x = get_vararg(args_idx++);
            uint32_t src_y = get_vararg(args_idx++);
            uint32_t src_offset_bytes = get_vararg(args_idx++);
            uint32_t dst_offset_bytes = get_vararg(args_idx++);
            uint32_t transfer_size_bytes = get_vararg(args_idx++);
            uint32_t bank_id = get_vararg(args_idx++);

            // dst_offset_bytes is already relative to block buffer start (includes channel * block_size + column)
            if constexpr (is_input_in_dram != 0) {
                // DRAM bank-id read via the AllocatorBank<DRAM> endpoint. Folding src_offset_bytes into the
                // bank address offset is equivalent to the legacy bank-id addr-gen result plus
                // src_offset_bytes: both land the offset in the NOC address's low bits below
                // NOC_ADDR_COORD_SHIFT (see the DRAM bank-id addr-gen in dataflow_api_addrgen.h).
                AllocatorBank<AllocatorBankType::DRAM> dram_bank;
                noc.async_read(
                    dram_bank,
                    batch,
                    transfer_size_bytes,
                    {.bank_id = bank_id, .addr = dram_base_read_addr + src_offset_bytes},
                    {.offset_bytes = dst_offset_bytes});
            } else {
                UnicastEndpoint src_ep;
                noc.async_read(
                    src_ep,
                    batch,
                    transfer_size_bytes,
                    {.noc_x = src_x, .noc_y = src_y, .addr = l1_input_read_addr + src_offset_bytes},
                    {.offset_bytes = dst_offset_bytes});
            }
        }

        noc.async_read_barrier();
        batch.push_back(input_block_size_sticks_per_core);

        convert_to_hwc::write_transposed_block<
            num_output_channels_padded,
            num_full_tiles,
            total_tiles_per_block,
            true,
            element_size_bytes,
            l1_write_output_addr_stride>(noc, transpose, l1_output_write_addr);
    }
}
