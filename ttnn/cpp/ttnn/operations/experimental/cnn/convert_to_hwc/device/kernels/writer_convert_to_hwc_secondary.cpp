// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

#include "convert_to_hwc_writer_impl.hpp"

template <
    uint32_t num_output_channels_padded,
    uint32_t num_full_tiles,
    uint32_t total_tiles_per_block,
    uint32_t initial_write_stick_offset,
    uint32_t element_size_bytes,
    uint32_t input_num_blocks,
    uint32_t l1_write_output_addr_stride>
TT_KERNEL void secondary_writer() {
    Noc noc;
    DataflowBuffer transpose(dfb::transpose);
    DataflowBuffer output(dfb::output);
    uint32_t l1_output_write_addr =
        output.get_write_ptr() + initial_write_stick_offset * num_output_channels_padded * element_size_bytes;

    for (uint32_t block_id = 0; block_id < input_num_blocks; ++block_id) {
        convert_to_hwc::write_transposed_block<
            num_output_channels_padded,
            num_full_tiles,
            total_tiles_per_block,
            false,
            element_size_bytes,
            l1_write_output_addr_stride>(noc, transpose, l1_output_write_addr);
    }
}
