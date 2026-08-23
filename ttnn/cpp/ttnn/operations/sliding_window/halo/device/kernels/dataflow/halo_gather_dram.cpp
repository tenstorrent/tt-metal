// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "halo_gather_impl.hpp"

template <
    uint32_t pad_val,
    uint32_t input_npages,
    uint32_t skip_untilize,
    uint32_t aligned_stick_nbytes,
    uint32_t is_block_sharded,
    uint32_t is_col_major,
    uint32_t is_width_sharded,
    uint32_t block_size_height,
    uint32_t block_size_width_tiles,
    uint32_t block_start_offset,
    uint32_t block_stride,
    uint32_t config_tensor_in_dram,
    uint32_t enable_padding,
    uint32_t use_pad_scratch>
TT_KERNEL void halo_gather(uint32_t config_read_index) {
    halo::gather<
        pad_val,
        input_npages,
        skip_untilize,
        aligned_stick_nbytes,
        is_block_sharded,
        is_col_major,
        is_width_sharded,
        block_size_height,
        block_size_width_tiles,
        block_start_offset,
        block_stride,
        config_tensor_in_dram,
        enable_padding,
        use_pad_scratch>(config_read_index);
}
