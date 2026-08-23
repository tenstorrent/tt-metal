// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "padded_slice_reader_rm_impl.hpp"

template <uint32_t src_buffer_alignment, uint32_t num_trids>
TT_KERNEL void reader(
    uint32_t src_byte_offset,
    uint32_t padded_stick_size,
    uint32_t unpadded_stick_size,
    uint32_t stick_size_offset,
    uint32_t num_dims,
    uint32_t start_id,
    uint32_t num_sticks_per_core,
    uint32_t num_sticks_per_core_read,
    uint32_t num_read_per_barrier) {
    padded_slice::read_rm_non_aligned<src_buffer_alignment, num_trids>(
        src_byte_offset,
        padded_stick_size,
        unpadded_stick_size,
        stick_size_offset,
        num_dims,
        start_id,
        num_sticks_per_core,
        num_sticks_per_core_read,
        num_read_per_barrier,
        scratch::alignment);
}
