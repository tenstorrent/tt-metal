// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>

using address_t = uint32_t;

constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr uint32_t stick_size = get_compile_time_arg_val(1);
// TensorAccessorArgs at index 2 (variable length)
constexpr auto dst_args = TensorAccessorArgs<2>();

template <uint32_t stick_size_bytes>
inline void zeroWrite(uint64_t dst_noc_addr) {
    constexpr uint32_t num_full_writes = stick_size_bytes / MEM_ZEROS_SIZE;
    constexpr uint32_t partial_write_size = stick_size_bytes % MEM_ZEROS_SIZE;
    for (uint32_t i = 0; i < num_full_writes; ++i) {
        noc_async_write((uint32_t)MEM_ZEROS_BASE, dst_noc_addr, MEM_ZEROS_SIZE);
        dst_noc_addr += MEM_ZEROS_SIZE;
    }
    if constexpr (partial_write_size > 0) {
        noc_async_write((uint32_t)MEM_ZEROS_BASE, dst_noc_addr, partial_write_size);
    }
}

void kernel_main() {
    // Common runtime args (multicast once per kernel, not unicast per core)
    // CRTA[0] = input_addr (unused by writer, reserved for consistency with reader)
    const address_t output_tensor_address = get_common_arg_val<address_t>(1);
    const uint32_t stick_start_id = get_common_arg_val<uint32_t>(2);
    const uint32_t input_halo_dim_size = get_common_arg_val<uint32_t>(3);
    const uint32_t output_halo_dim_size = get_common_arg_val<uint32_t>(4);
    const uint32_t padding_left = get_common_arg_val<uint32_t>(5);
    const uint32_t num_sticks_to_read = get_common_arg_val<uint32_t>(6);
    const uint32_t num_sticks_per_halo_dim = get_common_arg_val<uint32_t>(7);
    const uint32_t logical_h = get_common_arg_val<uint32_t>(8);
    const uint32_t device_h_offset = get_common_arg_val<uint32_t>(9);
    const uint32_t t_front_pad_stick_offset = get_common_arg_val<uint32_t>(10);
    const bool do_masking = (logical_h > 0);

    // Per-core runtime args (only work distribution — truly unique per core)
    uint32_t arg_idx = 0;
    const uint32_t total_rows_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rows_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t zero_fill_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t zero_fill_count = get_arg_val<uint32_t>(arg_idx++);

    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address);

    // Phase A: zero-fill this core's slice of the T-front output region.
    // Covers all H positions (interior + H-halo) at T < t_front_pad.
    for (uint32_t s = 0; s < zero_fill_count; ++s) {
        uint64_t dst_noc_addr = dst_accessor.get_noc_addr(zero_fill_start + s);
        zeroWrite<stick_size>(dst_noc_addr);
        noc_async_write_barrier();
    }

    // Phase B: copy input sticks (from CB) to output at T-offset t_front_pad_stick_offset.
    for (uint32_t s = 0; s < rows_count; s++) {
        const uint32_t linear_row = total_rows_start + s;  // [0 .. outer_dim_size*input_halo_dim_size)
        const uint32_t outer_idx = linear_row / input_halo_dim_size;
        const uint32_t t = linear_row % input_halo_dim_size;
        const uint32_t outer_dim_offset = outer_idx * (num_sticks_per_halo_dim * output_halo_dim_size);
        const bool masked = do_masking && (device_h_offset + t >= logical_h);

        uint32_t dst_stick_id =
            (t + padding_left) * num_sticks_per_halo_dim + stick_start_id + outer_dim_offset + t_front_pad_stick_offset;
        for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
            cb_wait_front(cb_output_id, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_output_id);
            uint64_t dst_noc_addr = dst_accessor.get_noc_addr(dst_stick_id);
            if (masked) {
                zeroWrite<stick_size>(dst_noc_addr);
            } else {
                noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            }

            dst_stick_id++;
            noc_async_write_barrier();
            cb_pop_front(cb_output_id, 1);
        }
    }
    noc_async_write_barrier();
}
