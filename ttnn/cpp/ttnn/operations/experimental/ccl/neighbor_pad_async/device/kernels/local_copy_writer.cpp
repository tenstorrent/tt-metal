// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

void kernel_main() {
    // Common runtime args (multicast once per kernel, not unicast per core)
    // CRTA[0] = input_addr (unused by writer, reserved for consistency with reader)
    const address_t output_tensor_address = get_common_arg_val<address_t>(1);
    const uint32_t phase2_barrier_sem = get_common_arg_val<uint32_t>(2);
    const uint32_t stick_start_id = get_common_arg_val<uint32_t>(3);
    const uint32_t input_halo_dim_size = get_common_arg_val<uint32_t>(4);
    const uint32_t output_halo_dim_size = get_common_arg_val<uint32_t>(5);
    const uint32_t padding_left = get_common_arg_val<uint32_t>(6);
    const uint32_t num_sticks_to_read = get_common_arg_val<uint32_t>(7);
    const uint32_t num_sticks_per_halo_dim = get_common_arg_val<uint32_t>(8);
    // Phase 2 barrier signal targets (uniform across all local-copy cores)
    constexpr uint32_t MAX_PHASE2_SIGNAL_TARGETS = 8;
    const uint32_t num_phase2_signal_targets = get_common_arg_val<uint32_t>(9);
    uint8_t signal_noc_x[MAX_PHASE2_SIGNAL_TARGETS];
    uint8_t signal_noc_y[MAX_PHASE2_SIGNAL_TARGETS];
    for (uint32_t t = 0; t < MAX_PHASE2_SIGNAL_TARGETS; t++) {
        signal_noc_x[t] = get_common_arg_val<uint32_t>(10 + t * 2);
        signal_noc_y[t] = get_common_arg_val<uint32_t>(10 + t * 2 + 1);
    }

    // Per-core runtime args (only work distribution — truly unique per core)
    uint32_t arg_idx = 0;
    const uint32_t total_rows_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rows_count = get_arg_val<uint32_t>(arg_idx++);

    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);

    for (uint32_t s = 0; s < rows_count; s++) {
        const uint32_t linear_row = total_rows_start + s;  // [0 .. outer_dim_size*input_halo_dim_size)
        const uint32_t outer_idx = linear_row / input_halo_dim_size;
        const uint32_t t = linear_row % input_halo_dim_size;
        const uint32_t outer_dim_offset = outer_idx * (num_sticks_per_halo_dim * output_halo_dim_size);

        uint32_t dst_stick_id = (t + padding_left) * num_sticks_per_halo_dim + stick_start_id + outer_dim_offset;
        for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
            cb_wait_front(cb_output_id, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_output_id);
            uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size);
            dst_stick_id++;
            noc_async_write_barrier();
            cb_pop_front(cb_output_id, 1);
        }
    }
    noc_async_write_barrier();

    // Signal Phase 2 W fabric reader cores that Phase 1 writes are complete.
    // Guard with rows_count > 0: cores with no work assigned (skipped during work distribution)
    // must not signal, since barrier_count only counts active cores.
    if (rows_count > 0) {
        for (uint32_t t = 0; t < num_phase2_signal_targets; t++) {
            uint64_t sem_noc_addr = get_noc_addr(signal_noc_x[t], signal_noc_y[t], phase2_barrier_sem);
            noc_semaphore_inc(sem_noc_addr, 1);
        }
        // Ensure sem inc signals are delivered before kernel exits.
        noc_async_write_barrier();
    }
    noc_async_atomic_barrier();
}
