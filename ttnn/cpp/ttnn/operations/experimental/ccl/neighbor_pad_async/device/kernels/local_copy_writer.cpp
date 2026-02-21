// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>
//
using address_t = uint32_t;
//
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr uint32_t stick_size = get_compile_time_arg_val(1);
// TensorAccessorArgs at index 2 (variable length)
constexpr auto dst_args = TensorAccessorArgs<2>();
constexpr uint32_t ct_after_dst = dst_args.next_compile_time_args_offset();
constexpr bool use_boundary_buf = get_compile_time_arg_val(ct_after_dst);
constexpr uint32_t w_padding = get_compile_time_arg_val(ct_after_dst + 1);
constexpr uint32_t boundary_sticks_per_row = 2 * w_padding;
//
void kernel_main() {
    DeviceZoneScopedN("NPAD-WRITER");
    // Args
    uint32_t arg_idx = 0;
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);  // not used in writer
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t total_rows_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t rows_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);
    // Phase 2 barrier signal targets (0 for 1D, >0 for 2D)
    const uint32_t num_phase2_signal_targets = get_arg_val<uint32_t>(arg_idx++);
    uint8_t signal_noc_x[2];
    uint8_t signal_noc_y[2];
    uint32_t signal_sem_addr[2];
    for (uint32_t t = 0; t < 2; t++) {
        signal_noc_x[t] = get_arg_val<uint32_t>(arg_idx++);
        signal_noc_y[t] = get_arg_val<uint32_t>(arg_idx++);
        signal_sem_addr[t] = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    }
    const uint32_t w_boundary_buf_addr = get_arg_val<uint32_t>(arg_idx++);
    //
    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);

    //
    for (uint32_t s = 0; s < rows_count; s++) {
        const uint32_t linear_row = total_rows_start + s;  // [0 .. outer_dim_size*input_halo_dim_size)
        const uint32_t outer_idx = linear_row / input_halo_dim_size;
        const uint32_t t = linear_row % input_halo_dim_size;
        const uint32_t outer_dim_offset = outer_idx * (num_sticks_per_halo_dim * output_halo_dim_size);

        // Boundary buffer row index: global output row = outer_idx * output_halo_dim_size + (t + padding_left)
        uint32_t boundary_buf_base = 0;
        if constexpr (use_boundary_buf && w_padding > 0) {
            uint32_t boundary_row = outer_idx * output_halo_dim_size + (t + padding_left);
            boundary_buf_base = boundary_row * boundary_sticks_per_row * stick_size;
        }

        uint32_t dst_stick_id = (t + padding_left) * num_sticks_per_halo_dim + stick_start_id + outer_dim_offset;
        { DeviceZoneScopedN("NPAD-WR-STICKS");
        for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
            cb_wait_front(cb_output_id, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_output_id);
            uint64_t dst_noc_addr = get_noc_addr(dst_stick_id, dst_accessor);
            noc_async_write(l1_read_addr, dst_noc_addr, stick_size);

            // Write W boundary sticks to Phase 2 W reader cores' L1 boundary buffer
            if constexpr (use_boundary_buf && w_padding > 0) {
                if (iter < w_padding) {
                    // Left boundary: slot left[iter]
                    uint32_t buf_offset = boundary_buf_base + iter * stick_size;
                    for (uint32_t wt = 0; wt < num_phase2_signal_targets; wt++) {
                        uint64_t addr =
                            get_noc_addr(signal_noc_x[wt], signal_noc_y[wt], w_boundary_buf_addr + buf_offset);
                        noc_async_write(l1_read_addr, addr, stick_size);
                    }
                }
                if (iter >= num_sticks_to_read - w_padding) {
                    // Right boundary: slot right[iter - (num_sticks_to_read - w_padding)]
                    uint32_t right_idx = iter - (num_sticks_to_read - w_padding);
                    uint32_t buf_offset = boundary_buf_base + (w_padding + right_idx) * stick_size;
                    for (uint32_t wt = 0; wt < num_phase2_signal_targets; wt++) {
                        uint64_t addr =
                            get_noc_addr(signal_noc_x[wt], signal_noc_y[wt], w_boundary_buf_addr + buf_offset);
                        noc_async_write(l1_read_addr, addr, stick_size);
                    }
                }
            }

            dst_stick_id++;
            noc_async_write_barrier();
            cb_pop_front(cb_output_id, 1);
        }
        }
    }
    noc_async_write_barrier();

    // Signal Phase 2 W fabric reader cores that Phase 1 writes are complete
    for (uint32_t t = 0; t < num_phase2_signal_targets; t++) {
        uint64_t sem_noc_addr = get_noc_addr(signal_noc_x[t], signal_noc_y[t], signal_sem_addr[t]);
        noc_semaphore_inc(sem_noc_addr, 1);
    }
    // Ensure sem inc signals are delivered before kernel exits.
    noc_async_write_barrier();
}
