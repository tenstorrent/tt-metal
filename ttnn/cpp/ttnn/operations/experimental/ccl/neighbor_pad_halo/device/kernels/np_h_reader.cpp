// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/neighbor_pad_halo/device/kernels/np_zero_pad.hpp"
#include <cstdint>

using address_t = uint32_t;

// Compile-time args (uniform across all H fabric reader cores)
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Input TensorAccessorArgs at index 3 (variable length)
constexpr auto src_ct_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_src = src_ct_args.next_compile_time_args_offset();
// L1 intermediate config
constexpr bool use_l1_intermediate = get_compile_time_arg_val(ct_after_src);
constexpr uint32_t recv_cb_id = get_compile_time_arg_val(ct_after_src + 1);
// Dedicated send CB for the batched H-halo send (separate ring from cb_output_id so the
// per-stick recv/is_first pushes don't desync the row reserves).
constexpr uint32_t send_cb_id = get_compile_time_arg_val(ct_after_src + 2);
// H-send bank-major coalesce factor (0 = row-contiguous). When > 0, gather the H-halo row into send_cb
// in dst-bank-major order (w=0,8,..; 1,9,..) so the writer ships same-bank sticks as one 4KB packet.
constexpr uint32_t H_COALESCE = get_compile_time_arg_val(ct_after_src + 3);
// When set (H-mux path), this reader owns the H->W barrier: it signals the W-reader cores only AFTER its
// incoming H has landed, so the W corner reads (which read this device's H-section) can't race the H
// exchange. On the direct path (0) np_writer signals the barrier instead.
constexpr uint32_t H_SIGNAL_W_RECV = get_compile_time_arg_val(ct_after_src + 4);
constexpr uint32_t NP_NUM_DRAM_BANKS = 8;
// Max W reader cores this H reader signals on the H->W barrier: pad2_num_links * 2 * num_w_workers
// under W-mux (MAX_PAD2_NUM_LINKS 4 * 2 * 4 workers = 32). Must match MAX_W_BARRIER_TARGETS in the factory.
constexpr uint32_t MAX_W_BAR_TARGETS = 32;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Common runtime args (uniform across all cores, updated between dispatches). Index 1 (output addr)
    // is part of the shared CRTA layout but unused by this reader.
    const address_t input_tensor_address = get_common_arg_val<address_t>(0);
    const size_t h_neighbor_sem = get_common_arg_val<uint32_t>(2);

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_offset_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);
    // Number of corner sticks per pad row in L1 recv buffer (= pad2_left + pad2_right for 2D corners-only)
    const uint32_t num_l1_recv_sticks_per_row = get_arg_val<uint32_t>(arg_idx++);
    // Rows per input frame for the per-frame stride (num_sticks_per_halo_dim * input_frame_rows). Equals
    // input_halo_dim_size for a contiguous input; equals the PADDED H for a padded-input (strided) read,
    // so the frame advance skips the padded border rows while the edge formula still uses the interior H.
    const uint32_t input_frame_rows = get_arg_val<uint32_t>(arg_idx++);
    // Per-core direction args passed at runtime (not compile-time) so one kernel binary serves every core
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    // H->W barrier targets (H-mux path only): W-reader cores this reader signals once its recv is done.
    uint32_t num_w_bar = 0;
    uint8_t w_bar_x[MAX_W_BAR_TARGETS];
    uint8_t w_bar_y[MAX_W_BAR_TARGETS];
    if constexpr (H_SIGNAL_W_RECV) {
        num_w_bar = get_arg_val<uint32_t>(arg_idx++);
        for (uint32_t t = 0; t < MAX_W_BAR_TARGETS; t++) {
            w_bar_x[t] = get_arg_val<uint32_t>(arg_idx++);
            w_bar_y[t] = get_arg_val<uint32_t>(arg_idx++);
        }
    }

    uint32_t read_size = stick_size;
    const auto src_accessor = TensorAccessor(src_ct_args, input_tensor_address);

    uint32_t outer_dim_offset = outer_dim_offset_start_id;
    uint32_t recv_buf_addr = 0;
    uint32_t buf_offset = 0;  // recv: accumulates across all outer_dims (no L1 reuse)
    if constexpr (use_l1_intermediate) {
        if (!is_first_chip) {
            recv_buf_addr = get_write_ptr(recv_cb_id);
        }
    }
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        if (is_first_chip) {
            if (!is_padding_zeros) {
                // Replicate a slice of 1 from input to output
                uint32_t src_stick_id = 0;
                if (direction) {
                    src_stick_id = num_sticks_per_halo_dim * (input_halo_dim_size - 1) + stick_start_id;
                } else {
                    src_stick_id = stick_start_id;
                }
                src_stick_id += outer_dim_offset;
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    cb_reserve_back(cb_output_id, 1);
                    uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);

                    uint64_t src_noc_addr = get_noc_addr(src_stick_id, src_accessor);
                    noc_async_read(src_noc_addr, src_buffer_l1_addr, read_size);

                    src_stick_id++;

                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, 1);
                }
            } else {
                cb_reserve_back(cb_output_id, 1);
                zeroPadCb<stick_size>(cb_output_id);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }

        if (!is_last_chip) {
            // Read the "end" of each slice into the dedicated send CB. Batch the whole row: one
            // cb_reserve + one barrier for all num_sticks_to_read sticks instead of per stick. The
            // per-stick path issued ~18k read+barrier pairs and was latency-bound; coalescing the
            // row makes it bandwidth-bound. send_cb_id holds exactly 2 rows so a row reserve never
            // wraps mid-batch, and the paired writer drains it one stick at a time.
            for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                uint32_t src_stick_id = 0;
                if (direction) {
                    src_stick_id = (padding - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                } else {
                    src_stick_id = (input_halo_dim_size - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                }
                src_stick_id += outer_dim_offset;
                cb_reserve_back(send_cb_id, num_sticks_to_read);
                uint32_t row_base_l1_addr = get_write_ptr(send_cb_id);
                if constexpr (H_COALESCE > 0) {
                    // Bank-major gather: send_cb[m] = column w in dst-bank order (0,8,..; 1,9,..) so the
                    // writer coalesces same-bank sticks. src is scattered; the batch shares one barrier.
                    uint32_t m = 0;
                    for (uint32_t j = 0; j < NP_NUM_DRAM_BANKS; j++) {
                        for (uint32_t w = j; w < num_sticks_to_read; w += NP_NUM_DRAM_BANKS) {
                            noc_async_read(
                                get_noc_addr(src_stick_id + w, src_accessor),
                                row_base_l1_addr + m * stick_size,
                                read_size);
                            m++;
                        }
                    }
                } else {
                    for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                        uint64_t src_noc_addr = get_noc_addr(src_stick_id + iter, src_accessor);
                        noc_async_read(src_noc_addr, row_base_l1_addr + iter * stick_size, read_size);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(send_cb_id, num_sticks_to_read);
            }
        }

        // No local interior copy in this kernel. Dedicated local-copy kernels handle that work.

        outer_dim_offset += (num_sticks_per_halo_dim * input_frame_rows);

        // Per-batch H-commit: pull this od's incoming H-halo from the L1 recv buffer into the CB as
        // soon as its sender link delivers it (per-core cumulative count, 1 inc/outer_dim), so the
        // paired writer commits + signals HT/HB this batch rather than after the whole send pass.
        if constexpr (use_l1_intermediate) {
            if (!is_first_chip) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), outer_dim + 1);
                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    // num_l1_recv_sticks_per_row (corners-only) not num_sticks_to_read (full row).
                    for (uint32_t iter = 0; iter < num_l1_recv_sticks_per_row; iter++) {
                        cb_reserve_back(cb_output_id, 1);
                        uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(get_noc_addr(recv_buf_addr + buf_offset), dst_l1_addr, stick_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id, 1);
                        buf_offset += stick_size;
                    }
                }
            }
        }
    }

    // Drain the per-core fabric-arrival sem. 2D recvs were interleaved above; 1D wrote straight to
    // DRAM so just wait for all outer_dims here, then reset.
    if (!is_first_chip) {
        if constexpr (use_l1_intermediate) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), 0);
        } else {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), outer_dim_size);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), 0);
        }
    }

    // H-mux path: the incoming H has now landed (recv drained above). Signal the H->W barrier on each
    // W-reader core so the W corner reads (which read this device's H-section) see complete H data. Done
    // here (recv-authority) not in the writer (send-done) so it holds for >2 H-axes and small shapes.
    if constexpr (H_SIGNAL_W_RECV) {
        const uint32_t barrier_sem = get_common_arg_val<uint32_t>(3);
        for (uint32_t t = 0; t < num_w_bar; t++) {
            noc_semaphore_inc(safe_get_noc_addr(w_bar_x[t], w_bar_y[t], barrier_sem, 0), 1);
        }
        noc_async_atomic_barrier();
    }
}
