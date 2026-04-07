// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 W fabric reader for fused 2D neighbor pad.
// Reads W boundary sticks directly from the output DRAM tensor (written by Phase 1)
// instead of from an L1 boundary buffer. Phase 1 cores call noc_async_write_barrier()
// before signaling the Phase 2 barrier semaphore, guaranteeing DRAM writes are committed.
//
// DRAM writes are handled by the paired writer (minimal_default_writer).
//
// fabric_only mode: in this mode the output tensor is the compact halo buffer. The W reader
// must gather W-boundary sticks from three sources per T-slice:
//   - H-top halo rows  (h_ext < padding_h):     read from compact halo buffer H-top section
//   - Interior rows    (h_ext in [ph, ph+H_dev)): read from the input tensor directly
//   - H-bot halo rows  (h_ext >= ph + H_dev):    read from compact halo buffer H-bot section
// The gathered sticks feed the W writer, which sends them to the neighbor and writes the
// extended W halo section of the compact buffer.

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>

using address_t = uint32_t;

// Compile-time args (uniform across all W reader cores)
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Output (compact halo) buffer TensorAccessorArgs start at index 3 (variable length)
constexpr auto dst_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_dst = dst_args.next_compile_time_args_offset();
// Input tensor TensorAccessorArgs start right after dst_args (used in fabric_only mode)
constexpr auto src_args = TensorAccessorArgs<ct_after_dst>();
// NP_W_HALO_L1: recv_cb_id comes after src_args (only when NP_W_HALO_L1 is defined).
// The W writer on the neighbor chip delivers W-halo sticks to our L1 recv_cb instead of DRAM,
// guaranteeing NOC-ordering within L1 (fabric data write → sem_inc, both to L1 = ordered).
#if defined(NP_W_HALO_L1)
constexpr uint32_t ct_after_src = src_args.next_compile_time_args_offset();
constexpr uint32_t recv_cb_id = get_compile_time_arg_val(ct_after_src);
#endif

template <uint32_t stick_size_bytes>
inline void zeroPad(uint32_t cb_id) {
    constexpr uint32_t num_full_reads = stick_size_bytes / MEM_ZEROS_SIZE;
    constexpr uint32_t partial_read_size = stick_size_bytes % MEM_ZEROS_SIZE;
    const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t cb_write_addr = get_write_ptr(cb_id);

    for (uint32_t i = 0; i < num_full_reads; ++i) {
        noc_async_read(zeros_noc_addr, cb_write_addr, MEM_ZEROS_SIZE);
        cb_write_addr += MEM_ZEROS_SIZE;
    }
    if (partial_read_size > 0) {
        noc_async_read(zeros_noc_addr, cb_write_addr, partial_read_size);
    }
}

void kernel_main() {
    // Common runtime args (uniform across all cores, updated between dispatches)
    const address_t output_tensor_address = get_common_arg_val<address_t>(0);
    const uint32_t barrier_sem_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_neighbor_sem_addr = get_common_arg_val<uint32_t>(2);
#if defined(NP_PROGRESS_SEM)
    // [3] progress_sem_addr: GlobalSemaphore address on conv3d reader cores.
    //     W-reader signals this after w_nbr_sem wait (guarantees W-halo from neighbor
    //     is in compact buffer DRAM). All NP_NUM_W_WRITERS W-readers signal once each.
    // [4] num_reader_cores: number of conv3d reader cores to signal.
    // [5+]: NOC coords of conv3d reader cores.
    const uint32_t progress_sem_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t num_reader_cores = get_common_arg_val<uint32_t>(4);
#endif

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_row_width = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pad2_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_interior_sticks = get_arg_val<uint32_t>(arg_idx++);
    // Per-core direction args (moved from compile-time for kernel consolidation)
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    // fabric_only extension args: non-zero only in fabric_only 2D mode.
    // input_buffer_addr: address of the unpadded input tensor (for interior row reads).
    // h_dev: number of interior H rows in the input tensor (= input_halo_dim_size).
    // h_padding: H padding (ph); top halo rows = [0, h_padding), bot = [h_padding+h_dev, H_total).
    // h_halo_hbot_base: first page of H-bot section in the compact halo buffer
    //                   = outer_dim_size * h_padding * num_interior_sticks.
    const address_t input_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_dev = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_halo_hbot_base = get_arg_val<uint32_t>(arg_idx++);
#if defined(NP_W_HALO_L1)
    // [14] recv_dram_offset: base DRAM page in the compact halo buffer for received W-halo sticks.
    // Equals the sender chip's outer_dim_offset_start_id (same value on all chips, uniform MeshBuffer).
    // Only valid when NP_W_HALO_L1 is defined (fabric_only mode).
    const uint32_t recv_dram_offset = get_arg_val<uint32_t>(arg_idx++);
#endif
#if defined(NP_PROGRESS_SEM)
    const uint32_t progress_t_batch_size = get_arg_val<uint32_t>(arg_idx++);
#endif

    const bool is_fabric_only = (input_buffer_addr != 0);

    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);
    const auto src_accessor = TensorAccessor(src_args, input_buffer_addr, stick_size);

    const uint32_t h_total = h_dev + 2u * h_padding;

    // Batch sizing: when per-T-batch pipelining is active, process rows in T-batch-sized
    // chunks synchronized with the H writer.  Otherwise process all rows in one shot.
#if defined(NP_PROGRESS_SEM)
    const uint32_t rows_per_batch = (progress_t_batch_size > 0) ? progress_t_batch_size * h_total : outer_dim_size;
#else
    const uint32_t rows_per_batch = outer_dim_size;
#endif
    const uint32_t num_full_batches = outer_dim_size / rows_per_batch;
    const uint32_t remainder = outer_dim_size - num_full_batches * rows_per_batch;
    const uint32_t total_batches = num_full_batches + (remainder > 0 ? 1u : 0u);

    volatile tt_l1_ptr uint32_t* barrier_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr);
    volatile tt_l1_ptr uint32_t* w_neighbor_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr);

    uint32_t od_start = 0;
    for (uint32_t batch = 0; batch < total_batches; batch++) {
        const uint32_t batch_rows = (batch < num_full_batches) ? rows_per_batch : remainder;

        // Wait for H writer to commit this T-batch's H-halo (cumulative, no reset).
        if (barrier_count > 0) {
            noc_semaphore_wait_min(barrier_sem_ptr, (batch + 1) * barrier_count);
        }

        // Main loop: read boundary sticks from source buffer(s) → CB for the paired writer.
        for (uint32_t outer_dim = od_start; outer_dim < od_start + batch_rows; outer_dim++) {
            if (is_first_chip) {
                if (!is_padding_zeros) {
                    cb_reserve_back(cb_output_id, 1);
                    uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                    if (is_fabric_only) {
                        uint32_t global_row = outer_dim_start + outer_dim;
                        uint32_t t = global_row / h_total;
                        uint32_t h_ext = global_row % h_total;
                        uint32_t w_col = direction ? (num_interior_sticks - 1u) : 0u;
                        uint32_t page;
                        if (h_ext < h_padding) {
                            page = t * h_padding * num_interior_sticks + h_ext * num_interior_sticks + w_col;
                            noc_async_read(get_noc_addr(page, dst_accessor), dst_l1_addr, stick_size);
                        } else if (h_ext < h_padding + h_dev) {
                            uint32_t h_local = h_ext - h_padding;
                            page = t * h_dev * num_interior_sticks + h_local * num_interior_sticks + w_col;
                            noc_async_read(get_noc_addr(page, src_accessor), dst_l1_addr, stick_size);
                        } else {
                            uint32_t h_bot = h_ext - h_padding - h_dev;
                            page = h_halo_hbot_base + t * h_padding * num_interior_sticks +
                                   h_bot * num_interior_sticks + w_col;
                            noc_async_read(get_noc_addr(page, dst_accessor), dst_l1_addr, stick_size);
                        }
                    } else {
                        uint32_t row_base = (outer_dim_start + outer_dim) * output_row_width;
                        uint32_t col;
                        if (direction) {
                            col = pad2_left + num_interior_sticks - 1;
                        } else {
                            col = pad2_left;
                        }
                        noc_async_read(get_noc_addr(row_base + col, dst_accessor), dst_l1_addr, stick_size);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, 1);
                } else {
                    cb_reserve_back(cb_output_id, 1);
                    zeroPad<stick_size>(cb_output_id);
                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, 1);
                }
            }

            if (!is_last_chip) {
                for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                    cb_reserve_back(cb_output_id, 1);
                    uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                    if (is_fabric_only) {
                        uint32_t global_row = outer_dim_start + outer_dim;
                        uint32_t t = global_row / h_total;
                        uint32_t h_ext = global_row % h_total;
                        uint32_t w_col;
                        if (direction) {
                            w_col = padding - pad_id;
                        } else {
                            w_col = num_interior_sticks - pad_id;
                        }
                        uint32_t page;
                        if (h_ext < h_padding) {
                            page = t * h_padding * num_interior_sticks + h_ext * num_interior_sticks + w_col;
                            noc_async_read(get_noc_addr(page, dst_accessor), dst_l1_addr, stick_size);
                        } else if (h_ext < h_padding + h_dev) {
                            uint32_t h_local = h_ext - h_padding;
                            page = t * h_dev * num_interior_sticks + h_local * num_interior_sticks + w_col;
                            noc_async_read(get_noc_addr(page, src_accessor), dst_l1_addr, stick_size);
                        } else {
                            uint32_t h_bot = h_ext - h_padding - h_dev;
                            page = h_halo_hbot_base + t * h_padding * num_interior_sticks +
                                   h_bot * num_interior_sticks + w_col;
                            noc_async_read(get_noc_addr(page, dst_accessor), dst_l1_addr, stick_size);
                        }
                    } else {
                        uint32_t row_base = (outer_dim_start + outer_dim) * output_row_width;
                        uint32_t col;
                        if (direction) {
                            col = pad2_left + (padding - pad_id);
                        } else {
                            col = pad2_left + num_interior_sticks - pad_id;
                        }
                        noc_async_read(get_noc_addr(row_base + col, dst_accessor), dst_l1_addr, stick_size);
                    }
                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, 1);
                }
            }
        }

        // Wait for neighbor's W-halo sticks for this batch (cumulative).
        if (!is_first_chip) {
            noc_semaphore_wait_min(w_neighbor_sem_ptr, od_start + batch_rows);
#if defined(NP_W_HALO_L1)
            {
                const uint32_t recv_buf_base = get_write_ptr(recv_cb_id);
                uint32_t dram_page = recv_dram_offset + od_start;
                for (uint32_t od = 0; od < batch_rows; od++) {
                    const uint32_t l1_read_addr = recv_buf_base + (od_start + od) * stick_size;
                    noc_async_write(l1_read_addr, get_noc_addr(dram_page, dst_accessor), stick_size);
                    dram_page++;
                }
            }
#endif
        }

#if defined(NP_PROGRESS_SEM)
        // Signal conv3d readers that this T-batch's H-halo + W-halo are committed.
        // Only dir=0 fires to match the single per-T-batch increment conv3d expects.
        if (progress_t_batch_size > 0 && !direction) {
            noc_async_write_barrier();
            for (uint32_t i = 0; i < num_reader_cores; i++) {
                const uint32_t rx = get_common_arg_val<uint32_t>(5 + i * 2);
                const uint32_t ry = get_common_arg_val<uint32_t>(5 + i * 2 + 1);
                noc_semaphore_inc(get_noc_addr(rx, ry, progress_sem_addr), 1);
            }
            noc_async_atomic_barrier();
        }
#endif

        od_start += batch_rows;
    }

    // Reset semaphores after all batches.
    noc_semaphore_set(barrier_sem_ptr, 0);
    if (!is_first_chip) {
        noc_semaphore_set(w_neighbor_sem_ptr, 0);
    }

#if defined(NP_PROGRESS_SEM)
    // One-shot signal for non-per-T-batch paths (standalone NP with NP_PROGRESS_SEM).
    if (progress_t_batch_size == 0) {
        noc_async_write_barrier();
        for (uint32_t i = 0; i < num_reader_cores; i++) {
            const uint32_t rx = get_common_arg_val<uint32_t>(5 + i * 2);
            const uint32_t ry = get_common_arg_val<uint32_t>(5 + i * 2 + 1);
            noc_semaphore_inc(get_noc_addr(rx, ry, progress_sem_addr), 1);
        }
        noc_async_atomic_barrier();
    }
#endif
}
