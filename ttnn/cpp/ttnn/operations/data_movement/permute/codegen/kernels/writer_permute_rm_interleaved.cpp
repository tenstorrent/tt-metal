// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BRISC writer for RM row-invariant permute (W dimension unchanged).
// Computes permuted output address for each input row, writes with
// pipelined batched NOC barriers.
//
// CT args: cb_id, page_size, TensorAccessorArgs(out_t)..., BATCH, N
// RT args: dst_addr, num_rows, start_row, input_shape[N], perm[N], dest_strides[N]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "rm_shard_split.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(1);
    uint32_t start_row = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t requested_write_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    constexpr uint32_t N = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 1);
    // ROW_MAJOR width/block-sharded destination: a page is one shard-width
    // slice, so `page_id = dest_linear_idx` names the wrong bytes. Host emits
    // (0, 0) otherwise and the header collapses to the historical write.
    constexpr uint32_t DST_PAGES_PER_ROW = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t DST_LOGICAL_PAGE_SIZE = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 3);

    const uint32_t destination_page_size = dst_args.get_aligned_page_size();
    const auto d = TensorAccessor(dst_args, dst_addr, destination_page_size);
    const uint32_t l1_page_stride = get_local_cb_interface(cb_id).fifo_page_size << cb_addr_shift;
    // When the destination is width/block-sharded, `destination_page_size` is
    // ONE SHARD's page, not the row -- clamping to it would truncate every row
    // to its first shard. The logical row is PAGES_PER_ROW such pages, and the
    // header splits the write across them. Constexpr, so this folds away.
    const uint32_t row_bytes_dst =
        (DST_PAGES_PER_ROW > 1) ? (DST_PAGES_PER_ROW * DST_LOGICAL_PAGE_SIZE) : destination_page_size;
    const uint32_t write_size_dst = requested_write_size < row_bytes_dst ? requested_write_size : row_bytes_dst;
    const uint32_t write_size = write_size_dst < l1_page_stride ? write_size_dst : l1_page_stride;

    Noc noc;
    CircularBuffer cb_out(cb_id);

    // Read input_shape, perm, dest_strides from runtime args (starting at index 3)
    uint32_t input_shape[N], perm[N], dest_strides[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_arg_val<uint32_t>(3 + i);
        perm[i] = get_arg_val<uint32_t>(3 + N + i);
        dest_strides[i] = get_arg_val<uint32_t>(3 + 2 * N + i);
    }

    uint32_t row = start_row;
    uint32_t rows_left = num_rows;

    if constexpr (BATCH > 1) {
        // Pipelined batched writer: overlap NOC DMA with compute delivery.
        // Prime the pipeline
        uint32_t batch = (rows_left < BATCH) ? rows_left : BATCH;
        cb_out.wait_front(batch);
        uint32_t l1_offset = 0;

        for (uint32_t t = 0; t < batch; t++) {
            // Decompose row index into multi-dim source index
            uint32_t src_multi_idx[N];
            uint32_t remaining = row;
            for (uint32_t i = 0; i < N - 1; ++i) {
                uint32_t dim = N - 2 - i;
                src_multi_idx[dim] = remaining % input_shape[dim];
                remaining /= input_shape[dim];
            }
            src_multi_idx[N - 1] = 0;  // W dimension (row-invariant)

            // Apply permutation
            uint32_t dest_linear_idx = 0;
            for (uint32_t i = 0; i < N - 1; ++i) {
                dest_linear_idx += src_multi_idx[perm[i]] * dest_strides[i];
            }

            ttdm::noc_write_row_split<DST_PAGES_PER_ROW, DST_LOGICAL_PAGE_SIZE>(
                noc, cb_out, l1_offset, d, dest_linear_idx, /*dst_offset=*/0, write_size);
            l1_offset += l1_page_stride;
            row++;
        }
        rows_left -= batch;
        uint32_t prev_batch = batch;

        // Steady state
        while (rows_left > 0) {
            batch = (rows_left < BATCH) ? rows_left : BATCH;
            cb_out.wait_front(prev_batch + batch);
            noc.async_writes_flushed();
            cb_out.pop_front(prev_batch);

            l1_offset = 0;
            for (uint32_t t = 0; t < batch; t++) {
                uint32_t src_multi_idx[N];
                uint32_t remaining = row;
                for (uint32_t i = 0; i < N - 1; ++i) {
                    uint32_t dim = N - 2 - i;
                    src_multi_idx[dim] = remaining % input_shape[dim];
                    remaining /= input_shape[dim];
                }
                src_multi_idx[N - 1] = 0;

                uint32_t dest_linear_idx = 0;
                for (uint32_t i = 0; i < N - 1; ++i) {
                    dest_linear_idx += src_multi_idx[perm[i]] * dest_strides[i];
                }

                ttdm::noc_write_row_split<DST_PAGES_PER_ROW, DST_LOGICAL_PAGE_SIZE>(
                    noc, cb_out, l1_offset, d, dest_linear_idx, /*dst_offset=*/0, write_size);
                l1_offset += l1_page_stride;
                row++;
            }
            rows_left -= batch;
            prev_batch = batch;
        }

        // Drain final batch
        noc.async_writes_flushed();
        cb_out.pop_front(prev_batch);
    } else {
        // Non-batched: per-row barrier
        for (uint32_t i = 0; i < num_rows; i++) {
            uint32_t src_multi_idx[N];
            uint32_t remaining = row;
            for (uint32_t j = 0; j < N - 1; ++j) {
                uint32_t dim = N - 2 - j;
                src_multi_idx[dim] = remaining % input_shape[dim];
                remaining /= input_shape[dim];
            }
            src_multi_idx[N - 1] = 0;

            uint32_t dest_linear_idx = 0;
            for (uint32_t j = 0; j < N - 1; ++j) {
                dest_linear_idx += src_multi_idx[perm[j]] * dest_strides[j];
            }

            cb_out.wait_front(1);
            ttdm::noc_write_row_split<DST_PAGES_PER_ROW, DST_LOGICAL_PAGE_SIZE>(
                noc,
                cb_out,
                /*src_offset=*/0,
                d,
                dest_linear_idx,
                /*dst_offset=*/0,
                write_size);
            noc.async_writes_flushed();
            cb_out.pop_front(1);
            row++;
        }
    }
    noc.async_write_barrier();
}
