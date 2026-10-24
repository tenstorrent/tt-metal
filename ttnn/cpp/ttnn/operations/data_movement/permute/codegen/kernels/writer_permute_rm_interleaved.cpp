// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scatter writer for the permute that keeps the last axis in place: input row i goes to the output
// row seq_permute_next names. Common RT: dst_addr. RT: rows, row_axes, then output row strides, input
// row shape, row permutation and starting row index (row_axes values each).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/codegen/sequencers.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    uint32_t rows_left = get_arg_val<uint32_t>(0);
    // Read at run time: with a compile-time axis count the compiler sees the unset tail of the state's
    // fixed-width arrays and fails the build with -Werror=maybe-uninitialized.
    const uint32_t row_axes = get_arg_val<uint32_t>(1);
    constexpr uint32_t kSequencerArgsStart = 2;

    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_id");
    constexpr uint32_t write_size = get_named_compile_time_arg_val("write_size");
    constexpr uint32_t BATCH = get_named_compile_time_arg_val("batch");
    constexpr auto dst_args = TensorAccessorArgs<0>();

    const auto d = TensorAccessor(dst_args, dst_addr, dst_args.get_aligned_page_size());
    const uint32_t l1_page_stride = get_local_cb_interface(cb_id).fifo_page_size << cb_addr_shift;

    auto seq = seq_permute_init(row_axes, kSequencerArgsStart);

    Noc noc;
    CircularBuffer cb_out(cb_id);

    uint32_t prev_batch = 0;
    while (rows_left > 0) {
        const uint32_t batch = rows_left < BATCH ? rows_left : BATCH;
        cb_out.wait_front(prev_batch + batch);
        noc.async_writes_flushed();
        cb_out.pop_front(prev_batch);
        for (uint32_t t = 0; t < batch; t++) {
            const uint32_t out_row = seq_permute_next(seq);
            noc.async_write(
                cb_out, d, write_size, {.offset_bytes = t * l1_page_stride}, {.page_id = out_row, .offset_bytes = 0});
        }
        rows_left -= batch;
        prev_batch = batch;
    }
    noc.async_writes_flushed();
    cb_out.pop_front(prev_batch);
    noc.async_write_barrier();
}
