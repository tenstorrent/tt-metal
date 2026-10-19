// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Scatter writer for the permute that keeps the last axis in place: input row i goes to the output
// row the sequencer names. Common RT: dst_addr. RT: rows, then the sequencer args (output row
// strides, input row shape, row permutation, starting row index).
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/operations/data_movement/common/kernels/codegen/sequencers.h"

void kernel_main() {
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    uint32_t rows_left = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_id = get_named_compile_time_arg_val("cb_id");
    constexpr uint32_t write_size = get_named_compile_time_arg_val("write_size");
    constexpr uint32_t BATCH = get_named_compile_time_arg_val("batch");
    constexpr uint32_t row_axes = get_named_compile_time_arg_val("row_axes");
    constexpr auto dst_args = TensorAccessorArgs<0>();
    constexpr uint32_t seq_rt_start = 1;

    const auto d = TensorAccessor(dst_args, dst_addr, dst_args.get_aligned_page_size());
    const uint32_t l1_page_stride = get_local_cb_interface(cb_id).fifo_page_size << cb_addr_shift;
    auto seq = seq_permute_init(row_axes, seq_rt_start);

    Noc noc;
    CircularBuffer cb_out(cb_id);

    uint32_t prev_batch = 0;
    while (rows_left > 0) {
        const uint32_t batch = rows_left < BATCH ? rows_left : BATCH;
        cb_out.wait_front(prev_batch + batch);
        noc.async_writes_flushed();
        cb_out.pop_front(prev_batch);
        for (uint32_t t = 0; t < batch; t++) {
            noc.async_write(
                cb_out,
                d,
                write_size,
                {.offset_bytes = t * l1_page_stride},
                {.page_id = seq_permute_next(seq), .offset_bytes = 0});
        }
        rows_left -= batch;
        prev_batch = batch;
    }
    noc.async_writes_flushed();
    cb_out.pop_front(prev_batch);
    noc.async_write_barrier();
}
