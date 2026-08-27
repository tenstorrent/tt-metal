// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// BRISC writer for RM row-invariant permute (W dimension unchanged).
// Computes permuted output address for each input row, writes with
// pipelined batched NOC barriers.
//
// CT args: cb_id, stick_bytes, TensorAccessorArgs(out_t)..., BATCH, N
// RT args: num_rows, start_row, input_shape[N], perm[N], dest_strides[N]
// Common RT args: dst_addr
//
// The output address is a common arg because it is the one arg a later dispatch can change: a
// cache hit re-patches every buffer binding, and a per-core binding makes that one lookup per
// core rather than one for the program. The rest are per-core because they are never patched.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    uint32_t num_rows = get_arg_val<uint32_t>(0);
    uint32_t start_row = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t write_size = get_compile_time_arg_val(1);
    constexpr auto dst_args = TensorAccessorArgs<2>();
    constexpr uint32_t BATCH = get_compile_time_arg_val(dst_args.next_compile_time_args_offset());
    constexpr uint32_t N = get_compile_time_arg_val(dst_args.next_compile_time_args_offset() + 1);

    // A stick is never split on this path: the last axis is fixed, so an output page and a CB slot
    // are both this same stick rounded up to their own buffer's alignment. The write length is
    // therefore the unrounded stick, with no runtime clamp against either. The page half is asserted
    // here; the CB half is the program factory's, which checks the slot stride against these same
    // accessor page sizes before building the program.
    static_assert(!dst_args.page_size_is_crta, "row-invariant permute needs a compile-time output page size");
    static_assert(write_size <= dst_args.get_aligned_page_size(), "a stick overruns the output page it writes to");

    const auto d = TensorAccessor(dst_args, dst_addr, dst_args.get_aligned_page_size());
    const uint32_t l1_page_stride = get_local_cb_interface(cb_id).fifo_page_size << cb_addr_shift;

    Noc noc;
    CircularBuffer cb_out(cb_id);

    uint32_t input_shape[N];
    for (uint32_t i = 0; i < N; i++) {
        input_shape[i] = get_arg_val<uint32_t>(2 + i);
    }

    // Output-page distance of one step along input axis d. dest_strides is indexed by output axis,
    // so the permutation is applied once here rather than per row. This inversion needs perm to be
    // injective over the leading axes or two output strides would land on one input axis: the
    // factory builds this kernel only for a genuine permutation (is_permutation) that fixes the last
    // axis (is_row_invariant), which makes perm[0..N-2] a bijection onto the same range.
    uint32_t out_stride[N] = {};
    for (uint32_t i = 0; i + 1 < N; ++i) {
        out_stride[get_arg_val<uint32_t>(2 + N + i)] = get_arg_val<uint32_t>(2 + 2 * N + i);
    }

    // Rows are visited in input order one at a time, so the index and its output page are carried
    // forward as an odometer. input_shape is a runtime value, so decomposing each row instead would
    // cost N-1 divisions per row on a core with no divider; this pays them once per core.
    uint32_t idx[N];
    uint32_t remaining = start_row;
    for (uint32_t i = 0; i + 1 < N; ++i) {
        const uint32_t dim = N - 2 - i;
        idx[dim] = remaining % input_shape[dim];
        remaining /= input_shape[dim];
    }
    idx[N - 1] = 0;

    uint32_t dest_page = 0;
    uint32_t wrap[N];
    for (uint32_t dim = 0; dim + 1 < N; ++dim) {
        dest_page += idx[dim] * out_stride[dim];
        wrap[dim] = (input_shape[dim] - 1) * out_stride[dim];
    }

    auto advance_row = [&]() {
        for (uint32_t i = 0; i + 1 < N; ++i) {
            const uint32_t dim = N - 2 - i;
            if (++idx[dim] < input_shape[dim]) {
                dest_page += out_stride[dim];
                return;
            }
            idx[dim] = 0;
            dest_page -= wrap[dim];
        }
    };

    // Pipelined batched writer: the previous batch is held until the current one is in the CB, so
    // its NOC writes overlap the reader filling the next slots. The CB has to be two write batches
    // deep for that hold to be satisfiable.
    uint32_t rows_left = num_rows;
    uint32_t prev_batch = 0;
    while (rows_left > 0) {
        const uint32_t batch = (rows_left < BATCH) ? rows_left : BATCH;
        cb_out.wait_front(prev_batch + batch);
        if (prev_batch > 0) {
            // Freeing the slot only needs the write off the local NOC, not landed at the
            // destination; completion is claimed once for the whole kernel below.
            noc.async_writes_flushed();
            cb_out.pop_front(prev_batch);
        }

        uint32_t l1_offset = 0;
        for (uint32_t t = 0; t < batch; t++) {
            noc.async_write(
                cb_out, d, write_size, {.offset_bytes = l1_offset}, {.page_id = dest_page, .offset_bytes = 0});
            l1_offset += l1_page_stride;
            advance_row();
        }
        rows_left -= batch;
        prev_batch = batch;
    }
    if (prev_batch > 0) {
        noc.async_writes_flushed();
        cb_out.pop_front(prev_batch);
    }

    noc.async_write_barrier();
}
