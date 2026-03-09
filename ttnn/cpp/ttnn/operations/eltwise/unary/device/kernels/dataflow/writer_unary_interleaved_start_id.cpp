// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"
#include "experimental/circular_buffer.h"

namespace {
struct kernel_args {
    const uint32_t dst_addr;
#ifdef STRIDED_L1_ACCESS
    const uint32_t total_pages;
    const uint32_t bank_id;
    const uint32_t stride;
#else
    uint32_t num_pages;
    uint32_t start_id;
#endif
};

struct kernel_params {
    uint32_t cb_id_out;
    TensorAccessorArgs<1> dst_args;
    uint32_t batch_size;
};

inline kernel_args get_kernel_args() {
    return {
        .dst_addr = get_arg_val<uint32_t>(0),
#ifdef STRIDED_L1_ACCESS
        .total_pages = get_arg_val<uint32_t>(1),
        .bank_id = get_arg_val<uint32_t>(2),
        .stride = get_arg_val<uint32_t>(3),
#else
        .num_pages = get_arg_val<uint32_t>(1),
        .start_id = get_arg_val<uint32_t>(2),
#endif
    };
}

constexpr inline kernel_params get_kernel_params() {
    return {
        .cb_id_out = get_compile_time_arg_val(0),
        .dst_args = TensorAccessorArgs<1>(),
#ifdef BATCHED_CB_ACCESS
        .batch_size = 2,
#else
        .batch_size = 1,
#endif
    };
}
}  // namespace

void kernel_main() {
    const auto args = get_kernel_args();
    constexpr auto params = get_kernel_params();
    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(params.cb_id_out).fifo_page_size;

    experimental::Noc noc;
    experimental::CircularBuffer cb(params.cb_id_out);

#ifdef OUT_SHARDED
    cb.wait_front(args.num_pages);
#else

    const auto s = TensorAccessor(params.dst_args, args.dst_addr, page_bytes);

#ifdef STRIDED_L1_ACCESS
    // Strided access: each core writes only to its local L1 bank.
    // Single-tile loop — local L1 writes have zero NOC latency so
    // batching overhead hurts more than it helps.
    for (uint32_t i = args.bank_id; i < args.total_pages; i += args.stride) {
        cb.wait_front(1);
        noc.async_write(cb, s, page_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
#else
    // Contiguous access: each core writes a sequential range of pages.
#ifdef BACKWARDS
    const uint32_t end_id = args.start_id - args.num_pages;
    for (uint32_t i = args.start_id; i != end_id; --i) {
        cb.wait_front(1);
        noc.async_write(cb, s, page_bytes, {.offset_bytes = 0}, {.page_id = i});
        noc.async_write_barrier();
        cb.pop_front(1);
    }
#else
    const uint32_t end_id = args.start_id + args.num_pages;
    for (uint32_t i = args.start_id; i < end_id; i += params.batch_size) {
        const uint32_t cur_batch = (end_id - i < params.batch_size) ? (end_id - i) : params.batch_size;
        cb.wait_front(cur_batch);
        for (uint32_t j = 0; j < cur_batch; ++j) {
            noc.async_write(cb, s, page_bytes, {.offset_bytes = j * page_bytes}, {.page_id = i + j});
        }
        noc.async_write_barrier();
        cb.pop_front(cur_batch);
    }
#endif
#endif
#endif
}
