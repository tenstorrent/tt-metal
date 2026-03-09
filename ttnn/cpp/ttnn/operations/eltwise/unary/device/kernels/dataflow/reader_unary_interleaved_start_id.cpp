// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/tensor.h"
#include "experimental/circular_buffer.h"

namespace {
struct kernel_args {
    const uint32_t src_addr;
#ifdef STRIDED_L1_ACCESS
    const uint32_t total_pages;
    const uint32_t bank_id;
    const uint32_t stride;
#else
    const uint32_t num_pages;
    const uint32_t start_id;
#endif
};

struct kernel_params {
    uint32_t batch_size;
    TensorAccessorArgs<0> src_args;
};

inline kernel_args get_args() {
    return kernel_args{
        .src_addr = get_arg_val<uint32_t>(0),
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

inline constexpr kernel_params get_params() {
    return kernel_params{
#ifdef BATCHED_CB_ACCESS
        .batch_size = 2,
#else
        .batch_size = 1,
#endif
        .src_args{},
    };
}

}  // namespace

void kernel_main() {
    const auto args = get_args();
    constexpr auto params = get_params();
    constexpr uint32_t cb_id_in0 = 0;

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;

    const auto s = TensorAccessor(params.src_args, args.src_addr, page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);

#ifdef STRIDED_L1_ACCESS
    // Strided access: each core reads only from its local L1 bank.
    // Single-tile loop — local L1 reads have zero NOC latency so
    // batching overhead hurts more than it helps.
    for (uint32_t i = args.bank_id; i < args.total_pages; i += args.stride) {
        cb.reserve_back(1);
        noc.async_read(s, cb, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(1);
    }
#else
    // Contiguous access: each core reads a sequential range of pages.

// read a ublock of pages from src to CB, and then push the ublock to unpacker
#ifdef BACKWARDS
    const uint32_t end_id = args.start_id - args.num_pages;
    for (uint32_t i = args.start_id; i != end_id; --i) {
        cb.reserve_back(1);
        noc.async_read(s, cb, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(1);
    }
#else
    const uint32_t end_id = args.start_id + args.num_pages;
    for (uint32_t i = args.start_id; i < end_id; i += params.batch_size) {
        const uint32_t cur_batch = (end_id - i < params.batch_size) ? (end_id - i) : params.batch_size;
        cb.reserve_back(cur_batch);
        for (uint32_t j = 0; j < cur_batch; ++j) {
            noc.async_read(s, cb, page_bytes, {.page_id = i + j}, {.offset_bytes = j * page_bytes});
        }
        noc.async_read_barrier();
        cb.push_back(cur_batch);
    }
#endif
#endif
}
