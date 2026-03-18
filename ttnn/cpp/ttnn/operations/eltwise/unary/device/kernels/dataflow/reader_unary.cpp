// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Unified unary reader: interleaved (TensorAccessor + noc read) and sharded (cb_push_back only).
 * Compile-time args: [cb_id, ...TensorAccessorArgs(source buffer)].
 * Runtime args: src_addr(0), num_pages(1), start_id(2), num_tiles_sharded(3).
 * Define SRC_SHARDED=1 for sharded path.
 */

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    // Compile-time: cb_id at index 0; TensorAccessorArgs for source start at index 1 when !SRC_SHARDED
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

#if SRC_SHARDED
    const uint32_t num_tiles_per_core = get_arg_val<uint32_t>(3);
    experimental::CircularBuffer cb(cb_id_in0);
    cb.reserve_back(num_tiles_per_core);
    cb.push_back(num_tiles_per_core);
#else
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<1, 0>();

    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(src_args, src_addr, page_bytes);

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_in0);

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb.reserve_back(onepage);
        noc.async_read(s, cb, page_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb.push_back(onepage);
    }
#endif
}
