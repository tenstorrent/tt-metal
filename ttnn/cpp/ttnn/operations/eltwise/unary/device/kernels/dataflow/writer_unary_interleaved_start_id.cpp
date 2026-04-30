// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

#if defined(REDUCE_METAL2_NAMED_ARGS) || defined(REDUCE_MCW_NAMED_ARGS)
    constexpr uint32_t cb_id_out = get_named_compile_time_arg_val("output_cb_index");
    constexpr uint32_t dst_args_config = get_named_compile_time_arg_val("dst_args_config");
    constexpr uint32_t dst_page_size = get_named_compile_time_arg_val("dst_page_size");
#else
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
#endif

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

    experimental::Noc noc;
    experimental::CircularBuffer cb(cb_id_out);

#ifdef OUT_SHARDED
    cb.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

#if defined(REDUCE_METAL2_NAMED_ARGS) || defined(REDUCE_MCW_NAMED_ARGS)
    constexpr bool dst_is_dram = (dst_args_config & static_cast<uint32_t>(tensor_accessor::ArgConfig::IsDram)) != 0;
    const auto s = TensorAccessor(tensor_accessor::make_interleaved_dspec<dst_is_dram>(), dst_addr, dst_page_size);
#else
    const auto s = TensorAccessor(dst_args, dst_addr);
#endif

#ifdef BACKWARDS
    uint32_t end_id = start_id - num_pages;
    for (uint32_t i = start_id; i != end_id; --i) {
#else
    uint32_t end_id = start_id + num_pages;
    for (uint32_t i = start_id; i < end_id; ++i) {
#endif
        cb.wait_front(onepage);
        noc.async_write(cb, s, page_bytes, {}, {.page_id = i});
        noc.async_writes_flushed();
        cb.pop_front(onepage);
    }
    noc.async_write_barrier();
#endif
}
