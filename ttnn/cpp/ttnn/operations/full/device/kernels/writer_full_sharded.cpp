// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "full_kernel_common.hpp"

void kernel_main() {
    uint32_t fill_value = get_arg(args::fill_value);
    uint32_t start_page_id = get_arg(args::start_page_id);
    uint32_t num_pages_per_shard_row = get_arg(args::num_pages_per_shard_row);
    uint32_t num_pages_per_shard_col = get_arg(args::num_pages_per_shard_col);

    constexpr uint32_t elems_per_page = get_arg(args::elems_per_page);
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t tensor_width_in_pages = get_arg(args::tensor_width_in_pages);

    value val;
    val.u = fill_value;

    Noc noc;
    // Holds the single fill-value page this instance builds and then writes to every page of its
    // shard. This instance is the buffer's only toucher: it fills the page and drains it itself.
    DataflowBuffer dfb(dfb::value);

    dfb.reserve_back(onepage);

    uint32_t write_addr = dfb.get_write_ptr();

    if (val.u == 0) {
        zero_buffer(dfb, page_size);
    } else {
#ifdef OUTPUT_DTYPE_BFLOAT16
        auto ptr = reinterpret_cast<uint16_t*>(write_addr);
        for (uint32_t i = 0; i < elems_per_page; ++i) {
            ptr[i] = val.u >> 16;
        }
#endif
#ifdef OUTPUT_DTYPE_INT32
        auto ptr = reinterpret_cast<uint32_t*>(write_addr);
        for (uint32_t i = 0; i < elems_per_page; ++i) {
            ptr[i] = fill_value;
        }
#endif
#ifdef OUTPUT_DTYPE_FLOAT32
        auto ptr = reinterpret_cast<float*>(write_addr);
        for (uint32_t i = 0; i < elems_per_page; ++i) {
            ptr[i] = val.f;
        }
#endif
    }

    dfb.push_back(1);

    const auto dst_accessor = TensorAccessor(tensor::output);

    dfb.wait_front(1);

    for (uint32_t shard_row_id = 0; shard_row_id < num_pages_per_shard_col; ++shard_row_id) {
        uint32_t curr_page_id = start_page_id;
        for (uint32_t shard_col_id = 0; shard_col_id < num_pages_per_shard_row; ++shard_col_id) {
            noc.async_write(dfb, dst_accessor, dst_accessor.get_aligned_page_size(), {}, {.page_id = curr_page_id});
            curr_page_id++;
        }
        start_page_id += tensor_width_in_pages;
    }

    noc.async_writes_flushed();
    dfb.pop_front(1);
    noc.async_write_barrier();
}
