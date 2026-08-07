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
    uint32_t start_shard_id = get_arg(args::start_shard_id);

    constexpr uint32_t elems_per_page = get_arg(args::elems_per_page);
    constexpr uint32_t page_size = get_arg(args::page_size);
    constexpr uint32_t num_shards = get_arg(args::num_shards);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    value val;
    val.u = fill_value;

    Noc noc;
    // Holds the single fill-value page this instance builds and then writes to every page of every
    // shard it owns. This instance is the buffer's only toucher: it fills the page and drains it itself.
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

    for (uint32_t shard_id = start_shard_id; shard_id < num_shards; shard_id += num_cores) {
        auto shard_pages = dst_accessor.shard_pages(shard_id);
        for (auto page_iter = shard_pages.begin(); page_iter != shard_pages.end(); ++page_iter) {
            uint32_t page_id = page_iter->page_id();
            noc.async_write(dfb, dst_accessor, dst_accessor.get_aligned_page_size(), {}, {.page_id = page_id});
        }
    }

    noc.async_writes_flushed();
    dfb.pop_front(1);
    noc.async_write_barrier();
}
