// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
    uint32_t num_pages_per_core = get_arg(args::num_pages_per_core);
    uint32_t start_id = get_arg(args::start_id);

    constexpr uint32_t elems_per_page = get_arg(args::elems_per_page);
    constexpr uint32_t page_size = get_arg(args::page_size);

    value val;
    val.u = fill_value;

    Noc noc;
    // Holds the single fill-value page this instance builds and then writes to every output page it
    // owns. This instance is the buffer's only toucher: it fills the page and drains it itself.
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

    const auto s = TensorAccessor(tensor::output);

    dfb.wait_front(1);

    uint32_t end_id = start_id + num_pages_per_core;
    for (std::uint32_t i = start_id; i < end_id; i++) {
        noc.async_write(dfb, s, s.get_aligned_page_size(), {}, {.page_id = i});
    }
    noc.async_writes_flushed();
    dfb.pop_front(1);
    noc.async_write_barrier();
}
