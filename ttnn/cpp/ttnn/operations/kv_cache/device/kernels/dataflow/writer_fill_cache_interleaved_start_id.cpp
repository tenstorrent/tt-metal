// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Writes this core's input tile rows into the cache. A core's rows can cross from one head into the
// next, and a head spans input_Ht tile rows in the input but cache_HtWt / Wt in the cache, so the
// cache id cannot simply be walked with ++. Each tile row is addressed from its own (head, row).
void kernel_main() {
    constexpr std::uint32_t Wt = get_arg(args::Wt);

    // Runtime rather than compile-time, so one binary serves every prompt length.
    const std::uint32_t input_Ht = get_arg(args::input_Ht);
    // Cache tiles between the end of one head's filled rows and the start of the next head's.
    const std::uint32_t cache_head_skip = get_arg(args::cache_head_skip);
    const std::uint32_t num_blocks = get_arg(args::num_blocks);
    const std::uint32_t start_id = get_arg(args::start_id);
    const std::uint32_t start_row = get_arg(args::start_row);

    Noc noc;
    DataflowBuffer dfb(dfb::out);
    const std::uint32_t page_bytes = dfb.get_entry_size();
    const auto s = TensorAccessor(tensor::dst);

    std::uint32_t cache_id = start_id;
    std::uint32_t row = start_row;
    for (std::uint32_t block = 0; block < num_blocks; ++block) {
        for (std::uint32_t w = 0; w < Wt; ++w) {
            dfb.wait_front(1);
            noc.async_write(dfb, s, page_bytes, {}, {.page_id = cache_id + w});
            noc.async_writes_flushed();
            dfb.pop_front(1);
        }
        cache_id += Wt;
        if (++row == input_Ht) {
            row = 0;
            cache_id += cache_head_skip;
        }
    }
    noc.async_write_barrier();
}
