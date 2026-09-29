// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Make n reads defined by num_reads
// Writes to the bound Dataflow Buffer in L1
// Expects n input tensor bindings, reached positionally through the `inputs` binding sequence
void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    const uint32_t start_tensor = get_arg(args::start_tensor);
    const uint32_t start_tensor_id = get_arg(args::start_tensor_id);

    // The tensor binding sequence carries its own length, so the host passes no tensor count.
    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;

    // ublocks size defined in pages
    constexpr uint32_t ublock_size_pages = 1;

    [[maybe_unused]] uint32_t num_pages_per_block[num_tensors];
    uint32_t page_id_per_tensor[num_tensors];

    auto tensor_accessors_tuple = make_tensor_accessors(tensor::inputs);
    auto abstract_tensor_accessor_wrappers = make_abstract_tensor_accessor_wrappers(tensor_accessors_tuple);

    // Two num_tensors-element runtime vararg blocks, in the order the host supplies them:
    // num_pages_per_block first, then page_id_per_tensor.
    constexpr uint32_t page_id_per_tensor_offset = num_tensors;
    for (uint32_t i = 0; i < num_tensors; ++i) {
        num_pages_per_block[i] = get_vararg(i);
        page_id_per_tensor[i] = get_vararg(page_id_per_tensor_offset + i);
    }

    DataflowBuffer dfb_in(dfb::in);
    Noc noc;

    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    // FIX RM CONCAT WIDTH
    for (uint32_t i = 0; i < num_pages; ++i) {
        dfb_in.reserve_back(ublock_size_pages);
        uint32_t l1_write_addr = dfb_in.get_write_ptr();
#ifdef WIDTH_CONCAT
        // For width concat we know we start at curr_tensor=0
        // num_pages_per_block[curr_tensor] is always one for width concat
        for (uint32_t j = 0; j < num_tensors; ++j) {
            // The per-tensor page sizes are compile-time varargs: baked into the program, but
            // selected here by a value that advances at run time.
            auto page_size = get_compile_time_vararg(curr_tensor);
            noc.async_read(
                abstract_tensor_accessor_wrappers[curr_tensor],
                CoreLocalMem<uint8_t>(l1_write_addr),
                page_size,
                {.page_id = page_id_per_tensor[curr_tensor]},
                {});
            l1_write_addr += page_size;
            page_id_per_tensor[curr_tensor]++;
            curr_tensor++;
        }
        curr_tensor = 0;
#else
        auto page_size = get_compile_time_vararg(curr_tensor);
        noc.async_read(
            abstract_tensor_accessor_wrappers[curr_tensor],
            CoreLocalMem<uint8_t>(l1_write_addr),
            page_size,
            {.page_id = page_id_per_tensor[curr_tensor]},
            {});

        page_id_per_tensor[curr_tensor]++;
        curr_tensor_id++;

        if (curr_tensor_id == num_pages_per_block[curr_tensor]) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
            }
        }
#endif
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_pages);
    }
}
