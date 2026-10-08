// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Issues reads of num_pages consecutive pages of input I, starting at page page_id, to consecutive
// page-sized slots from l1_write_addr. The caller issues the read barrier.
// Only input I's accessor is built, so stack use does not grow with the number of inputs.
template <uint32_t I>
void read_input_pages(Noc& noc, uint32_t l1_write_addr, uint32_t page_id, uint32_t num_pages) {
    // The per-tensor page sizes are compile-time varargs.
    constexpr uint32_t page_size = get_compile_time_vararg<I>();
    const auto accessor = TensorAccessor(std::get<I>(tensor::inputs));
    for (uint32_t p = 0; p < num_pages; ++p) {
        noc.async_read(accessor, CoreLocalMem<uint8_t>(l1_write_addr), page_size, {.page_id = page_id + p}, {});
        l1_write_addr += page_size;
    }
}

// Calls read_input_pages<tensor_idx>() for an input index known only at run time.
template <uint32_t... Is>
void read_input_pages(
    uint32_t tensor_idx,
    Noc& noc,
    uint32_t l1_write_addr,
    uint32_t page_id,
    uint32_t num_pages,
    std::integer_sequence<uint32_t, Is...>) {
    (void)((tensor_idx == Is && (read_input_pages<Is>(noc, l1_write_addr, page_id, num_pages), true)) || ...);
}

// Reads num_pages pages into the bound Dataflow Buffer in L1.
// Expects n input tensor bindings, reached positionally through the `inputs` binding sequence.
void kernel_main() {
    const uint32_t num_pages = get_arg(args::num_pages);
    [[maybe_unused]] const uint32_t start_tensor = get_arg(args::start_tensor);
    [[maybe_unused]] const uint32_t start_tensor_id = get_arg(args::start_tensor_id);

    // The tensor binding sequence carries its own length, so the host passes no tensor count.
    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;
    constexpr auto tensor_indices = std::make_integer_sequence<uint32_t, num_tensors>();

    // ublocks size defined in pages
    constexpr uint32_t ublock_size_pages = 1;

    // Two num_tensors-element runtime vararg blocks, in the order the host supplies them:
    // num_pages_per_block first, then each input's first page id on this core. They are read when
    // needed instead of being copied to per-input stack arrays: with many inputs, per-input stack
    // objects overflow the small kernel stack (8 KB shared with TLS per DM core on Quasar).
    constexpr uint32_t page_id_per_tensor_offset = num_tensors;

    DataflowBuffer dfb_in(dfb::in);
    Noc noc;

#ifdef WIDTH_CONCAT
    // Each output page is one page from every input, in input order. Every block is one page and
    // this core starts at input 0, so output page i of this core reads page (first page id + i)
    // of each input.
    for (uint32_t i = 0; i < num_pages; ++i) {
        dfb_in.reserve_back(ublock_size_pages);
        uint32_t l1_write_addr = dfb_in.get_write_ptr();
        for (uint32_t j = 0; j < num_tensors; ++j) {
            read_input_pages(j, noc, l1_write_addr, get_vararg(page_id_per_tensor_offset + j) + i, 1, tensor_indices);
            l1_write_addr += get_compile_time_vararg(j);
        }
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_pages);
    }
#else
    // Pages are read round-robin, one block of num_pages_per_block[t] pages from each input t.
    uint32_t curr_tensor = start_tensor;
    uint32_t curr_tensor_id = start_tensor_id;
    uint32_t num_wraps = 0;  // times curr_tensor wrapped from the last input back to input 0
    for (uint32_t i = 0; i < num_pages; ++i) {
        const uint32_t num_pages_per_block = get_vararg(curr_tensor);

        // Pages of curr_tensor this core has already read: one block per earlier visit, less the part
        // of the first block skipped by starting mid-block. Inputs before start_tensor are first
        // visited after the first wrap.
        uint32_t pages_already_read = num_wraps * num_pages_per_block + curr_tensor_id;
        if (curr_tensor < start_tensor) {
            pages_already_read -= num_pages_per_block;
        } else if (curr_tensor == start_tensor) {
            pages_already_read -= start_tensor_id;
        }
        const uint32_t page_id = get_vararg(page_id_per_tensor_offset + curr_tensor) + pages_already_read;

        dfb_in.reserve_back(ublock_size_pages);
        read_input_pages(curr_tensor, noc, dfb_in.get_write_ptr(), page_id, 1, tensor_indices);
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_pages);

        curr_tensor_id++;
        if (curr_tensor_id == num_pages_per_block) {
            curr_tensor_id = 0;
            curr_tensor++;
            if (curr_tensor == num_tensors) {
                curr_tensor = 0;
                num_wraps++;
            }
        }
    }
#endif
}
