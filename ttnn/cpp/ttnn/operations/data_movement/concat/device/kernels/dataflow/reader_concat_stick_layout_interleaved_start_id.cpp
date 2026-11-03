// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <utility>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "ckernel.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/concat/device/kernels/dataflow/concat_reader_common.hpp"

// Reads input I; staged reads finish here, while direct reads use the caller's barrier.
// The per-tensor page sizes are compile-time varargs.
template <uint32_t I>
FORCE_INLINE void read_input_page(Noc& noc, uint32_t l1_write_addr, uint32_t page_id) {
    if constexpr (get_compile_time_vararg<I>() == 0) {
        return;
    }
    const auto accessor = TensorAccessor(std::get<I>(tensor::inputs));
#ifdef STAGE_UNALIGNED
    constexpr uint32_t num_tensors = std::tuple_size_v<decltype(tensor::inputs)>;
    if constexpr (get_compile_time_vararg<num_tensors + I>() != 0) {
        Scratchpad<volatile uint8_t> staging(scratch::staging);
        Scratchpad<volatile uint16_t> staging_words(scratch::staging);
        constexpr uint32_t page_bytes = get_compile_time_vararg<I>();
        constexpr uint32_t chunk_bytes = get_compile_time_vararg<2 * num_tensors>();
        for (uint32_t offset = 0; offset < page_bytes; offset += chunk_bytes) {
            const uint32_t bytes = page_bytes - offset < chunk_bytes ? page_bytes - offset : chunk_bytes;
            noc.async_read(accessor, staging, bytes, {.page_id = page_id, .offset_bytes = offset}, {});
            noc.async_read_barrier();
            const auto src = staging.local_mem();
            CoreLocalMem<volatile uint8_t> dst(l1_write_addr + offset);
            CoreLocalMem<volatile uint16_t> dst_words((l1_write_addr + offset + 1) & ~1U);
            uint32_t byte;
            if ((l1_write_addr & 1U) == 0) {
                const auto src_words = staging_words.local_mem();
                for (uint32_t word = 0; word < bytes / sizeof(uint16_t); ++word) {
                    dst_words[word] = src_words[word];
                }
                byte = bytes & ~1U;
            } else {
                dst[0] = src[0];
                for (byte = 1; byte + 1 < bytes; byte += 2) {
                    dst_words[(byte - 1) / 2] =
                        static_cast<uint16_t>(src[byte]) | (static_cast<uint16_t>(src[byte + 1]) << 8);
                }
            }
            if (byte < bytes) {
                dst[byte] = src[byte];
            }
        }
        // L1 stores stay ordered; match this final halfword store to drain the output stores too.
        staging_words[0] = 0;
        (void)ckernel::load_blocking(&staging_words[0]);
    } else
#endif
    {
        noc.async_read(
            accessor, CoreLocalMem<uint8_t>(l1_write_addr), get_compile_time_vararg<I>(), {.page_id = page_id}, {});
    }
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
            concat_reader::with_input(j, tensor_indices, [&](auto I) {
                read_input_page<decltype(I)::value>(noc, l1_write_addr, get_vararg(page_id_per_tensor_offset + j) + i);
            });
            l1_write_addr += get_compile_time_vararg(j);
        }
        noc.async_read_barrier();
        dfb_in.push_back(ublock_size_pages);
    }
#else
    // Pages are read round-robin, one block of num_pages_per_block[t] pages from each input t. Each step
    // reads the rest of the current input's block, which is no pages for an input that is empty along
    // dim, so the cursor still moves on to the next input.
    concat_reader::RoundRobinCursor cursor(start_tensor, start_tensor_id);
    uint32_t pages_left = num_pages;
    while (pages_left > 0) {
        const uint32_t num_pages_per_block = get_vararg(cursor.tensor);
        const uint32_t page_id =
            get_vararg(page_id_per_tensor_offset + cursor.tensor) + cursor.pages_already_read(num_pages_per_block);

        uint32_t pages_to_read = num_pages_per_block - cursor.id_in_block;
        if (pages_to_read > pages_left) {
            pages_to_read = pages_left;
        }
        concat_reader::with_input(cursor.tensor, tensor_indices, [&](auto I) {
            constexpr uint32_t input = decltype(I)::value;
            const auto accessor = TensorAccessor(std::get<input>(tensor::inputs));
            for (uint32_t p = 0; p < pages_to_read; ++p) {
                dfb_in.reserve_back(ublock_size_pages);
                noc.async_read(
                    accessor,
                    CoreLocalMem<uint8_t>(dfb_in.get_write_ptr()),
                    get_compile_time_vararg<input>(),
                    {.page_id = page_id + p},
                    {});
                noc.async_read_barrier();
                dfb_in.push_back(ublock_size_pages);
            }
        });
        pages_left -= pages_to_read;
        cursor.advance(pages_to_read, num_pages_per_block, num_tensors);
    }
#endif
}
