// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t batch_id_size = get_arg(args::batch_id_size);
    uint32_t batch_size_in_pages = get_arg(args::batch_size_in_pages);
    uint32_t my_batch_id = get_arg(args::my_batch_id);

    constexpr uint32_t page_size = get_arg(args::page_size);

    // Mode encoding (the `mode` compile-time arg):
    //   0 = generic fallback  (TensorAccessor page-by-page from input_a)
    //   1 = native fast path  (HEIGHT_SHARDED SRAM one-batch-per-core, DFB borrowed from output)
    //   2 = shard-local       (WIDTH/BLOCK_SHARDED, all-batches-per-core, DFB borrowed, INTERLEAVED input_b)
    //   3 = shard-local       (same as 2 but input_b is same-sharded; direct SRAM reads for input_b too)
    constexpr uint32_t mode = get_arg(args::mode);
    constexpr bool IS_NATIVE = (mode == 1);
    constexpr bool IS_SHARD_LOCAL = (mode >= 2);
    constexpr bool IS_B_SAME_SHARDED = (mode == 3);

    // NOC addresses require 32B alignment; round up page_size so that shard-local direct
    // address arithmetic produces aligned base addresses on every core.
    constexpr uint32_t shard_page_stride = (page_size + 31u) & ~31u;

    const auto s0 = TensorAccessor(tensor::input_a);
    const auto s1 = TensorAccessor(tensor::input_b);
    const auto batchAddr = TensorAccessor(tensor::batch_ids);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer batch_dfb(dfb::batch);

    volatile tt_l1_ptr int* addr_ptr = nullptr;

    if (batch_id_size > 0) {
        batch_dfb.reserve_back(1);
        uint32_t l1_write_addr = batch_dfb.get_write_ptr();
        noc.async_read(batchAddr, batch_dfb, (batch_id_size << 2), {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        batch_dfb.push_back(1);
        addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(l1_write_addr);
    }

    if constexpr (IS_SHARD_LOCAL) {
        const uint32_t batch_offset_a = get_arg(args::batch_offset_a);
        const uint32_t total_local_batches = get_arg(args::total_local_batches);
        const uint32_t b_full_ppb = get_arg(args::b_full_ppb);
        const uint32_t shard_tile_w = get_arg(args::shard_tile_w);
        const uint32_t full_tile_w = get_arg(args::full_tile_w);
        const uint32_t col_page_offset = get_arg(args::col_page_offset);
        const uint32_t col_byte_offset = get_arg(args::col_byte_offset);
        const uint32_t shard_ppb = batch_size_in_pages;

        // Raw SRAM base addresses for this core's shard (Case 2 bindings): direct-address
        // arithmetic into input_a (both shard-local modes) and input_b (same-sharded mode only).
        const uint32_t input_addr_a = s0.get_bank_base_address();
        [[maybe_unused]] const uint32_t input_addr_b = s1.get_bank_base_address();

        if (shard_ppb == 0 || total_local_batches == 0) {
            return;
        }

        for (uint32_t b_local = 0; b_local < total_local_batches; ++b_local) {
            const uint32_t b_global = b_local + batch_offset_a;

            // Full scan (no early break) mirrors PyTorch index_copy_ semantics: duplicate
            // indices give the last-listed value priority.
            bool replace_b = false;
            uint32_t replace_src = 0;
            if (addr_ptr) {
                for (uint32_t k = 0; k < batch_id_size; ++k) {
                    if (static_cast<uint32_t>(addr_ptr[k]) == b_global) {
                        replace_b = true;
                        replace_src = k;
                    }
                }
            }

            // p_row/p_col track 2D tile position incrementally to avoid integer
            // division/modulo (expensive on RISC-V Tensix cores). For ROW_MAJOR,
            // shard_tile_w==1 so p_col is always 0 and p_row==p.
            uint32_t p_row = 0;
            uint32_t p_col = 0;
            for (uint32_t p = 0; p < shard_ppb; ++p) {
                dfb_in0.reserve_back(1);
                if (replace_b) {
                    if constexpr (IS_B_SAME_SHARDED) {
                        const uint32_t b_offset = (replace_src * shard_ppb + p) * shard_page_stride;
                        noc.async_read(
                            UnicastEndpoint{},
                            dfb_in0,
                            page_size,
                            {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                             .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                             .addr = input_addr_b + b_offset},
                            {.offset_bytes = 0});
                        noc.async_read_barrier();
                    } else {
                        // Global tile index for the column slice this core handles:
                        //   TILE:      replace_src * full_ppb + p_row * full_tile_w + col_page_offset + p_col
                        //   ROW_MAJOR: shard_tile_w==1, col_page_offset==0 → reduces to replace_src * b_full_ppb + p
                        const uint32_t src_page =
                            replace_src * b_full_ppb + p_row * full_tile_w + col_page_offset + p_col;
                        noc.async_read(
                            s1,
                            dfb_in0,
                            page_size,
                            {.page_id = src_page, .offset_bytes = col_byte_offset},
                            {.offset_bytes = 0});
                        noc.async_read_barrier();
                    }
                } else {
                    const uint32_t a_offset = (b_local * shard_ppb + p) * shard_page_stride;
                    noc.async_read(
                        UnicastEndpoint{},
                        dfb_in0,
                        page_size,
                        {.noc_x = (uint32_t)my_x[noc.get_noc_id()],
                         .noc_y = (uint32_t)my_y[noc.get_noc_id()],
                         .addr = input_addr_a + a_offset},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                }
                dfb_in0.push_back(1);
                if (++p_col == shard_tile_w) {
                    p_col = 0;
                    ++p_row;
                }
            }
        }

    } else {
        if constexpr (IS_NATIVE) {
            bool replace_batch = false;
            uint32_t batch_to_replace_id = 0;
            if (batch_id_size > 0) {
                for (uint32_t i = 0; i < batch_id_size; ++i) {
                    if (static_cast<uint32_t>(addr_ptr[i]) == my_batch_id) {
                        replace_batch = true;
                        batch_to_replace_id = i;
                    }
                }
            }

            // Inactive core — batch_size_in_pages set to 0 by the host.
            if (batch_size_in_pages == 0) {
                return;
            }

            // Passthrough copies the full batch slab in one bulk NoC read (input_a is
            // HEIGHT_SHARDED SRAM; pages are contiguous within the shard). The replace branch
            // falls back to per-page reads since input_b may be interleaved.
            const uint32_t start_id = my_batch_id * batch_size_in_pages;
            if (replace_batch) {
                const uint32_t b_start = batch_to_replace_id * batch_size_in_pages;
                const uint32_t b_end = b_start + batch_size_in_pages;
                for (uint32_t i = b_start; i < b_end; ++i) {
                    dfb_in0.reserve_back(1);
                    noc.async_read(s1, dfb_in0, page_size, {.page_id = i}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    dfb_in0.push_back(1);
                }
            } else {
                dfb_in0.reserve_back(batch_size_in_pages);
                noc.async_read(
                    s0, dfb_in0, page_size * batch_size_in_pages, {.page_id = start_id}, {.offset_bytes = 0});
                noc.async_read_barrier();
                dfb_in0.push_back(batch_size_in_pages);
            }
        } else {
            // Generic fallback path. Runtime args (named):
            //   slice_start (via my_batch_id) — first slice index for this core
            //   outer_count    — dims before the target dim (1 for dim=0)
            //   outer_stride_a — page stride per outer step in input_a
            //   outer_stride_b — page stride per outer step in input_b
            //   num_slices     — slices assigned to this core (work-splitting)
            //   inner_count (via batch_size_in_pages) — pages per (outer, slice) pair
            const uint32_t slice_start = my_batch_id;
            const uint32_t num_slices = get_arg(args::num_slices);
            const uint32_t outer_count = get_arg(args::outer_count);
            const uint32_t outer_stride_a = get_arg(args::outer_stride_a);
            const uint32_t outer_stride_b = get_arg(args::outer_stride_b);
            const uint32_t inner_count = batch_size_in_pages;

            if (num_slices == 0) {
                return;
            }

            for (uint32_t s = 0; s < num_slices; ++s) {
                const uint32_t my_slice = slice_start + s;

                bool replace_batch = false;
                uint32_t batch_to_replace_id = 0;
                if (addr_ptr) {
                    for (uint32_t k = 0; k < batch_id_size; ++k) {
                        if (static_cast<uint32_t>(addr_ptr[k]) == my_slice) {
                            replace_batch = true;
                            batch_to_replace_id = k;
                        }
                    }
                }

                for (uint32_t outer = 0; outer < outer_count; ++outer) {
                    for (uint32_t inner = 0; inner < inner_count; ++inner) {
                        dfb_in0.reserve_back(1);
                        if (replace_batch) {
                            const uint32_t pid = outer * outer_stride_b + batch_to_replace_id * inner_count + inner;
                            noc.async_read(s1, dfb_in0, page_size, {.page_id = pid}, {.offset_bytes = 0});
                        } else {
                            const uint32_t pid = outer * outer_stride_a + my_slice * inner_count + inner;
                            noc.async_read(s0, dfb_in0, page_size, {.page_id = pid}, {.offset_bytes = 0});
                        }
                        noc.async_read_barrier();
                        dfb_in0.push_back(1);
                    }
                }
            }
        }
    }
}
