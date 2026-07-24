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

void kernel_main() {
    uint32_t batch_ids_addr = get_arg_val<uint32_t>(0);
    uint32_t batch_id_size = get_arg_val<uint32_t>(1);
    uint32_t input_addr_a = get_arg_val<uint32_t>(2);
    uint32_t input_addr_b = get_arg_val<uint32_t>(3);
    uint32_t batch_size_in_pages = get_arg_val<uint32_t>(4);
    uint32_t my_batch_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t dfb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t batch_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t page_size = get_compile_time_arg_val(2);

    // Mode encoding (compile-time arg 3):
    //   0 = generic fallback  (TensorAccessor page-by-page from input_a)
    //   1 = native fast path  (HEIGHT_SHARDED L1 one-batch-per-core, CB aliased to output)
    //   2 = shard-local       (WIDTH/BLOCK_SHARDED, all-batches-per-core, CB aliased, INTERLEAVED input_b)
    //   3 = shard-local       (same as 2 but input_b is same-sharded; direct L1 reads for input_b too)
    constexpr uint32_t mode = get_compile_time_arg_val(3);
    constexpr bool IS_NATIVE = (mode == 1);
    constexpr bool IS_SHARD_LOCAL = (mode >= 2);
    constexpr bool IS_B_SAME_SHARDED = (mode == 3);

    // NOC addresses require 32B alignment; round up page_size so that shard-local direct
    // address arithmetic produces aligned base addresses on every core.
    constexpr uint32_t shard_page_stride = (page_size + 31u) & ~31u;

    constexpr auto src0_args = TensorAccessorArgs<4>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto batch_ids_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(src0_args, input_addr_a);
    const auto s1 = TensorAccessor(src1_args, input_addr_b);

    // page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto batchAddr = TensorAccessor(batch_ids_args, batch_ids_addr, batch_id_size << 2);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    DataflowBuffer batch_dfb(batch_dfb_id);

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
        const uint32_t batch_offset_a = get_arg_val<uint32_t>(6);
        const uint32_t total_local_batches = get_arg_val<uint32_t>(7);
        const uint32_t b_full_ppb = get_arg_val<uint32_t>(8);
        const uint32_t shard_tile_w = get_arg_val<uint32_t>(9);
        const uint32_t full_tile_w = get_arg_val<uint32_t>(10);
        const uint32_t col_page_offset = get_arg_val<uint32_t>(11);
        const uint32_t col_byte_offset = get_arg_val<uint32_t>(12);
        const uint32_t shard_ppb = batch_size_in_pages;

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
            // HEIGHT_SHARDED L1; pages are contiguous within the shard). The replace branch
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
            // Generic fallback path. Runtime args:
            //   arg[5] = slice_start    — first slice index for this core
            //   arg[6] = outer_count    — dims before the target dim (1 for dim=0)
            //   arg[7] = outer_stride_a — page stride per outer step in input_a
            //   arg[8] = outer_stride_b — page stride per outer step in input_b
            //   arg[9] = num_slices     — slices assigned to this core (work-splitting)
            //   arg[4] = inner_count    — pages per (outer, slice) pair
            const uint32_t slice_start = my_batch_id;
            const uint32_t num_slices = get_arg_val<uint32_t>(9);
            const uint32_t outer_count = get_arg_val<uint32_t>(6);
            const uint32_t outer_stride_a = get_arg_val<uint32_t>(7);
            const uint32_t outer_stride_b = get_arg_val<uint32_t>(8);
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
