// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

#include "api/debug/dprint.h"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

void kernel_main() {
    uint32_t batch_ids_addr = get_arg_val<uint32_t>(0);
    uint32_t batch_id_size = get_arg_val<uint32_t>(1);
    uint32_t input_addr_a = get_arg_val<uint32_t>(2);
    uint32_t input_addr_b = get_arg_val<uint32_t>(3);
    uint32_t batch_size_in_pages = get_arg_val<uint32_t>(4);
    uint32_t my_batch_id = get_arg_val<uint32_t>(5);
    // Args 6 and 7 are only used by IS_SHARD_LOCAL path (see below).

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t batch_cb_id = get_compile_time_arg_val(1);
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

    // Address stride between consecutive shard-local pages (rounded to 32B; same as page_size
    // for TILE since tile_size is always 32B-aligned).
    constexpr uint32_t shard_page_stride = (page_size + 31u) & ~31u;

    constexpr auto src0_args = TensorAccessorArgs<4>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto batch_ids_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(src0_args, input_addr_a);
    const auto s1 = TensorAccessor(src1_args, input_addr_b);

    // Third argument page_size from runtime args overrides TensorAccessorArgs::AlignedPageSize, which may be stale on
    // program cache hits.
    const auto batchAddr = TensorAccessor(batch_ids_args, batch_ids_addr, batch_id_size << 2);

    experimental::Noc noc;
    experimental::CircularBuffer cb_in0(cb_id_in0);
    experimental::CircularBuffer batch_cb(batch_cb_id);

    // 1. Read batch_ids into L1.
    volatile tt_l1_ptr int* addr_ptr = nullptr;

    if (batch_id_size > 0) {
        batch_cb.reserve_back(1);
        uint32_t l1_write_addr = batch_cb.get_write_ptr();
        noc.async_read(batchAddr, batch_cb, (batch_id_size << 2), {.page_id = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        batch_cb.push_back(1);
        addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(l1_write_addr);
    }

    // 2. Dispatch to the appropriate path.
    if constexpr (IS_SHARD_LOCAL) {
        // -------------------------------------------------------------------------------
        // Shard-local path (WIDTH_SHARDED / BLOCK_SHARDED).
        //
        // Each core independently processes ALL of its shard's batches.  The CB is
        // aliased to the output buffer so every page written here lands directly in the
        // output L1 shard at the correct sequential offset.
        //
        // Runtime args 6-7 carry the extra state needed for this path:
        //   arg[6] = batch_offset_a  (first global batch index for this core's shard;
        //                             0 for WIDTH_SHARDED, r*(B/n_y) for BLOCK_SHARDED)
        //   arg[7] = total_local_batches  (how many batches this core handles)
        // `batch_size_in_pages` (arg[4]) is the per-batch page count (shard_ppb).
        // -------------------------------------------------------------------------------
        const uint32_t batch_offset_a = get_arg_val<uint32_t>(6);
        const uint32_t total_local_batches = get_arg_val<uint32_t>(7);
        // Args 8-12: column-offset state for INTERLEAVED input_b reads.
        const uint32_t b_full_ppb = get_arg_val<uint32_t>(8);        // full pages/batch in input_b
        const uint32_t shard_tile_w = get_arg_val<uint32_t>(9);      // tile cols per shard (1 for ROW_MAJOR)
        const uint32_t full_tile_w = get_arg_val<uint32_t>(10);      // tile cols in full tensor (1 for ROW_MAJOR)
        const uint32_t col_page_offset = get_arg_val<uint32_t>(11);  // tile col offset (0 for ROW_MAJOR)
        const uint32_t col_byte_offset = get_arg_val<uint32_t>(12);  // byte offset into row (0 for TILE)
        const uint32_t shard_ppb = batch_size_in_pages;              // pages per batch in this shard

        if (shard_ppb == 0 || total_local_batches == 0) {
            return;
        }

        for (uint32_t b_local = 0; b_local < total_local_batches; ++b_local) {
            const uint32_t b_global = b_local + batch_offset_a;

            // Find whether this global batch is in the replacement set.
            // The full scan (no early break) is intentional: it mirrors PyTorch
            // index_copy_ semantics where duplicate indices give the last-listed
            // value priority.  Device kernels cannot use hash maps, so O(local_batches
            // × batch_id_size) is acceptable for typical small index counts.
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

            // Write shard_ppb pages sequentially into the aliased CB / output shard.
            // p_row/p_col track the 2D tile position within the shard incrementally to
            // avoid integer division/modulo (which are expensive on RISC-V Tensix cores).
            // For ROW_MAJOR shard_tile_w==1, so p_col is always 0 and p_row==p.
            uint32_t p_row = 0;
            uint32_t p_col = 0;
            for (uint32_t p = 0; p < shard_ppb; ++p) {
                cb_in0.reserve_back(1);
                if (replace_b) {
                    if constexpr (IS_B_SAME_SHARDED) {
                        // input_b is on this core's L1 shard — direct address arithmetic.
                        const uint32_t b_offset = (replace_src * shard_ppb + p) * shard_page_stride;
                        const uint64_t src_noc = get_noc_addr(input_addr_b + b_offset);
                        noc_async_read(src_noc, cb_in0.get_write_ptr(), page_size);
                        noc_async_read_barrier();
                    } else {
                        // input_b is INTERLEAVED — compute the global page index for the
                        // column slice this core is responsible for.
                        //
                        // ROW_MAJOR: each page is a full D-wide row; shard_tile_w==1,
                        //   full_tile_w==1, col_page_offset==0, col_byte_offset encodes cx.
                        //   src_page = replace_src * H_N + p  (unchanged stride).
                        //
                        // TILE: each page is one tile; the correct global tile index is:
                        //   replace_src * full_ppb + p_row * full_tile_w + col_page_offset + p_col
                        const uint32_t src_page =
                            replace_src * b_full_ppb + p_row * full_tile_w + col_page_offset + p_col;
                        noc.async_read(
                            s1,
                            cb_in0,
                            page_size,
                            {.page_id = src_page, .offset_bytes = col_byte_offset},
                            {.offset_bytes = 0});
                        noc.async_read_barrier();
                    }
                } else {
                    // Passthrough from input_a's local shard — direct address arithmetic.
                    const uint32_t a_offset = (b_local * shard_ppb + p) * shard_page_stride;
                    const uint64_t src_noc = get_noc_addr(input_addr_a + a_offset);
                    noc_async_read(src_noc, cb_in0.get_write_ptr(), page_size);
                    noc_async_read_barrier();
                }
                cb_in0.push_back(1);
                if (++p_col == shard_tile_w) {
                    p_col = 0;
                    ++p_row;
                }
            }
        }

    } else {
        // Decide replace / passthrough for this core's single batch.
        bool replace_batch = false;
        uint32_t batch_to_replace_id = 0;

        if (batch_id_size > 0) {
            // writes to the same destination overwrite earlier ones).
            for (uint32_t i = 0; i < batch_id_size; ++i) {
                if (static_cast<uint32_t>(addr_ptr[i]) == my_batch_id) {
                    replace_batch = true;
                    batch_to_replace_id = i;
                }
            }
        }

        // Inactive core — inner_count (= batch_size_in_pages) is set to 0 by the host.
        if (batch_size_in_pages == 0) {
            return;
        }

        if constexpr (IS_NATIVE) {
            // Native fast path: ONE NoC read of the whole batch slab. The sharded
            // TensorAccessor for input_a (HEIGHT_SHARDED, one batch per core) resolves
            // get_noc_addr(start_page) to this core's local L1 shard base; consecutive
            // pages within the shard are physically contiguous, so reading
            // `page_size * batch_size_in_pages` bytes from that base copies the whole
            // shard into the (output-aliased) data CB in a single request.
            const uint32_t start_id = my_batch_id * batch_size_in_pages;
            if (replace_batch) {
                const uint32_t b_start = batch_to_replace_id * batch_size_in_pages;
                const uint32_t b_end = b_start + batch_size_in_pages;
                for (uint32_t i = b_start; i < b_end; ++i) {
                    cb_in0.reserve_back(1);
                    noc.async_read(s1, cb_in0, page_size, {.page_id = i}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    cb_in0.push_back(1);
                }
            } else {
                cb_in0.reserve_back(batch_size_in_pages);
                noc.async_read(s0, cb_in0, page_size * batch_size_in_pages, {.page_id = start_id}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_in0.push_back(batch_size_in_pages);
            }
        } else {
            // Generic path (also used for arbitrary dim != 0 via stride args):
            //   arg[6] = outer_count    — dims before the target dim (1 for dim=0)
            //   arg[7] = outer_stride_a — page stride per outer step in input_a
            //   arg[8] = outer_stride_b — page stride per outer step in input_b
            //   arg[4] = inner_count    — pages per (outer, slice) pair
            // For dim=0: outer_count=1, inner_count=pages_per_batch — single loop,
            // behaviour is identical to the original code.
            const uint32_t outer_count = get_arg_val<uint32_t>(6);
            const uint32_t outer_stride_a = get_arg_val<uint32_t>(7);
            const uint32_t outer_stride_b = get_arg_val<uint32_t>(8);
            const uint32_t inner_count = batch_size_in_pages;

            if (inner_count == 0 || outer_count == 0) {
                return;
            }

            for (uint32_t outer = 0; outer < outer_count; ++outer) {
                for (uint32_t inner = 0; inner < inner_count; ++inner) {
                    cb_in0.reserve_back(1);
                    if (replace_batch) {
                        const uint32_t pid = outer * outer_stride_b + batch_to_replace_id * inner_count + inner;
                        noc.async_read(s1, cb_in0, page_size, {.page_id = pid}, {.offset_bytes = 0});
                    } else {
                        const uint32_t pid = outer * outer_stride_a + my_batch_id * inner_count + inner;
                        noc.async_read(s0, cb_in0, page_size, {.page_id = pid}, {.offset_bytes = 0});
                    }
                    noc.async_read_barrier();
                    cb_in0.push_back(1);
                }
            }
        }
    }
}
