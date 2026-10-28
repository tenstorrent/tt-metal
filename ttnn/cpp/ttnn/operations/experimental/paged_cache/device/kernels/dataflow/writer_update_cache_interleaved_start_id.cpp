// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/experimental/paged_cache/device/kernels/dataflow/update_cache_rows.hpp"

void kernel_main() {
    Noc noc;

    const auto my_batch_idx = get_arg(args::my_batch_idx);
    const bool send_signal = get_arg(args::send_signal) == 1;
    const auto send_core_x = get_arg(args::send_core_x);
    const auto send_core_y = get_arg(args::send_core_y);

    constexpr auto cache_batch_num_tiles = get_arg(args::cache_batch_num_tiles);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto Wbytes = get_arg(args::Wbytes);

    // paged_cache args
    constexpr auto num_heads = get_arg(args::num_heads);

    constexpr auto St = get_arg(args::St);
    constexpr auto input_is_row_major = get_arg(args::input_is_row_major);
    // Rows written per user; also the fixed number of group slots (see the reader).
    constexpr auto num_rows = get_arg(args::num_rows);

    constexpr uint32_t head_offset_t = Wt * St;

    constexpr uint32_t TILE_HEIGHT = 32;

    const auto s0 = TensorAccessor(tensor::cache);

    // `dfb::cache` is this kernel's name for the *output* buffer, which compute fills with the
    // retilized block; the reader's same-named handle is the cache buffer it read from DRAM.
    DataflowBuffer dfb_cache(dfb::cache);
    DataflowBuffer dfb_untilized_cache(dfb::untilized_cache);
    DataflowBuffer dfb_untilized_cache2(dfb::untilized_cache2);
    DataflowBuffer dfb_untilized_input(dfb::untilized_input);
#ifdef USE_INDEX_TENSOR
    DataflowBuffer dfb_index(dfb::index);
#endif
#ifdef IS_PAGED_CACHE
    DataflowBuffer dfb_page_table(dfb::page_table);
#endif

    const uint32_t cache_tile_bytes = dfb_cache.get_tile_size();

#ifdef USE_INDEX_TENSOR
    dfb_index.wait_front(1);
    volatile tt_l1_ptr uint32_t* user_positions =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_index.get_read_ptr()) + my_batch_idx * num_rows;
    auto position = [user_positions](uint32_t s) -> uint32_t { return user_positions[s]; };
#else
    auto position = [](uint32_t s) -> uint32_t { return get_vararg(s); };
#endif

    volatile tt_l1_ptr uint32_t* page_table_ptr = nullptr;
#ifdef IS_PAGED_CACHE
    dfb_page_table.wait_front(1);
    page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dfb_page_table.get_read_ptr());
#endif

    const UpdateCacheRows<decltype(position)> rows{
        .position = position,
        .batch_tile_offset = my_batch_idx * cache_batch_num_tiles,
        .page_table = page_table_ptr,
    };

    constexpr uint32_t input_cb_pages = input_is_row_major ? 1 : Wt;
    dfb_untilized_input.wait_front(input_cb_pages);  // input tensor
#ifndef ARCH_QUASAR
    // Gen1 patches the new row into the untilized cache block with a NoC loopback read from our own
    // L1; Quasar copies directly (see the ARCH_QUASAR branch below), so these are Gen1-only.
    const uint8_t noc_id = noc.get_noc_id();
    const uint32_t my_noc_x = my_x[noc_id];
    const uint32_t my_noc_y = my_y[noc_id];
    UnicastEndpoint local_src;
#endif
    const uint32_t input_l1_read_addr = dfb_untilized_input.get_read_ptr();

    uint32_t num_groups = 0;
    for (uint32_t s = 0; s < num_rows; ++s) {
        uint32_t cache_id = rows.group_tile_id(s);
        if (cache_id == UPDATE_CACHE_SKIP_ROW) {
            continue;
        }
        ++num_groups;
        for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
            // Wait on compute to untilize a block. Patch every row of this tile-row into it in SRAM;
            // rows targeting the same position are applied in order, so the last one wins.
            dfb_untilized_cache.wait_front(Wt);
            dfb_untilized_cache2.reserve_back(Wt);

            const uint32_t untilized_cache_addr = dfb_untilized_cache.get_read_ptr();
            for (uint32_t r = s; r < num_rows; ++r) {
                if (r != s && rows.tile_id(r) != cache_id) {
                    continue;
                }
                const uint32_t src_addr = input_l1_read_addr + (r * num_heads + cur_head) * Wbytes;
                const uint32_t dst_addr = untilized_cache_addr + (position(r) % TILE_HEIGHT) * Wbytes;
#ifdef ARCH_QUASAR
                // Quasar: this is a local L1->L1 self-copy. A NoC loopback read on the emulator can spin
                // on can_post or silently drop, and the DFB pointer getters return the uncached alias the
                // NoC will not accept — so copy directly with the RISC through the uncached aliases (both
                // rows are 16 B aligned, Wbytes is a multiple of 4).
                volatile tt_l1_ptr uint32_t* src_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_addr);
                volatile tt_l1_ptr uint32_t* dst_row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
                for (uint32_t w = 0; w < Wbytes / sizeof(uint32_t); ++w) {
                    dst_row[w] = src_row[w];
                }
#else
                noc.async_read(
                    local_src,
                    CoreLocalMem<uint32_t>(dst_addr),
                    Wbytes,
                    {.noc_x = my_noc_x, .noc_y = my_noc_y, .addr = src_addr},
                    {});
                // Same-position rows must land in order; reads may complete out of order.
                noc.async_read_barrier();
#endif
            }
#ifdef ARCH_QUASAR
            // Orders the row stores ahead of the dfb_untilized_cache2 credit posted below, so compute
            // cannot re-tilize a stale row.
            __asm__ __volatile__("fence" ::: "memory");
#endif
            dfb_untilized_cache2.push_back(Wt);
            dfb_untilized_cache.pop_front(Wt);

            // Wait on compute to tilize an updated block. Write that block to DRAM
            dfb_cache.wait_front(Wt);
#ifdef ARCH_QUASAR
            // Quasar: DFB endpoint (cached address) + per-tile offset into the Wt-entry front, instead
            // of walking the uncached get_read_ptr() alias (see the self-copy above).
            for (uint32_t t = 0; t < Wt; ++t) {
                noc.async_write(
                    dfb_cache, s0, cache_tile_bytes, {.offset_bytes = t * cache_tile_bytes}, {.page_id = cache_id + t});
            }
#else
            uint32_t out_l1_read_addr = dfb_cache.get_read_ptr();
            for (uint32_t curr_cache_id = cache_id; curr_cache_id < cache_id + Wt; ++curr_cache_id) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(out_l1_read_addr), s0, cache_tile_bytes, {}, {.page_id = curr_cache_id});
                out_l1_read_addr += cache_tile_bytes;
            }
#endif
            noc.async_writes_flushed();
            dfb_cache.pop_front(Wt);
            // Delay syncing the writes to maximize perf.
            noc.async_write_barrier();

            cache_id += head_offset_t;
        }
    }

    // Unused group slots: let compute run through, drop its result.
    for (; num_groups < num_rows; ++num_groups) {
        for (uint32_t cur_head = 0; cur_head < num_heads; ++cur_head) {
            dfb_untilized_cache.wait_front(Wt);
            dfb_untilized_cache2.reserve_back(Wt);
            dfb_untilized_cache2.push_back(Wt);
            dfb_untilized_cache.pop_front(Wt);
            dfb_cache.wait_front(Wt);
            dfb_cache.pop_front(Wt);
        }
    }

    dfb_untilized_input.pop_front(input_cb_pages);
#ifdef USE_INDEX_TENSOR
    dfb_index.pop_front(1);
#endif
#ifdef IS_PAGED_CACHE
    dfb_page_table.pop_front(1);
#endif

    if (send_signal) {
        // send signal to receiver core that we are done using the input buffer
        Semaphore(sem::in0_seq).up(noc, send_core_x, send_core_y, 1);
        noc.async_atomic_barrier();
    }
}
