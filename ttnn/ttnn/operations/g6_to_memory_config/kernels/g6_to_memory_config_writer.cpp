// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config push mover and bounce writer: every NoC write is sourced from this core's own L1.
//   WALK_SHARD  push_shard  write each page of the owned, resident source shard to its target page
//   WALK_BANK   push_bank   write each page of this core's source L1 bank stripe to its target page
//   WALK_BOUNCE bounce      drain cb_bounce_pages block by block to the target pages
// The resident source is consumed in place, never re-read over the NoC. No kernel_lib helper applies:
// the op moves whole pages as bytes (no compute phase).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_to_memory_config_common.hpp"

namespace {

template <typename Src, typename Dst>
FORCE_INLINE void push_shard(const Src& src, const Dst& dst, uint32_t owned_shard_id) {
    if (owned_shard_id == g6::NO_SHARD) {
        return;
    }
    ASSERT(src.is_local_shard(owned_shard_id));
    const g6::Slice slice = g6::mover_slice(src.dspec().shard_volume());
    if (slice.begin == slice.end) {
        return;
    }
    {
        DeviceZoneScopedN("G6_PUSH_ISSUE");
        // shard_pages skips the slots of a ragged last shard that lie past the tensor.
        for (const auto& page : src.shard_pages(owned_shard_id, slice.begin, slice.end)) {
            noc_async_write(g6::local_slot(page.noc_addr()), dst.get_noc_addr(page.page_id()), g6::page_size);
        }
    }
    {
        DeviceZoneScopedN("G6_PUSH_DRAIN");
        noc_async_write_barrier();
    }
}

template <typename Src, typename Dst>
FORCE_INLINE void push_bank(const Src& src, const Dst& dst) {
    uint32_t bank;
    {
        DeviceZoneScopedN("G6_OWN_SCAN");
        bank = g6::find_own_bank(src);
    }
    if (bank == g6::NO_BANK) {
        return;
    }
    constexpr uint32_t stride = g6::movers_per_owner * NUM_L1_BANKS;
    {
        DeviceZoneScopedN("G6_PUSH_ISSUE");
        for (uint32_t page = bank + g6::mover_index * NUM_L1_BANKS; page < g6::num_pages; page += stride) {
            noc_async_write(g6::local_slot(src.get_noc_addr(page)), dst.get_noc_addr(page), g6::page_size);
        }
    }
    {
        DeviceZoneScopedN("G6_PUSH_DRAIN");
        noc_async_write_barrier();
    }
}

// Pages from `page` (before `end`) that sit back to back in the target, so one write covers them.
// The accessor stops a run at a shard edge and at the tensor edge, so a ragged last shard never
// writes into its padding slots; an interleaved target (page stride = bank count) gives runs of 1.
template <typename Dst>
FORCE_INLINE uint32_t target_run_pages(const Dst& dst, uint32_t page, uint32_t end) {
    if constexpr (g6::bounce_write_run_pages == 1) {
        return 1;
    } else {
        if (dst.contiguous_page_stride() != 1) {
            return 1;
        }
        const uint32_t run = dst.num_contiguous_pages(page, end);
        if constexpr (g6::bounce_write_run_pages == 0) {
            return run;
        } else {
            return run < g6::bounce_write_run_pages ? run : g6::bounce_write_run_pages;
        }
    }
}

template <typename Dst>
FORCE_INLINE void bounce_write(const Dst& dst, uint32_t start_page, uint32_t core_pages) {
    for (uint32_t done = 0; done < core_pages;) {
        const uint32_t remaining = core_pages - done;
        const uint32_t block_pages = remaining < g6::bounce_block_pages ? remaining : g6::bounce_block_pages;
        {
            DeviceZoneScopedN("G6_BOUNCE_WR_WAIT_DATA");
            cb_wait_front(g6::cb_bounce_pages, g6::bounce_block_pages);
        }
        {
            DeviceZoneScopedN("G6_BOUNCE_WR_ISSUE");
            uint32_t slot = get_read_ptr(g6::cb_bounce_pages);
            const uint32_t end = start_page + done + block_pages;
            for (uint32_t page = start_page + done; page < end;) {
                const uint32_t run = target_run_pages(dst, page, end);
#ifndef G6_ABLATE_WRITE
                noc_async_write(slot, dst.get_noc_addr(page), run * g6::page_size);
#endif
                slot += run * g6::page_size;
                page += run;
            }
        }
        {
            DeviceZoneScopedN("G6_BOUNCE_WR_FLUSH");
            // The slot is reusable once its bytes have left L1; completion is awaited once at the end.
            noc_async_writes_flushed();
        }
        cb_pop_front(g6::cb_bounce_pages, g6::bounce_block_pages);
        done += block_pages;
    }
    {
        DeviceZoneScopedN("G6_BOUNCE_WR_DRAIN");
        noc_async_write_barrier();
    }
}

}  // namespace

void kernel_main() {
    const uint32_t src_base = get_common_arg_val<uint32_t>(0);
    const uint32_t dst_base = get_common_arg_val<uint32_t>(1);
    [[maybe_unused]] const auto src = TensorAccessor(g6::src_args, src_base, g6::page_size);
    [[maybe_unused]] const auto dst = TensorAccessor(g6::dst_args, dst_base, g6::page_size);

    if constexpr (g6::walk == g6::WALK_SHARD) {
        push_shard(src, dst, get_arg_val<uint32_t>(0));
    } else if constexpr (g6::walk == g6::WALK_BANK) {
        push_bank(src, dst);
    } else {
        bounce_write(dst, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    }
}
