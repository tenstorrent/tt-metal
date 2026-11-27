// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config push mover and bounce writer: every NoC write is sourced from this core's own L1.
//   WALK_SHARD  push_shard  write each page of the owned, resident source shard to its target page
//   WALK_BANK   push_bank   write each page of this core's source L1 bank stripe to its target page
//   WALK_BOUNCE bounce      drain cb_bounce_pages block by block to the target pages, one write per run of
//                           pages contiguous in the target
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

// Pages from `page` (before `end`) that sit back to back in the target, so one write covers them, capped at
// bounce_write_run_pages. The accessor stops a run at a shard edge and at the tensor edge, so a ragged last shard
// never writes into its padding slots; an interleaved target (page stride = bank count) gives runs of 1.
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

// Drains cb_bounce_pages block by block: wait for the nominal block, write its pages, flush, pop. The walk hands
// over runs of pages contiguous in the target; a run that starts where the open one ends extends it (up to
// bounce_write_run_pages), so one NoC write covers each stretch contiguous in the target. Runs never cross a block.
class BounceDrain {
public:
    explicit BounceDrain(uint32_t core_pages) : left_(core_pages) {}

    // Pages the open block still takes; opens a block when none is open.
    FORCE_INLINE uint32_t room() {
        if (in_block_ == 0) {
            block_pages_ = left_ < g6::bounce_block_pages ? left_ : g6::bounce_block_pages;
            {
                DeviceZoneScopedN("G6_BOUNCE_WR_WAIT_DATA");
                cb_wait_front(g6::cb_bounce_pages, g6::bounce_block_pages);
            }
            slot_ = get_read_ptr(g6::cb_bounce_pages);
        }
        return block_pages_ - in_block_;
    }

    // The next `pages` pages of the walk, back to back in the target from dst_addr; pages <= room().
    FORCE_INLINE void pages(uint64_t dst_addr, uint32_t pages) {
        if (run_pages_ != 0 && run_pages_ + pages <= max_run_pages &&
            dst_addr == run_addr_ + run_pages_ * g6::page_size) {
            run_pages_ += pages;
        } else {
            write_run();
            run_addr_ = dst_addr;
            run_slot_ = slot_;
            run_pages_ = pages;
        }
        slot_ += pages * g6::page_size;
        in_block_ += pages;
        if (in_block_ == block_pages_) {
            write_run();
            {
                DeviceZoneScopedN("G6_BOUNCE_WR_FLUSH");
                // The slot is reusable once its bytes have left L1; completion is awaited once at the end.
                noc_async_writes_flushed();
            }
            cb_pop_front(g6::cb_bounce_pages, g6::bounce_block_pages);
            left_ -= block_pages_;
            in_block_ = 0;
        }
    }

private:
    static constexpr uint32_t max_run_pages =
        g6::bounce_write_run_pages == 0 ? 0xFFFFFFFFu : g6::bounce_write_run_pages;

    FORCE_INLINE void write_run() {
        if (run_pages_ != 0) {
#ifndef G6_ABLATE_WRITE
            noc_async_write(run_slot_, run_addr_, run_pages_ * g6::page_size);
#endif
            run_pages_ = 0;
        }
    }

    uint32_t left_;
    uint32_t in_block_ = 0;
    uint32_t block_pages_ = 0;
    uint32_t slot_ = 0;
    uint64_t run_addr_ = 0;
    uint32_t run_slot_ = 0;
    uint32_t run_pages_ = 0;
};

template <typename Dst>
FORCE_INLINE void bounce_write(const Dst& dst, uint32_t first, uint32_t core_pages, uint32_t rotation) {
    BounceDrain drain(core_pages);
    if constexpr (g6::bounce_order == g6::BOUNCE_STORAGE_ORDER) {
        g6::for_each_storage_page(dst, first, core_pages, [&](const auto& page) {
            drain.room();
            drain.pages(page.noc_addr(), 1);
        });
    } else {
        g6::for_each_page_segment(first, core_pages, rotation, [&](uint32_t begin, uint32_t end) {
            for (uint32_t page = begin; page < end;) {
                const uint32_t block_end = page + drain.room();
                const uint32_t run = target_run_pages(dst, page, block_end < end ? block_end : end);
                drain.pages(dst.get_noc_addr(page), run);
                page += run;
            }
        });
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
        bounce_write(dst, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2));
    }
}
