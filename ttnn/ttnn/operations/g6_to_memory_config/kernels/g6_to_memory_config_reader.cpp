// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config pull mover and bounce reader: every NoC read lands in this core's own L1.
//   WALK_SHARD  pull_shard  read each page of the owned target shard straight into its final slot
//   WALK_BANK   pull_bank   read each page of this core's target L1 bank stripe into its final slot
//   WALK_BOUNCE bounce      read bounce_block_pages pages per block into cb_bounce_pages for the writer, in
//                           page order or in the target's storage order (bounce_order)
// No kernel_lib helper applies: the op moves whole pages as bytes (no compute phase), and the
// self-aimed local-copy helpers cannot read a remote source.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "g6_to_memory_config_common.hpp"

namespace {

template <typename Src, typename Dst>
FORCE_INLINE void pull_shard(const Src& src, const Dst& dst, uint32_t owned_shard_id) {
    if (owned_shard_id == g6::NO_SHARD) {
        return;
    }
    ASSERT(dst.is_local_shard(owned_shard_id));
    const g6::Slice slice = g6::mover_slice(dst.dspec().shard_volume());
    if (slice.begin == slice.end) {
        return;
    }
    {
        DeviceZoneScopedN("G6_PULL_ISSUE");
        // shard_pages skips the slots of a ragged last shard that lie past the tensor.
        for (const auto& page : dst.shard_pages(owned_shard_id, slice.begin, slice.end)) {
            noc_async_read(src.get_noc_addr(page.page_id()), g6::local_slot(page.noc_addr()), g6::page_size);
        }
    }
    {
        DeviceZoneScopedN("G6_PULL_DRAIN");
        noc_async_read_barrier();
    }
}

template <typename Src, typename Dst>
FORCE_INLINE void pull_bank(const Src& src, const Dst& dst) {
    uint32_t bank;
    {
        DeviceZoneScopedN("G6_OWN_SCAN");
        bank = g6::find_own_bank(dst);
    }
    if (bank == g6::NO_BANK) {
        return;
    }
    constexpr uint32_t stride = g6::movers_per_owner * NUM_L1_BANKS;
    {
        DeviceZoneScopedN("G6_PULL_ISSUE");
        for (uint32_t page = bank + g6::mover_index * NUM_L1_BANKS; page < g6::num_pages; page += stride) {
            noc_async_read(src.get_noc_addr(page), g6::local_slot(dst.get_noc_addr(page)), g6::page_size);
        }
    }
    {
        DeviceZoneScopedN("G6_PULL_DRAIN");
        noc_async_read_barrier();
    }
}

// Fills cb_bounce_pages block by block: reserve the nominal block, read its pages, one barrier, push.
// The nominal count is reserved and pushed even for the short last block, so slots stay aligned to the ring.
class BounceFill {
public:
    explicit BounceFill(uint32_t core_pages) : left_(core_pages) {}

    // CB slot of the next page; opens a block when none is open.
    FORCE_INLINE uint32_t next_slot() {
        if (in_block_ == 0) {
            block_pages_ = left_ < g6::bounce_block_pages ? left_ : g6::bounce_block_pages;
            {
                DeviceZoneScopedN("G6_BOUNCE_RD_WAIT_FREE");
                cb_reserve_back(g6::cb_bounce_pages, g6::bounce_block_pages);
            }
            slot_ = get_write_ptr(g6::cb_bounce_pages);
        }
        const uint32_t slot = slot_;
        slot_ += g6::page_size;
        return slot;
    }

    // Closes the block once its last read is issued.
    FORCE_INLINE void page_issued() {
        if (++in_block_ == block_pages_) {
            {
                DeviceZoneScopedN("G6_BOUNCE_RD_DRAIN");
                noc_async_read_barrier();
            }
            cb_push_back(g6::cb_bounce_pages, g6::bounce_block_pages);
            left_ -= block_pages_;
            in_block_ = 0;
        }
    }

private:
    uint32_t left_;
    uint32_t in_block_ = 0;
    uint32_t block_pages_ = 0;
    uint32_t slot_ = 0;
};

template <typename Src, typename Dst>
FORCE_INLINE void bounce_read(const Src& src, const Dst& dst, uint32_t first, uint32_t core_pages, uint32_t rotation) {
    BounceFill fill(core_pages);
    auto read_page = [&](uint32_t page_id) {
#ifndef G6_ABLATE_READ
        noc_async_read(src.get_noc_addr(page_id), fill.next_slot(), g6::page_size);
#else
        fill.next_slot();
#endif
        fill.page_issued();
    };
    if constexpr (g6::bounce_order == g6::BOUNCE_STORAGE_ORDER) {
        g6::for_each_storage_page(dst, first, core_pages, [&](const auto& page) { read_page(page.page_id()); });
    } else {
        g6::for_each_page_segment(first, core_pages, rotation, [&](uint32_t begin, uint32_t end) {
            for (uint32_t page_id = begin; page_id < end; ++page_id) {
                read_page(page_id);
            }
        });
    }
}

}  // namespace

void kernel_main() {
    const uint32_t src_base = get_common_arg_val<uint32_t>(0);
    const uint32_t dst_base = get_common_arg_val<uint32_t>(1);
    [[maybe_unused]] const auto src = TensorAccessor(g6::src_args, src_base, g6::page_size);
    [[maybe_unused]] const auto dst = TensorAccessor(g6::dst_args, dst_base, g6::page_size);

    if constexpr (g6::walk == g6::WALK_SHARD) {
        pull_shard(src, dst, get_arg_val<uint32_t>(0));
    } else if constexpr (g6::walk == g6::WALK_BANK) {
        pull_bank(src, dst);
    } else {
        bounce_read(src, dst, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1), get_arg_val<uint32_t>(2));
    }
}
