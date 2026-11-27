// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config pull mover and bounce reader: every NoC read lands in this core's own L1.
//   WALK_SHARD  pull_shard  read each page of the owned target shard straight into its final slot
//   WALK_BANK   pull_bank   read each page of this core's target L1 bank stripe into its final slot
//   WALK_BOUNCE bounce      read bounce_block_pages pages per block into cb_bounce_pages for the writer
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

template <typename Src>
FORCE_INLINE void bounce_read(const Src& src, uint32_t start_page, uint32_t core_pages) {
    for (uint32_t done = 0; done < core_pages;) {
        const uint32_t remaining = core_pages - done;
        const uint32_t block_pages = remaining < g6::bounce_block_pages ? remaining : g6::bounce_block_pages;
        {
            DeviceZoneScopedN("G6_BOUNCE_RD_WAIT_FREE");
            // Nominal count even for the short last block: slots stay aligned to the CB ring.
            cb_reserve_back(g6::cb_bounce_pages, g6::bounce_block_pages);
        }
        {
            DeviceZoneScopedN("G6_BOUNCE_RD_ISSUE");
            uint32_t slot = get_write_ptr(g6::cb_bounce_pages);
            const uint32_t first = start_page + done;
            for (uint32_t i = 0; i < block_pages; ++i) {
#ifndef G6_ABLATE_READ
                noc_async_read(src.get_noc_addr(first + i), slot, g6::page_size);
#endif
                slot += g6::page_size;
            }
        }
        {
            DeviceZoneScopedN("G6_BOUNCE_RD_DRAIN");
            noc_async_read_barrier();
        }
        cb_push_back(g6::cb_bounce_pages, g6::bounce_block_pages);
        done += block_pages;
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
        bounce_read(src, get_arg_val<uint32_t>(0), get_arg_val<uint32_t>(1));
    }
}
