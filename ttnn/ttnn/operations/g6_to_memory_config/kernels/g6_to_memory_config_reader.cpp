// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// g6_to_memory_config pull mover and bounce reader: every NoC read lands in this core's own L1.
//   WALK_SHARD  pull_shard  read each page of the owned target shard straight into its final slot
//   WALK_BANK   pull_bank   read each page of this core's target L1 bank stripe into its final slot
//   WALK_BOUNCE bounce      read bounce_block_pages pages per block into cb_bounce_pages for the writer, in
//                           page order or in the target's storage order (bounce_order); one read per page or per
//                           run contiguous in the source (bounce_read_coalesce)
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

// Visits the runs of source pages contiguous in memory that cover [begin, end), residue by residue of the source's
// contiguity stride: fn(first_page, run_pages, staged), where the run is pages first_page + k * stride, k < run_pages,
// and staged counts the pages of earlier runs. An interleaved source has one run per bank.
template <typename Src, typename Fn>
FORCE_INLINE void for_each_source_run(const Src& src, uint32_t begin, uint32_t end, uint32_t stride, Fn&& fn) {
    uint32_t staged = 0;
    for (uint32_t first = begin; first < end && first < begin + stride; ++first) {
        for (uint32_t page = first; page < end;) {
            const uint32_t run = src.num_contiguous_pages(page, end);
            fn(page, run, staged);
            staged += run;
            page += run * stride;
        }
    }
}

// One NoC read per page of [begin, end), page p to dst + (p - begin) * page_size.
template <typename Src>
FORCE_INLINE void read_each_page(const Src& src, uint32_t begin, uint32_t end, uint32_t dst) {
    for (uint32_t page = begin; page < end; ++page, dst += g6::page_size) {
#ifndef G6_ABLATE_READ
        noc_async_read(src.get_noc_addr(page), dst, g6::page_size);
#endif
    }
}

// Reads source pages [begin, end) to L1, page p at dst + (p - begin) * page_size. With bounce_read_coalesce one read
// covers each run contiguous in the source: a stride-1 source lands in place; a strided source (interleaved:
// consecutive pages sit in different banks) lands run by run in cb_bounce_stage, bounce_stage_pages at a time, and
// self-aimed reads put it in page order, so the DRAM sees a few long reads instead of one per page and the writer
// still gets page order.
template <typename Src>
FORCE_INLINE void read_pages(const Src& src, uint32_t begin, uint32_t end, uint32_t dst) {
    if constexpr (!g6::bounce_read_coalesce) {
        read_each_page(src, begin, end, dst);
    } else {
        // A single page is its own run.
        const uint32_t stride = end - begin == 1 ? 1 : src.contiguous_page_stride();
        if (stride != 1 && g6::bounce_stage_pages == 0) {
            read_each_page(src, begin, end, dst);
            return;
        }
        if (stride == 1) {
            for (uint32_t page = begin; page < end;) {
                const uint32_t run = src.num_contiguous_pages(page, end);
#ifndef G6_ABLATE_READ
                noc_async_read(src.get_noc_addr(page), dst + (page - begin) * g6::page_size, run * g6::page_size);
#endif
                page += run;
            }
            return;
        }
        static_assert(g6::page_size <= NOC_MAX_BURST_SIZE, "the L1 reorder moves one page per packet");
        const uint32_t stage = get_write_ptr(g6::cb_bounce_stage);
        for (uint32_t chunk = begin; chunk < end; chunk += g6::bounce_stage_pages) {
            const uint32_t chunk_end = end - chunk < g6::bounce_stage_pages ? end : chunk + g6::bounce_stage_pages;
            for_each_source_run(src, chunk, chunk_end, stride, [&](uint32_t page, uint32_t run, uint32_t staged) {
#ifndef G6_ABLATE_READ
                noc_async_read(src.get_noc_addr(page), stage + staged * g6::page_size, run * g6::page_size);
#endif
            });
            noc_async_read_barrier();
#ifndef G6_ABLATE_REORDER
            // The DRAM reads above reprogrammed the read command buffer, so the self-aimed state is set per chunk.
            noc_async_read_one_packet_set_state(get_noc_addr(stage), g6::page_size);
            for_each_source_run(src, chunk, chunk_end, stride, [&](uint32_t page, uint32_t run, uint32_t staged) {
                uint32_t from = stage + staged * g6::page_size;
                uint32_t to = dst + (page - begin) * g6::page_size;
                for (uint32_t k = 0; k < run; ++k, from += g6::page_size, to += stride * g6::page_size) {
                    noc_async_read_one_packet_with_state(from, to);
                }
            });
            // The next chunk overwrites the stage.
            noc_async_read_barrier();
#endif
        }
    }
}

// Fills cb_bounce_pages block by block: reserve the nominal block, read its pages, one barrier, push. The walk hands
// over ranges of page ids; a range that crosses a block edge is split there. The nominal count is reserved and
// pushed even for the short last block, so slots stay aligned to the ring.
class BounceFill {
public:
    explicit BounceFill(uint32_t core_pages) : left_(core_pages) {}

    template <typename Src>
    FORCE_INLINE void pages(const Src& src, uint32_t begin, uint32_t end) {
        while (begin < end) {
            if (in_block_ == 0) {
                block_pages_ = left_ < g6::bounce_block_pages ? left_ : g6::bounce_block_pages;
                {
                    DeviceZoneScopedN("G6_BOUNCE_RD_WAIT_FREE");
                    cb_reserve_back(g6::cb_bounce_pages, g6::bounce_block_pages);
                }
                slot_ = get_write_ptr(g6::cb_bounce_pages);
            }
            const uint32_t room = block_pages_ - in_block_;
            const uint32_t n = end - begin < room ? end - begin : room;
            read_pages(src, begin, begin + n, slot_);
            slot_ += n * g6::page_size;
            in_block_ += n;
            begin += n;
            if (in_block_ == block_pages_) {
                {
                    DeviceZoneScopedN("G6_BOUNCE_RD_DRAIN");
                    noc_async_read_barrier();
                }
                cb_push_back(g6::cb_bounce_pages, g6::bounce_block_pages);
                left_ -= block_pages_;
                in_block_ = 0;
            }
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
    if constexpr (g6::bounce_order == g6::BOUNCE_STORAGE_ORDER) {
        g6::for_each_storage_page(
            dst, first, core_pages, [&](const auto& page) { fill.pages(src, page.page_id(), page.page_id() + 1); });
    } else {
        g6::for_each_page_segment(
            first, core_pages, rotation, [&](uint32_t begin, uint32_t end) { fill.pages(src, begin, end); });
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
