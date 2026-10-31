// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writing a run's partials for one site group, as compute emits them: the group's sites
// together, one position at a time. The partials are most of what the op writes, and on one
// NoC their traffic congests the paths it shares, so the groups alternate between the two
// data-movement cores, each on its own NoC: the writer takes the even groups, the reader the
// odd ones once it has built every group's weights.
//
// Include after the kernel's API headers.

#pragma once

#include <cstdint>

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_mix/device/kernels/attn_res_block_mix_items.hpp"

// The group's sites as compute packs them, out of `cb_obj`. A site's tile at (row, w) is
// page (r * Ht + row) * Wt + w of its tensor, r counted within it: the leading
// `num_closed` sites go to the closed reads and the rest to the partials.
template <uint32_t Ht, uint32_t Wt, uint32_t num_closed, typename Accessor>
inline void write_partials_group(
    Noc& noc,
    CircularBuffer& cb_obj,
    const Accessor& partials,
    const Accessor& closed,
    uint32_t tile_bytes,
    const AttnResBlockMixItem& item,
    uint32_t first_site,
    uint32_t sites) {
    for (uint32_t i = 0; i < item.num_tiles(); ++i) {
        const uint32_t w = item.begin + i;
        cb_obj.wait_front(sites);
        for (uint32_t s = 0; s < sites; ++s) {
            const uint32_t site = first_site + s;
            const bool is_closed = site < num_closed;
            noc.async_write(
                cb_obj,
                is_closed ? closed : partials,
                tile_bytes,
                {.offset_bytes = s * tile_bytes},
                {.page_id = ((is_closed ? site : site - num_closed) * Ht + item.row) * Wt + w});
        }
        // Flushed rather than acknowledged: the buffer only has to be free to reuse.
        noc.async_writes_flushed();
        cb_obj.pop_front(sites);
    }
}

// Which data-movement core writes a group's partials.
constexpr bool reader_writes_group(uint32_t group) { return group % 2 == 1; }
