// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// How a forwarding chunk is named, shared by combine_fabric2d and by the routed expert's overlapped fork.
// Its own header and its own namespace because both ops' kernel interfaces re-export it: the host that
// emits a chunk and the kernel that reads it back must agree on one layout, not two identical copies.

#pragma once

#include <cstdint>

#ifndef KERNEL_BUILD
#include <vector>
#endif

namespace ttnn::operations::experimental::deepseek_prefill::combine_chunk {

// Words per chunk descriptor.
constexpr uint32_t CHUNK_WORDS = 4;

// One chunk of a stream's forwarding region: whose tokens it carries, for which chip, and the share of each
// run those two chips agreed on. Enough to compute the chunk's token count, and so its page range once every
// chunk before it in the region has been counted too.
//
// Packed by position into the reader's compile-time args. to_words below is the only place that order is
// written down, and from_words mirrors it, so the host that emits a chunk and the kernel that reads it back
// cannot drift apart.
struct ChunkDescriptor {
    uint32_t origin_dg_index = 0;
    uint32_t dst_dg_index = 0;
    uint32_t split_idx = 0;
    uint32_t split_count = 1;

    void to_words(uint32_t* words) const {
        words[0] = origin_dg_index;
        words[1] = dst_dg_index;
        words[2] = split_idx;
        words[3] = split_count;
    }

    static ChunkDescriptor from_words(const uint32_t* words) {
        return ChunkDescriptor{words[0], words[1], words[2], words[3]};
    }

#ifndef KERNEL_BUILD
    void append_to(std::vector<uint32_t>& out) const {
        uint32_t words[CHUNK_WORDS];
        to_words(words);
        out.insert(out.end(), words, words + CHUNK_WORDS);
    }
#endif
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::combine_chunk
