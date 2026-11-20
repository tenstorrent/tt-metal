// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Helpers shared by the interleaved concat readers.

#pragma once

#include <stdint.h>
#include <type_traits>
#include <utility>

#include "internal/risc_attribs.h"

namespace concat_reader {

// Calls f(std::integral_constant<uint32_t, I>{}) for the input I == tensor_idx, an index known only at
// run time. The caller builds input I's accessor inside f, so only that input's accessor is on the stack
// and stack use does not grow with the number of inputs.
template <typename F, uint32_t... Is>
FORCE_INLINE void with_input(uint32_t tensor_idx, std::integer_sequence<uint32_t, Is...>, F&& f) {
    (void)((tensor_idx == Is && (f(std::integral_constant<uint32_t, Is>{}), true)) || ...);
}

// Walks the inputs round-robin, one block of pages_per_block pages from each input, starting at page
// start_tensor_id of input start_tensor. The host's first page id for input t already accounts for where
// this core starts: for t == start_tensor it includes start_tensor_id, and for t < start_tensor it
// points to the next block.
struct RoundRobinCursor {
    uint32_t start_tensor;
    uint32_t start_tensor_id;
    uint32_t tensor;       // current input
    uint32_t id_in_block;  // next page of the current input's block
    uint32_t num_wraps;    // times tensor wrapped from the last input back to input 0

    FORCE_INLINE RoundRobinCursor(uint32_t start_tensor, uint32_t start_tensor_id) :
        start_tensor(start_tensor),
        start_tensor_id(start_tensor_id),
        tensor(start_tensor),
        id_in_block(start_tensor_id),
        num_wraps(0) {}

    // Pages of the current input this core has already read: one block per earlier visit, less the part
    // of the first block skipped by starting mid-block. Inputs before start_tensor are first visited
    // after the first wrap.
    FORCE_INLINE uint32_t pages_already_read(uint32_t pages_per_block) const {
        uint32_t pages = num_wraps * pages_per_block + id_in_block;
        if (tensor < start_tensor) {
            pages -= pages_per_block;
        } else if (tensor == start_tensor) {
            pages -= start_tensor_id;
        }
        return pages;
    }

    // Moves past num_pages pages of the current input's block, then to the next input at the block's end.
    FORCE_INLINE void advance(uint32_t num_pages, uint32_t pages_per_block, uint32_t num_tensors) {
        id_in_block += num_pages;
        if (id_in_block == pages_per_block) {
            id_in_block = 0;
            if (++tensor == num_tensors) {
                tensor = 0;
                num_wraps++;
            }
        }
    }
};

}  // namespace concat_reader
