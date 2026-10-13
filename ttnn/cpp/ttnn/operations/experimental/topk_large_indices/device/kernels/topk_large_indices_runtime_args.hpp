// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace topk_common_args {
// These values are shared by every core running the corresponding kernel.
// valid_end_address: DRAM address of the 1-element real-token-end tensor (0 when uncapped). Like
// metadata_address it is refreshed on every dispatch, so a moved buffer stays addressable.
enum Reader {
    input_address = 0,
    metadata_address = 1,
    search_length = 2,
    input_row_bytes = 3,
    valid_end_address = 4
};
enum Compute { compute_search_length = 0 };
// writer_input_*: only read by the FP32 exact boundary re-rank (k < K) and row rescan (k == K).
// writer_search_length: only read by the FP32 k == K exact row rescan.
enum Writer { output_address = 0, writer_input_address = 1, writer_input_row_bytes = 2, writer_search_length = 3 };
}  // namespace topk_common_args

// FP32 k == K exact row rescan: layout of the rescan CB (offsets from its 64 B aligned base). The writer scans its
// row blocks in its own double buffer; on a core's last row the reader (idle by then) scans the second half of the
// row in its own double buffer and hands its band hits to the writer through the mailbox and hit list.
namespace topk_rescan_layout {
constexpr uint32_t block_elements = 1024;
constexpr uint32_t block_bytes = block_elements * 4;
// Last-row scans check their deadline every this many elements (~1 us), not once per block.
constexpr uint32_t poll_elements = 128;
constexpr uint32_t writer_blocks = 0;
constexpr uint32_t reader_blocks = 2 * block_bytes;
constexpr uint32_t mailbox = 4 * block_bytes;
constexpr uint32_t hits = mailbox + 64;
constexpr uint32_t max_helper_hits = 64;  // (index, ordered image) pairs
constexpr uint32_t cb_bytes = hits + max_helper_hits * 8 + 64;  // +64 B to align the base
// Mailbox words.
enum Word : uint32_t {
    go = 0,  // reader clears it at kernel start; the writer posts go_job or go_skip for the core's last row
    lo = 1,  // band = ordered images in [lo, lo + len)
    len = 2,
    positive = 3,  // band inside [+0, +inf]: raw words compare directly
    first_block = 4,
    end_block = 5,
    deadline = 6,
    row = 7,
    search_len = 8,
    done = 9,  // writer clears it before go_job; reader sets it after writing its hits
    hit_count = 10,
};
constexpr uint32_t go_job = 1;
constexpr uint32_t go_skip = 2;
}  // namespace topk_rescan_layout
