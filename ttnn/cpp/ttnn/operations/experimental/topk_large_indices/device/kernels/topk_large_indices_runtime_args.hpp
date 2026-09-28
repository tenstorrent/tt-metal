// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace topk_common_args {
// These values are shared by every core running the corresponding kernel.
enum Reader { input_address = 0, metadata_address = 1, search_length = 2, input_row_bytes = 3 };
enum Compute { compute_search_length = 0 };
enum Writer { output_address = 0 };
}  // namespace topk_common_args

// Per-core argument layout. A row is split into column segments of whole K chunks; the segment's
// survivors are merged across cores in a binary tree whose round r pairs segment i with i + 2^r.
namespace topk_core_args {
enum Reader { reader_start_row = 0, reader_num_rows = 1, reader_seg_first_chunk = 2, reader_seg_end_chunk = 3 };
enum Compute {
    compute_num_rows = 0,
    compute_seg_first_chunk = 1,
    compute_seg_end_chunk = 2,
    compute_num_recv_rounds = 3,
    compute_sends_survivor = 4,
    compute_num_args = 5,
};
enum Writer {
    writer_start_row = 0,
    writer_num_rows = 1,
    writer_num_recv_rounds = 2,
    writer_sends_survivor = 3,
    writer_parent_noc_x = 4,
    writer_parent_noc_y = 5,
    // (x, y) of the round r child at writer_child_coords_base + 2r; zero for rounds this core does not receive in.
    writer_child_coords_base = 6,
};
}  // namespace topk_core_args
