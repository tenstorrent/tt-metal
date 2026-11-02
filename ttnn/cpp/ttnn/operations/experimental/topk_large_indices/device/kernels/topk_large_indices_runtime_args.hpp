// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
enum Writer { output_address = 0 };
}  // namespace topk_common_args

// Per-core args. A core reduces num_rows rows: whole rows from start_row and, with has_split, last one half of
// split_row's K chunks. The two halves of a split row sit on a core pair; the first half receives the second
// half's survivor, merges it, and writes the row.
namespace topk_core_args {
// second_half says which half of split_row this core owns. Where the halves meet is NOT sent: the kernels cut the
// row at its own chunk count, which under a metadata length only the device knows. The second half sends its
// survivor, the first receives and writes the row, so those roles follow from this one bit too.
enum Reader {
    reader_start_row = 0,
    reader_num_rows = 1,
    reader_split_row = 2,
    reader_has_split = 3,
    reader_second_half = 4,
};
enum Compute {
    compute_num_rows = 0,
    compute_has_split = 1,
    compute_second_half = 2,
};
enum Writer {
    writer_start_row = 0,
    writer_num_rows = 1,
    writer_has_split = 2,
    writer_second_half = 3,
    // Virtual (translated) worker NoC coordinates of the other core in this row's pair.
    writer_partner_noc_x = 4,
    writer_partner_noc_y = 5,
    writer_split_row = 6,
};
}  // namespace topk_core_args
