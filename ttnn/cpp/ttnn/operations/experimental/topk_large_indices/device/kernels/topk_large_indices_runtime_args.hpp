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
    valid_end_address = 4,
    // Two-level programs only. A row runs two-level when its search length is at least the cutover; 0 forces
    // two-level, 0xFFFFFFFF forces single-level. block_max_address is the maxima tensor stage 2 reads (the
    // caller's, or the program's own that phase 1 writes); two_level_max_row_bytes its row pitch. The coordinator
    // (core index 0) hands out the phase-1 units.
    two_level_cutover = 5,
    block_max_address = 6,
    two_level_max_row_bytes = 7,
    two_level_num_input_rows = 8,
    two_level_num_cores = 9,
    reader_coordinator_noc_x = 10,
    reader_coordinator_noc_y = 11,
    // The rows halved across core pairs (see topk_two_level::row_owner); 0 without pairs.
    two_level_split_rows = 12
};
enum Compute { compute_search_length = 0, compute_two_level_num_input_rows = 1 };
// Two-level programs only: two_level_max_address is the maxima tensor phase 1 writes, mode_output_address a test
// output for the mode the program ran in (0 = none). writer_two_level_core_coords starts one word per core, in core
// index order: its virtual NoC x | y << 16 (a row's owner: topk_two_level::row_owner).
enum Writer {
    output_address = 0,
    two_level_max_address = 1,
    mode_output_address = 2,
    writer_two_level_max_row_bytes = 3,
    writer_two_level_num_input_rows = 4,
    writer_two_level_num_cores = 5,
    writer_two_level_split_rows = 6,
    writer_two_level_core_coords = 7
};
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
    // Two-level programs only (the split slots above are always present then): this core's index, which names its
    // rows.
    reader_two_level_core_index = 5,
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
