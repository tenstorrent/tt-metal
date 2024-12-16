// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include <tuple>
// uint32_t nearest_n(uint32_t x, uint32_t n) { return ((x + n - 1) / n) * n; }

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, bool>
get_speculative_runtime_args(
    int cur_pos,
    int cur_batch,
    int core_num,
    int num_cores_per_batch,
    uint32_t k_chunk_size,
    uint32_t speculative_chunk_size,
    uint32_t Spec_chunk_t) {
    /*
    This function divides work along sequence length into chunks of size k_chunk_sizes, where each core takes a
    contiguous set of chunks.
    Note that for speculative flash decode, the first and last speculative_chunk_size are reserved for speculation.
    So work division is from index speculative_chunk_size to cur_pos - speculative_chunk_size.
    */

    uint32_t speculative_height_dim_start_tile_offset1 = 0;
    uint32_t valid_seq_len_total = nearest_n(cur_pos + 1, tt::constants::TILE_HEIGHT);
    uint32_t pst_value_total = valid_seq_len_total / tt::constants::TILE_HEIGHT;
    uint32_t speculative_height_dim_start_tile_offset2 = pst_value_total - Spec_chunk_t;

    uint32_t adjusted_cur_pos_for_spec =
        cur_pos - speculative_height_dim_start_tile_offset2 * tt::constants::TILE_HEIGHT;

    uint32_t non_spec_height_dim_start_tile_offset = Spec_chunk_t;
    uint32_t adjusted_cur_pos_for_non_spec = speculative_height_dim_start_tile_offset2 * tt::constants::TILE_HEIGHT - 1;

    uint32_t non_spec_seq_len = adjusted_cur_pos_for_non_spec - speculative_chunk_size;
    uint32_t valid_seq_len_non_spec = nearest_n(non_spec_seq_len + 1, k_chunk_size);
    uint32_t pst_value_non_spec = (valid_seq_len_non_spec + speculative_chunk_size) /
                                  tt::constants::TILE_HEIGHT;  // from start to adjusted_cur_pos_for_non_spec at the
                                                               // granularity of k_chunk_size in tiles
    uint32_t num_chunks_value = valid_seq_len_non_spec / k_chunk_size;

    uint32_t k_chunk_start = 0;
    uint32_t k_chunk_end = 0;

    if (num_cores_per_batch > int(num_chunks_value)) {
        int chunks_per_core = 1;
        if (core_num >= int(num_chunks_value)) {
            chunks_per_core = 0;
        }
        k_chunk_start = (num_chunks_value - core_num - 1) * chunks_per_core;
        k_chunk_end = (num_chunks_value - core_num) * chunks_per_core;
    } else {
        int chunks_per_core = num_chunks_value / num_cores_per_batch;
        int residuals = num_chunks_value % num_cores_per_batch;
        int reversed_core_num = num_cores_per_batch - core_num - 1;
        k_chunk_start = reversed_core_num * chunks_per_core + std::min(residuals, reversed_core_num);
        k_chunk_end = k_chunk_start + chunks_per_core;
        if (reversed_core_num < residuals) {
            k_chunk_end += 1;
        }
    }

    bool do_speculative_compute = (core_num == 0);  // let reducer core do speculative compute for now. TODO: improve
                                                    // parallelization by offloading the work to more idle cores.
    return {
        num_chunks_value,
        k_chunk_start,
        k_chunk_end,
        speculative_height_dim_start_tile_offset1,
        speculative_height_dim_start_tile_offset2,
        non_spec_height_dim_start_tile_offset,
        adjusted_cur_pos_for_non_spec,
        adjusted_cur_pos_for_spec,
        do_speculative_compute};
}
